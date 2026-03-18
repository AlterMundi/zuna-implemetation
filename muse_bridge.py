"""
Muse Bridge — EEG-driven parameter modulation for the Harmonic Shaper

Reads live Muse 2 EEG via OSC and modulates harmonic_shaper parameters
within bounded limits around the user-set base values.

Every input is independent and optional.  If no Muse 2 is streaming,
nothing happens.  If no hr_relay is running, there is no heartbeat
pulse.  If no Launchpad slider sends /bridge/gain_depth, gain depth
stays at its initial CLI value.  Mix and match freely.

Parameter modes (--param):

  gain   — Spectral tilt: alpha/beta ratio tilts the gain curve.
           Relaxation boosts lower harmonics, focus boosts upper.

  phase  — Per-band rotation: each sensor's dominant band power controls
           the rotation speed of its matched harmonic's phase.
           Stronger brain activity = faster phase drift.
           H1 stays anchored. The cymatic pattern evolves continuously.
           Phase output is interpolated at --osc-rate (default 30 Hz) for
           smooth, jitter-free cymatic movement between EEG analysis ticks.

  both   — Phase rotation + gain tilt simultaneously.
           Gain depth is controlled by an external slider via
           /bridge/gain_depth OSC (e.g. Launchpad mod wheel through
           midi_relay.py). Slider at 0 = no gain modulation.
           Slider at max = full EEG influence on gains.

Heartbeat pulse (--pulse, optional):
  When hr_relay.py sends /bridge/heartbeat or /bridge/heartbeat_bpm,
  a short exponential gain envelope fires on each beat, creating a
  visible "breathing" in the cymatic pattern.  Works in any param mode.
  Set --pulse 0 to disable.  Runs at osc_rate for smooth rendering.

Session configurations:

  # 1. Muse-only: phase control
  python muse_bridge.py --param phase --depth 30

  # 2. Muse-only: gain tilt
  python muse_bridge.py --param gain --depth 0.20

  # 3. Muse phase + Launchpad gain (slider controls EEG gain depth)
  python midi_relay.py --target-port 5000 &
  python muse_bridge.py --param both --depth 30

  # 4. Muse phase + Fitbit gain pulse (no EEG gain, heartbeat only)
  python hr_relay.py --mode simulate --bpm 72 &
  python muse_bridge.py --param phase --depth 30

  # 5. All three: Muse phase + Launchpad gain + Fitbit pulse
  python midi_relay.py --target-port 5000 &
  python hr_relay.py --mode ble &
  python muse_bridge.py --param both --depth 30

  # 6. Fitbit pulse only (no Muse, no Launchpad)
  python hr_relay.py --mode simulate --bpm 72 &
  python muse_bridge.py --param phase --depth 0

  # 7. Test without any hardware
  python simulate_eeg.py &
  python hr_relay.py --mode simulate --bpm 72 &
  python muse_bridge.py --param both --depth 30

Base values are fetched from harmonic_shaper's HTTP API on startup.
On exit, all modified parameters are restored to their base values.
"""

import argparse
import json
import math
import signal
import sys
import time
import threading
import urllib.request
from pathlib import Path

import numpy as np
from pythonosc import dispatcher, osc_server, udp_client

from osc_playback import compute_band_powers, BANDS

# ─────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
REC = CONFIG.get("recorder", {})
MB = CONFIG.get("muse_bridge", {})

CHANNELS = REC.get("channels", ["TP9", "AF7", "AF8", "TP10"])
SAMPLING_RATE = REC.get("sampling_rate", 256)

TILT_WEIGHTS = MB.get("tilt_weights", [-0.8, -0.4, 0.0, 0.4, 0.8])

DEFAULT_BASE_GAINS = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8}
DEFAULT_BASE_PHASES = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}

# Per-band phase rotation: which sensor/band drives which harmonic
PHASE_SENSOR_MAP = {
    2: {"channel": "TP9",  "band": "theta", "label": "L-temp theta"},
    3: {"channel": "AF7",  "band": "alpha", "label": "L-front alpha"},
    4: {"channel": "AF8",  "band": "beta",  "label": "R-front beta"},
    5: {"channel": "TP10", "band": "gamma", "label": "R-temp gamma"},
}


# ─────────────────────────────────────────────
# Shaper API client
# ─────────────────────────────────────────────

def fetch_shaper_state(api_base):
    """GET harmonic_shaper state to read current base gains and phases."""
    try:
        resp = urllib.request.urlopen(f"{api_base}/api/state", timeout=3)
        data = json.loads(resp.read())
        gains = {}
        phases = {}
        for k, v in data.get("voices", {}).items():
            n = int(k)
            gains[n] = v.get("gain", 0.8)
            phases[n] = v.get("phase_deg", 0.0)
        return gains, phases
    except Exception as e:
        print(f"  WARNING: Could not fetch shaper state ({e}), using defaults")
        return None, None


# ─────────────────────────────────────────────
# Muse Bridge
# ─────────────────────────────────────────────

class MuseBridge:
    """Modulates harmonic_shaper parameters from Muse 2 EEG."""

    def __init__(self, osc_out, shaper_api, param_mode, phase_depth, gain_depth,
                 update_rate, osc_rate=30.0, window_seconds=1.0,
                 smoothing_alpha=0.25, pulse_amplitude=0.15):
        self.osc_out = osc_out
        self.shaper_api = shaper_api
        self.param_mode = param_mode
        self.phase_depth = phase_depth   # max deg/s for phase rotation
        self.gain_depth = gain_depth     # fraction for gain tilt (0-1)
        self.update_rate = update_rate
        self.update_interval = 1.0 / update_rate
        self.osc_rate = osc_rate
        self.osc_interval = 1.0 / osc_rate
        self.smoothing_alpha = smoothing_alpha
        self.window_size = int(window_seconds * SAMPLING_RATE)

        # Ring buffer per channel
        self.buffers = {ch: np.zeros(self.window_size) for ch in CHANNELS}
        self.write_pos = 0
        self.samples_received = 0
        self.running = False

        # Signal quality
        self.contact_quality = {ch: 4.0 for ch in CHANNELS}

        # Base values (fetched from shaper)
        self.base_gains = dict(DEFAULT_BASE_GAINS)
        self.base_phases = dict(DEFAULT_BASE_PHASES)
        self.active_harmonics = list(range(1, 6))

        # Gain tilt state
        self.tilt_smooth = 0.0
        self.last_gains_sent = {}
        self.current_tilt_gains = {}

        # Phase rotation state
        self.phase_accumulators = {n: 0.0 for n in range(1, 6)}
        self.phase_velocities = {n: 0.0 for n in range(1, 6)}
        self.power_history = {ch: [] for ch in CHANNELS}
        self.last_phases_sent = {}

        # Heartbeat state
        self.pulse_amplitude = pulse_amplitude
        self.current_bpm = 0.0
        self.beat_phase = 999.0          # seconds since last beat (high = no pulse)
        self.beat_triggered = False      # True = real beats from BLE
        self.last_envelope = 0.0

        self._lock = threading.Lock()
        self.updates_sent = 0
        self.osc_sends = 0

    # ─── Base Value Management ───

    def refresh_base_values(self):
        """Fetch current state from harmonic_shaper as the new base."""
        gains, phases = fetch_shaper_state(self.shaper_api)
        if gains:
            self.base_gains = gains
            self.active_harmonics = sorted(gains.keys())
            if not self.active_harmonics:
                self.active_harmonics = list(range(1, 6))
        if phases:
            self.base_phases = phases

        if self.param_mode in ("gain", "both"):
            print(f"  Base gains:  {self._format_gains()}")
        if self.param_mode in ("phase", "both"):
            print(f"  Base phases: {self._format_phases()}")

    def _format_gains(self):
        return " ".join(
            f"H{n}={self.base_gains.get(n, 0.8):.2f}"
            for n in self.active_harmonics[:5]
        )

    def _format_phases(self):
        return " ".join(
            f"H{n}={self.base_phases.get(n, 0.0):.0f}\u00b0"
            for n in self.active_harmonics[:5]
        )

    # ─── OSC Input ───

    def eeg_handler(self, address, *args):
        if not self.running:
            return
        with self._lock:
            for i, ch in enumerate(CHANNELS):
                if i < len(args):
                    self.buffers[ch][self.write_pos % self.window_size] = float(args[i])
            self.write_pos += 1
            self.samples_received += 1

    def horseshoe_handler(self, address, *args):
        with self._lock:
            for i, ch in enumerate(CHANNELS):
                if i < len(args):
                    self.contact_quality[ch] = float(args[i])

    def gain_depth_handler(self, address, *args):
        """Receive live gain depth from external controller (e.g. Launchpad slider).

        Accepts 0-127 (MIDI range, auto-normalized) or 0.0-1.0.
        """
        if not args:
            return
        raw = float(args[0])
        if raw > 1.0:
            raw = raw / 127.0
        self.gain_depth = max(0.0, min(1.0, raw))

    def heartbeat_handler(self, address, *args):
        """Beat event from BLE or simulated source. Args: [bpm] or [bpm, rr_ms]."""
        if args:
            self.current_bpm = max(0.0, float(args[0]))
        self.beat_phase = 0.0
        self.beat_triggered = True

    def heartbeat_bpm_handler(self, address, *args):
        """BPM-only update from Fitbit Web API. Args: [bpm].

        Triggers local beat synthesis — the bridge generates the beat
        clock internally from the BPM value.
        """
        if args:
            self.current_bpm = max(0.0, float(args[0]))
        self.beat_triggered = False

    # ─── Analysis Helpers ───

    def _get_channel_window(self, ch):
        with self._lock:
            return np.roll(self.buffers[ch], -self.write_pos % self.window_size).copy()

    def _channel_ok(self, ch):
        with self._lock:
            return self.contact_quality.get(ch, 4.0) < 4.0

    def _has_good_frontal(self):
        return self._channel_ok("AF7") or self._channel_ok("AF8")

    def _has_any_good_channel(self):
        return any(self._channel_ok(ch) for ch in CHANNELS)

    def _compute_frontal_powers(self):
        """Average alpha and beta power from frontal channels with good contact."""
        alpha_total = 0.0
        beta_total = 0.0
        count = 0
        for ch in ["AF7", "AF8"]:
            if not self._channel_ok(ch):
                continue
            window = self._get_channel_window(ch)
            powers = compute_band_powers(window, SAMPLING_RATE)
            alpha_total += powers.get("alpha", 0.0)
            beta_total += powers.get("beta", 0.0)
            count += 1
        if count == 0:
            return 0.0, 0.0
        return alpha_total / count, beta_total / count

    def _get_band_power(self, ch, band):
        """Get power for a specific band from a channel."""
        window = self._get_channel_window(ch)
        powers = compute_band_powers(window, SAMPLING_RATE)
        return powers.get(band, 0.0)

    def _update_power_range(self, ch, power):
        """Adaptive normalization using rolling percentiles."""
        hist = self.power_history[ch]
        hist.append(power)
        if len(hist) > 50:
            hist.pop(0)
        if len(hist) > 5:
            return float(np.percentile(hist, 10)), float(np.percentile(hist, 90))
        return 0.0, max(power, 1e-10)

    # ─── Gain Tilt Mode ───

    def compute_tilt(self):
        """Alpha/beta ratio: +1 = relaxed, -1 = focused."""
        alpha, beta = self._compute_frontal_powers()
        total = alpha + beta
        if total < 1e-10:
            return 0.0
        return float((alpha - beta) / total)

    def compute_gain_modulation(self):
        raw_tilt = self.compute_tilt()
        self.tilt_smooth += (raw_tilt - self.tilt_smooth) * self.smoothing_alpha

        gains = {}
        harmonics = self.active_harmonics[:5]
        weights = TILT_WEIGHTS[:len(harmonics)]
        for i, n in enumerate(harmonics):
            w = weights[i] if i < len(weights) else 0.0
            modulator = float(np.clip(self.tilt_smooth * w, -1.0, 1.0))
            base = self.base_gains.get(n, 0.8)
            effective = base * (1.0 + self.gain_depth * modulator)
            gains[n] = float(np.clip(effective, 0.0, 1.0))
        return gains

    # ─── Phase Rotation Mode ───

    def analyze_phase_velocities(self):
        """EEG analysis step (slow, runs at update_rate).

        Reads band power from each sensor and updates the target velocity
        for each harmonic. Does NOT advance accumulators — that happens in
        the fast output tick.
        """
        for n, cfg in PHASE_SENSOR_MAP.items():
            ch = cfg["channel"]
            band = cfg["band"]

            if not self._channel_ok(ch):
                self.phase_velocities[n] *= (1.0 - self.smoothing_alpha)
                continue

            power = self._get_band_power(ch, band)
            p_low, p_high = self._update_power_range(ch, power)

            if p_high > p_low:
                normalized = (power - p_low) / (p_high - p_low)
            else:
                normalized = 0.0
            normalized = float(np.clip(normalized, 0.0, 1.0))

            target_velocity = normalized * self.phase_depth
            self.phase_velocities[n] += (target_velocity - self.phase_velocities[n]) * self.smoothing_alpha

    def advance_phases(self, dt):
        """Output step (fast, runs at osc_rate).

        Advances accumulators using the current velocities and returns
        the absolute phase for each harmonic.
        """
        phases = {}
        phases[1] = self.base_phases.get(1, 0.0)

        for n in PHASE_SENSOR_MAP:
            self.phase_accumulators[n] += self.phase_velocities[n] * dt
            self.phase_accumulators[n] %= 360
            base = self.base_phases.get(n, 0.0)
            phases[n] = (base + self.phase_accumulators[n]) % 360

        return phases

    # ─── Heartbeat Pulse ───

    def compute_heartbeat_envelope(self, dt):
        """Advance beat phase and return current envelope amplitude.

        In beat-triggered mode (BLE/simulate), beat_phase is reset to 0
        by the OSC handler on each beat.
        In BPM-synthesized mode (Web API), the local clock triggers beats.
        """
        if self.current_bpm < 1.0 or self.pulse_amplitude < 0.001:
            return 0.0

        self.beat_phase += dt

        if not self.beat_triggered:
            beat_interval = 60.0 / self.current_bpm
            if self.beat_phase >= beat_interval:
                self.beat_phase %= beat_interval

        # Auto-scale decay so pulses don't overlap
        decay_tau = 0.3 * (60.0 / self.current_bpm)
        self.last_envelope = self.pulse_amplitude * math.exp(-self.beat_phase / decay_tau)
        return self.last_envelope

    def apply_heartbeat_pulse(self, gains, dt):
        """Apply heartbeat gain pulse on top of existing gains.

        Returns (pulsed_gains, envelope_value).
        """
        envelope = self.compute_heartbeat_envelope(dt)
        if envelope < 0.001:
            return gains, 0.0
        pulsed = {}
        for n, g in gains.items():
            pulsed[n] = float(np.clip(g * (1.0 + envelope), 0.0, 1.0))
        return pulsed, envelope

    @property
    def heartbeat_active(self):
        return self.current_bpm >= 1.0 and self.pulse_amplitude >= 0.001

    # ─── OSC Output ───

    def send_gains(self, gains):
        for n, gain in gains.items():
            self.osc_out.send_message(f"/shaper/harmonic/{n}/gain", [gain])
        self.last_gains_sent = gains

    def send_phases(self, phases):
        for n, phase_deg in phases.items():
            self.osc_out.send_message(f"/shaper/harmonic/{n}/phase", [phase_deg])
        self.last_phases_sent = phases

    def restore_base(self):
        """Restore base values before exiting.

        Always restores gains if heartbeat was active, since the pulse
        modifies gains regardless of param_mode.
        """
        if self.param_mode in ("gain", "both") or self.heartbeat_active:
            self.send_gains(self.base_gains)
        if self.param_mode in ("phase", "both"):
            self.send_phases(self.base_phases)

    # ─── Main Update ───

    def update_gain(self):
        """Full gain update: analyze + send (runs at update_rate)."""
        if self.samples_received < self.window_size // 2:
            return None
        if not self._has_good_frontal():
            return {"status": "no_signal"}
        gains = self.compute_gain_modulation()
        self.send_gains(gains)
        self.updates_sent += 1
        return {"tilt": self.tilt_smooth, "gains": gains}

    def update_phase_analysis(self):
        """Phase analysis only: update velocities from EEG (runs at update_rate)."""
        if self.samples_received < self.window_size // 2:
            return None
        if not self._has_any_good_channel():
            return {"status": "no_signal"}
        self.analyze_phase_velocities()
        self.updates_sent += 1
        return "ok"

    def tick_phase_output(self, dt):
        """Advance phase accumulators and send (runs at osc_rate)."""
        phases = self.advance_phases(dt)
        self.send_phases(phases)
        self.osc_sends += 1
        return {"phases": phases, "velocities": dict(self.phase_velocities)}

    # ─── Display ───

    def format_status(self, result):
        if not result:
            return f"  Buffering... {self.samples_received}/{self.window_size // 2}"

        if result.get("status") == "no_signal":
            if self.param_mode == "gain":
                return "  Waiting for frontal contact... (AF7/AF8)"
            return "  Waiting for sensor contact..."

        if self.param_mode == "gain":
            return self._format_gain_status(result)
        if self.param_mode == "phase":
            return self._format_phase_status(result)
        return self._format_both_status(result)

    def _format_gain_status(self, result):
        tilt = result.get("tilt", 0.0)
        gains = result.get("gains", {})

        if tilt > 0.1:
            tilt_label = "RELAXED"
            tilt_arrow = "<<<"
        elif tilt < -0.1:
            tilt_label = "FOCUSED"
            tilt_arrow = ">>>"
        else:
            tilt_label = "neutral"
            tilt_arrow = " . "

        parts = [f"tilt={tilt:+.2f} {tilt_arrow} {tilt_label:>8s}"]
        for n in sorted(gains.keys()):
            g = gains[n]
            base = self.base_gains.get(n, 0.8)
            delta = g - base
            bar_w = 8
            fill = int(g / 1.0 * bar_w)
            bar = "\u2588" * fill + "\u2591" * (bar_w - fill)
            sign = "+" if delta >= 0 else ""
            parts.append(f"H{n}[{bar}]{g:.2f}({sign}{delta:.2f})")

        if self.heartbeat_active:
            beat_icon = "\u2665" if self.last_envelope > self.pulse_amplitude * 0.3 else "\u2661"
            parts.append(f"{beat_icon}{self.current_bpm:.0f}bpm")

        return "  " + "  ".join(parts)

    def _format_phase_status(self, result):
        phases = result.get("phases", {})
        velocities = result.get("velocities", {})

        parts = []
        for n in sorted(phases.keys()):
            ph = phases[n]
            vel = velocities.get(n, 0.0)
            base = self.base_phases.get(n, 0.0)
            offset = (ph - base) % 360
            if offset > 180:
                offset -= 360

            # Rotation speed indicator
            if n == 1:
                speed_icon = "\u2693"  # anchor
            elif vel < 1.0:
                speed_icon = "\u00b7"  # dot (nearly still)
            elif vel < 10.0:
                speed_icon = "\u223c"  # tilde (slow)
            elif vel < 20.0:
                speed_icon = "\u2248"  # approx (medium)
            else:
                speed_icon = "\u224b"  # triple (fast)

            cfg = PHASE_SENSOR_MAP.get(n, {})
            band_label = cfg.get("band", "---")[:3] if n > 1 else "anc"
            parts.append(f"H{n} {speed_icon} {ph:5.1f}\u00b0({offset:+.0f}\u00b0) {vel:4.1f}\u00b0/s {band_label}")

        if self.heartbeat_active:
            beat_icon = "\u2665" if self.last_envelope > self.pulse_amplitude * 0.3 else "\u2661"
            parts.append(f"{beat_icon}{self.current_bpm:.0f}bpm")

        return "  " + "  ".join(parts)

    def _format_both_status(self, result):
        phases = result.get("phases", {})
        velocities = result.get("velocities", {})
        tilt = result.get("tilt", 0.0)
        gains = result.get("gains", {})

        # Compact phase display
        ph_parts = []
        for n in sorted(phases.keys()):
            vel = velocities.get(n, 0.0)
            if n == 1:
                ph_parts.append(f"H1\u2693")
            else:
                ph_parts.append(f"H{n} {phases[n]:5.1f}\u00b0 {vel:4.1f}\u00b0/s")
        phase_str = " ".join(ph_parts)

        # Compact gain display
        if self.gain_depth < 0.001:
            gain_str = f"slider=0%"
        else:
            if tilt > 0.1:
                tilt_icon = "<<<RELAX"
            elif tilt < -0.1:
                tilt_icon = "FOCUS>>>"
            else:
                tilt_icon = "neutral"
            pct = int(self.gain_depth * 100)
            gain_str = f"slider={pct}% {tilt_icon}"
            if gains:
                g_parts = [f"H{n}={g:.2f}" for n, g in sorted(gains.items())]
                gain_str += " " + " ".join(g_parts)

        # Heartbeat indicator
        if self.heartbeat_active:
            beat_icon = "\u2665" if self.last_envelope > self.pulse_amplitude * 0.3 else "\u2661"
            hr_str = f" {beat_icon}{self.current_bpm:.0f}bpm"
        else:
            hr_str = ""

        return f"  {phase_str} | {gain_str}{hr_str}"

    # ─── Loop ───

    def run_loop(self):
        if self.param_mode == "gain":
            self._run_gain_loop()
        elif self.param_mode == "phase":
            self._run_phase_loop()
        else:
            self._run_both_loop()

    def _run_gain_loop(self):
        """Gain mode: dual-rate when heartbeat active, single-rate otherwise."""
        ticks_per_analysis = max(1, int(round(self.osc_rate / self.update_rate)))
        display_every = max(1, int(round(self.osc_rate / 4.0)))
        tick = 0
        last_result = None

        while self.running:
            t0 = time.monotonic()
            buffering = self.samples_received < self.window_size // 2

            if tick % ticks_per_analysis == 0 and not buffering:
                if self._has_good_frontal():
                    gains = self.compute_gain_modulation()
                    self.current_tilt_gains = gains
                    if not self.heartbeat_active:
                        self.send_gains(gains)
                    self.updates_sent += 1
                    last_result = {"tilt": self.tilt_smooth, "gains": gains}
                elif last_result is None:
                    last_result = {"status": "no_signal"}

            # Heartbeat pulse: send pulsed gains at osc_rate
            if self.heartbeat_active and self.current_tilt_gains:
                pulsed, env = self.apply_heartbeat_pulse(self.current_tilt_gains, self.osc_interval)
                self.send_gains(pulsed)
                self.osc_sends += 1
                if last_result and last_result.get("gains"):
                    last_result["gains"] = pulsed

            if tick % display_every == 0:
                status = self.format_status(last_result)
                suffix = f"  [a:{self.updates_sent} o:{self.osc_sends}]" if self.heartbeat_active else f"  [{self.updates_sent}]"
                print(f"\r{status}{suffix}", end="", flush=True)

            tick += 1
            elapsed = time.monotonic() - t0
            sleep_for = self.osc_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _run_phase_loop(self):
        """Phase mode: dual-rate loop.

        Outer clock runs at osc_rate (30 Hz default) for smooth phase
        interpolation. EEG analysis runs every analysis_every ticks
        (update_rate / osc_rate ratio).
        When heartbeat is active, gain pulses are overlaid on base gains.
        """
        ticks_per_analysis = max(1, int(round(self.osc_rate / self.update_rate)))
        display_every = max(1, int(round(self.osc_rate / 4.0)))
        tick = 0
        last_result = None

        while self.running:
            t0 = time.monotonic()

            if tick % ticks_per_analysis == 0:
                analysis = self.update_phase_analysis()
                if analysis == "ok" or (last_result and last_result.get("phases")):
                    phase_result = self.tick_phase_output(self.osc_interval)
                    last_result = phase_result
                elif analysis and analysis.get("status") == "no_signal":
                    last_result = analysis
            else:
                if last_result and last_result.get("phases"):
                    phase_result = self.tick_phase_output(self.osc_interval)
                    last_result = phase_result

            # Heartbeat pulse on base gains (no EEG tilt in phase-only mode)
            if self.heartbeat_active:
                pulsed, env = self.apply_heartbeat_pulse(self.base_gains, self.osc_interval)
                self.send_gains(pulsed)

            if tick % display_every == 0:
                status = self.format_status(last_result)
                print(f"\r{status}  [a:{self.updates_sent} o:{self.osc_sends}]", end="", flush=True)

            tick += 1
            elapsed = time.monotonic() - t0
            sleep_for = self.osc_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)

    def _run_both_loop(self):
        """Combined mode: phase rotation (interpolated) + gain tilt (on analysis ticks).

        Gain depth is dynamic — controlled by /bridge/gain_depth OSC from
        the Launchpad slider via midi_relay.py.
        Heartbeat pulse overlays on gains at osc_rate when active.
        """
        ticks_per_analysis = max(1, int(round(self.osc_rate / self.update_rate)))
        display_every = max(1, int(round(self.osc_rate / 4.0)))
        tick = 0
        last_result = None
        phase_active = False

        while self.running:
            t0 = time.monotonic()
            buffering = self.samples_received < self.window_size // 2

            if tick % ticks_per_analysis == 0 and not buffering:
                if self._has_any_good_channel():
                    self.analyze_phase_velocities()
                    phase_active = True
                    self.updates_sent += 1

                if (self.gain_depth > 0.001 or self.heartbeat_active) and self._has_good_frontal():
                    gains = self.compute_gain_modulation()
                    self.current_tilt_gains = gains
                    if not self.heartbeat_active:
                        self.send_gains(gains)
                    if last_result is None:
                        last_result = {}
                    last_result["tilt"] = self.tilt_smooth
                    last_result["gains"] = gains

            # Phase output tick (fast, every iteration)
            if phase_active:
                phase_result = self.tick_phase_output(self.osc_interval)
                if last_result is None:
                    last_result = {}
                last_result["phases"] = phase_result["phases"]
                last_result["velocities"] = phase_result["velocities"]

            # Heartbeat pulse on gains at osc_rate
            if self.heartbeat_active:
                source = self.current_tilt_gains if self.current_tilt_gains else self.base_gains
                pulsed, env = self.apply_heartbeat_pulse(source, self.osc_interval)
                self.send_gains(pulsed)
                if last_result is None:
                    last_result = {}
                last_result["gains"] = pulsed

            if tick % display_every == 0:
                status = self.format_status(last_result)
                print(f"\r{status}  [a:{self.updates_sent} o:{self.osc_sends}]", end="", flush=True)

            tick += 1
            elapsed = time.monotonic() - t0
            sleep_for = self.osc_interval - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Muse Bridge — EEG parameter modulation for Harmonic Shaper"
    )
    parser.add_argument("--param", choices=["gain", "phase", "both"], default="both",
                        help="Which shaper parameter to modulate (default: both)")
    parser.add_argument("--shaper-ip", default=MB.get("shaper_ip", "127.0.0.1"),
                        help="Harmonic Shaper IP (default: 127.0.0.1)")
    parser.add_argument("--shaper-port", type=int,
                        default=MB.get("shaper_port", 9002),
                        help="Shaper OSC port (default: 9002)")
    parser.add_argument("--shaper-api", default=MB.get("shaper_api", "http://127.0.0.1:8080"),
                        help="Shaper HTTP API base URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--listen-port", type=int,
                        default=REC.get("osc_port", 5000),
                        help="Port to listen for Muse 2 OSC and /bridge/* control (default: 5000)")
    parser.add_argument("--depth", type=float, default=None,
                        help="Phase max deg/s, or gain fraction in gain-only mode (default: gain=0.20, phase/both=30)")
    parser.add_argument("--gain-depth", type=float, default=None,
                        help="Initial gain modulation depth 0-1 (default: 0.20 for gain, 0 for both — slider takes over)")
    parser.add_argument("--update-rate", type=float,
                        default=MB.get("update_rate_hz", 4.0),
                        help="EEG analysis rate in Hz (default: 4)")
    parser.add_argument("--osc-rate", type=float,
                        default=MB.get("osc_rate_hz", 30.0),
                        help="Phase OSC output rate in Hz for smooth interpolation (default: 30)")
    parser.add_argument("--window", type=float,
                        default=MB.get("window_seconds", 1.0),
                        help="EEG analysis window in seconds (default: 1.0)")
    parser.add_argument("--smoothing", type=float,
                        default=MB.get("smoothing_alpha", 0.25),
                        help="EMA smoothing factor 0-1 (default: 0.25)")
    parser.add_argument("--pulse", type=float,
                        default=MB.get("pulse_amplitude", 0.15),
                        help="Heartbeat pulse amplitude 0-1 (default: 0.15, 0=disabled)")
    args = parser.parse_args()

    # Resolve depths per mode
    if args.param == "gain":
        phase_depth = 0.0
        gain_depth = args.gain_depth if args.gain_depth is not None else (args.depth if args.depth is not None else 0.20)
    elif args.param == "phase":
        phase_depth = args.depth if args.depth is not None else 30.0
        gain_depth = 0.0
    else:  # both
        phase_depth = args.depth if args.depth is not None else 30.0
        gain_depth = args.gain_depth if args.gain_depth is not None else 0.0

    # OSC output to harmonic_shaper
    osc_out = udp_client.SimpleUDPClient(args.shaper_ip, args.shaper_port)

    # Bridge
    bridge = MuseBridge(
        osc_out=osc_out,
        shaper_api=args.shaper_api,
        param_mode=args.param,
        phase_depth=phase_depth,
        gain_depth=gain_depth,
        update_rate=args.update_rate,
        osc_rate=args.osc_rate,
        window_seconds=args.window,
        smoothing_alpha=args.smoothing,
        pulse_amplitude=args.pulse,
    )

    # Fetch base values from shaper
    print(f"\n  Fetching base values from {args.shaper_api}...")
    bridge.refresh_base_values()

    # OSC input (Muse 2 EEG + bridge control)
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", bridge.eeg_handler)
    disp.map("/muse/elements/horseshoe", bridge.horseshoe_handler)
    disp.map("/bridge/gain_depth", bridge.gain_depth_handler)
    disp.map("/bridge/heartbeat", bridge.heartbeat_handler)
    disp.map("/bridge/heartbeat_bpm", bridge.heartbeat_bpm_handler)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.listen_port), disp)

    def signal_handler(sig, frame):
        print("\n\n  Stopping bridge...")
        bridge.running = False
        print("  Restoring base values...")
        bridge.restore_base()
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # ─── Banner ───
    if args.param == "gain":
        mode_label = "GAIN TILT"
        depth_label = f"+/-{int(gain_depth * 100)}% of base gain"
        mode_desc = [
            "Tilt mapping (alpha/beta ratio):",
            "  Relaxed (alpha high) --> boost lower harmonics (warm)",
            "  Focused (beta high)  --> boost upper harmonics (bright)",
            f"  Weights: {TILT_WEIGHTS}",
            "  /bridge/gain_depth OSC overrides depth in real-time",
        ]
        param_path = "gain"
    elif args.param == "phase":
        mode_label = "PHASE ROTATION"
        depth_label = f"0-{phase_depth:.0f} deg/s max rotation"
        mode_desc = [
            "Per-band rotation (band power --> phase velocity):",
            "  H1: anchored (no rotation)",
        ]
        for n in sorted(PHASE_SENSOR_MAP.keys()):
            cfg = PHASE_SENSOR_MAP[n]
            mode_desc.append(f"  H{n}: {cfg['label']} --> rotation speed")
        mode_desc.append("  Stronger band activity = faster rotation")
        param_path = "phase"
    else:
        mode_label = "PHASE + GAIN"
        depth_label = f"phase: 0-{phase_depth:.0f} deg/s | gain: slider-controlled (init {int(gain_depth * 100)}%)"
        mode_desc = [
            "Phase rotation (band power --> velocity, interpolated):",
            "  H1: anchored (no rotation)",
        ]
        for n in sorted(PHASE_SENSOR_MAP.keys()):
            cfg = PHASE_SENSOR_MAP[n]
            mode_desc.append(f"  H{n}: {cfg['label']} --> rotation speed")
        mode_desc.append("")
        mode_desc.append("Gain tilt (alpha/beta ratio, depth from slider):")
        mode_desc.append(f"  Weights: {TILT_WEIGHTS}")
        mode_desc.append("  Slider at 0% = gains untouched by EEG")
        mode_desc.append("  Slider at 50% = EEG can swing gain +/-50% of base")
        mode_desc.append("  Control: /bridge/gain_depth on this port (0-127 or 0.0-1.0)")
        param_path = "phase+gain"

    print(f"\n{'='*70}")
    print(f"  Muse Bridge [{mode_label}]")
    print(f"{'='*70}")
    print(f"  IN:       /muse/eeg @ 0.0.0.0:{args.listen_port}")
    print(f"  OUT:      /shaper/harmonic/N/{param_path} @ {args.shaper_ip}:{args.shaper_port}")
    print(f"  API:      {args.shaper_api}")
    print(f"  Depth:    {depth_label}")
    print(f"  EEG rate: {args.update_rate} Hz (analysis)")
    if args.param in ("phase", "both"):
        print(f"  OSC rate: {args.osc_rate} Hz (interpolated phase output)")
    print(f"  Window:   {args.window}s ({int(args.window * SAMPLING_RATE)} samples)")
    print(f"  Smooth:   alpha={args.smoothing}")
    if args.pulse > 0:
        print(f"  Pulse:    {int(args.pulse * 100)}% gain boost per heartbeat")
        print(f"            /bridge/heartbeat or /bridge/heartbeat_bpm on this port")
    print(f"")
    for line in mode_desc:
        print(f"  {line}")
    print(f"{'='*70}")
    print(f"  Waiting for EEG data... (Ctrl+C to stop)\n")

    bridge.running = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        bridge.run_loop()
    except KeyboardInterrupt:
        pass

    bridge.restore_base()
    server.shutdown()
    print(f"\n  Done. Sent {bridge.updates_sent} updates. Base values restored.\n")


if __name__ == "__main__":
    main()
