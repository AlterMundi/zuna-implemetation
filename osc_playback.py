"""
OSC Playback — Streams enhanced EEG data to the BeaconMagnetActuator

Reads a ZUNA-enhanced .fif file and maps brain activity to vibration
commands sent to the ESP32 harmonic surface.

Two transport modes:
  http  — uses HTTP POST /play, /stop (master branch firmware)
  osc   — uses OSC /fnote, /fnote/rel (feature/musical-controls firmware)

Four playback modes:
  spectral      — dominant EEG frequency × multiplier → actuator frequency
  band_power    — EEG band power drives velocity on matching harmonic tine
  concentration — composite focus score → single tine velocity
  multi_tine    — different brain regions drive different tines

Usage:
    python osc_playback.py --input enhanced/session_001.fif --ip 192.168.4.176 --transport http --mode spectral
    python osc_playback.py --input enhanced/session_001.fif --ip 192.168.1.50 --transport osc --mode band_power
"""

import argparse
import json
import signal
import sys
import time
import urllib.request
import urllib.parse
from pathlib import Path

import numpy as np
from scipy.signal import welch
import mne
from pythonosc import udp_client

# ─────────────────────────────────────────────
# Load config
# ─────────────────────────────────────────────

def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}

CONFIG = load_config()
PB = CONFIG.get("playback", {})

# ─────────────────────────────────────────────
# EEG frequency bands
# ─────────────────────────────────────────────

BANDS = {
    "delta": (0.5, 4),
    "theta": (4, 8),
    "alpha": (8, 13),
    "smr":   (12, 15),
    "beta":  (13, 30),
    "gamma": (30, 44),
}

# Concentration score weights (same as EEG-Game)
CONCENTRATION_WEIGHTS = {
    "beta_alpha": 0.5,
    "smr": 0.3,
    "inv_theta_beta": 0.2,
}

# ─────────────────────────────────────────────
# EEG Analysis Helpers
# ─────────────────────────────────────────────

def compute_band_powers(signal_data, sfreq):
    """Compute power for each EEG frequency band."""
    powers = {}
    for band_name, (low, high) in BANDS.items():
        if len(signal_data) < sfreq:
            powers[band_name] = 0.0
            continue
        freqs, psd = welch(signal_data, fs=sfreq, nperseg=min(len(signal_data), int(sfreq * 2)))
        band_idx = (freqs >= low) & (freqs <= high)
        powers[band_name] = float(np.mean(psd[band_idx])) if np.any(band_idx) else 0.0
    return powers


def find_dominant_frequency(signal_data, sfreq, fmin=0.5, fmax=44.0):
    """Find the dominant frequency in the EEG signal."""
    if len(signal_data) < sfreq:
        return 10.0, 0.0  # default alpha
    freqs, psd = welch(signal_data, fs=sfreq, nperseg=min(len(signal_data), int(sfreq * 2)))
    mask = (freqs >= fmin) & (freqs <= fmax)
    if not np.any(mask):
        return 10.0, 0.0
    peak_idx = np.argmax(psd[mask])
    peak_freq = freqs[mask][peak_idx]
    peak_power = psd[mask][peak_idx]
    return float(peak_freq), float(peak_power)


def compute_concentration(band_powers):
    """Compute concentration score (0-100) from band powers."""
    alpha = band_powers.get("alpha", 1e-10) + 1e-10
    beta = band_powers.get("beta", 1e-10) + 1e-10
    theta = band_powers.get("theta", 1e-10) + 1e-10
    smr = band_powers.get("smr", 1e-10) + 1e-10

    beta_alpha_ratio = beta / alpha
    smr_power = smr
    inv_theta_beta = 1.0 / ((theta / beta) + 1e-10)

    raw_score = (
        CONCENTRATION_WEIGHTS["beta_alpha"] * beta_alpha_ratio +
        CONCENTRATION_WEIGHTS["smr"] * smr_power +
        CONCENTRATION_WEIGHTS["inv_theta_beta"] * inv_theta_beta
    )
    return float(np.clip(raw_score * 50, 0, 100))


# ─────────────────────────────────────────────
# Value Mapping
# ─────────────────────────────────────────────

def map_to_velocity(value, vmin, vmax, out_min=0, out_max=127):
    """Map a value from [vmin, vmax] to [out_min, out_max]."""
    if vmax <= vmin:
        return out_min
    normalized = (value - vmin) / (vmax - vmin)
    normalized = np.clip(normalized, 0.0, 1.0)
    return float(out_min + normalized * (out_max - out_min))


def clamp_frequency(freq, fmin=20.0, fmax=2000.0):
    """Clamp frequency to actuator range."""
    return float(np.clip(freq, fmin, fmax))


# ─────────────────────────────────────────────
# Transport layers (HTTP vs OSC)
# ─────────────────────────────────────────────

class HttpTransport:
    """Send commands to the actuator via HTTP POST (master branch)."""

    def __init__(self, ip, fundamental=64.0):
        self.base_url = f"http://{ip}"
        self.fundamental = fundamental
        # Tine layout from the actuator's config
        # Index 0=H6, 1=H5, 2=H4, 3=H3, 4=H2
        self.tine_harmonics = [6, 5, 4, 3, 2]
        self.tine_freqs = [fundamental * h for h in self.tine_harmonics]
        self.current_note_active = False
        self._fetch_status()

    def _fetch_status(self):
        """Fetch live status from ESP32 to get actual tine frequencies."""
        try:
            resp = urllib.request.urlopen(f"{self.base_url}/status", timeout=3)
            status = json.loads(resp.read())
            self.fundamental = status.get("fundamental_hz", self.fundamental)
            tines = status.get("tines", [])
            self.tine_harmonics = [t["harmonic"] for t in tines]
            self.tine_freqs = [t["freq"] for t in tines]
            print(f"  Connected to ESP32: {len(tines)} tines, fundamental={self.fundamental} Hz")
            for i, t in enumerate(tines):
                print(f"    [{i}] {t['name']}: {t['freq']} Hz (H{t['harmonic']})")
        except Exception as e:
            print(f"  WARNING: Could not fetch ESP32 status ({e}), using defaults")

    def freq_to_tine_index(self, target_freq):
        """Find the tine whose frequency is closest to the target."""
        if not self.tine_freqs:
            return 0
        distances = [abs(f - target_freq) for f in self.tine_freqs]
        return distances.index(min(distances))

    def send_note_on(self, freq, velocity, note_id):
        """HTTP POST /play?tine=X&vel=Y&dur=Z"""
        tine_idx = self.freq_to_tine_index(freq)
        actual_freq = self.tine_freqs[tine_idx] if tine_idx < len(self.tine_freqs) else freq
        # Map velocity 0-127 → 0-255 for HTTP API
        vel_http = int(np.clip(velocity * 2, 0, 255))
        params = urllib.parse.urlencode({"tine": tine_idx, "vel": vel_http, "dur": 5000})
        try:
            req = urllib.request.Request(f"{self.base_url}/play?{params}", method="POST")
            urllib.request.urlopen(req, timeout=2)
            self.current_note_active = True
        except Exception as e:
            print(f"  HTTP error: {e}")
        return actual_freq, velocity

    def send_note_off(self, note_id):
        """HTTP POST /stop"""
        if self.current_note_active:
            try:
                req = urllib.request.Request(f"{self.base_url}/stop", method="POST")
                urllib.request.urlopen(req, timeout=2)
                self.current_note_active = False
            except Exception:
                pass

    def send_panic(self):
        """HTTP POST /stop"""
        try:
            req = urllib.request.Request(f"{self.base_url}/stop", method="POST")
            urllib.request.urlopen(req, timeout=2)
            self.current_note_active = False
        except Exception:
            pass


class OscTransport:
    """Send commands via OSC /fnote (feature/musical-controls branch)."""

    def __init__(self, ip, port, config):
        self.client = udp_client.SimpleUDPClient(ip, port)
        self.osc_addr = config.get("osc_address", "/fnote")
        self.osc_addr_rel = config.get("osc_address_rel", "/fnote/rel")
        self.osc_addr_panic = config.get("osc_address_panic", "/allnotesoff")
        self.current_note_active = False

    def send_note_on(self, freq, velocity, note_id):
        self.client.send_message(self.osc_addr, [freq, velocity, note_id])
        self.current_note_active = True
        return freq, velocity

    def send_note_off(self, note_id):
        if self.current_note_active:
            self.client.send_message(self.osc_addr_rel, [0.0, 0.0, note_id])
            self.current_note_active = False

    def send_panic(self):
        self.client.send_message(self.osc_addr_panic, [])
        self.current_note_active = False


# ─────────────────────────────────────────────
# Playback Engine
# ─────────────────────────────────────────────

class PlaybackEngine:
    """Maps EEG epochs to vibration commands for the actuator."""

    def __init__(self, transport, config):
        self.transport = transport
        self.config = config
        self.note_id = 0
        self.current_note_active = False

        self.harmonic_multiplier = config.get("harmonic_multiplier", 32)
        self.vel_min = config.get("velocity_min", 0)
        self.vel_max = config.get("velocity_max", 127)
        self.freq_min = config.get("frequency_min", 20.0)
        self.freq_max = config.get("frequency_max", 2000.0)
        self.fundamental = config.get("fundamental_hz", 64.0)

        # Track power range for adaptive normalization
        self.power_history = []
        self.power_percentile_low = 0.0
        self.power_percentile_high = 1.0

    def send_note_on(self, freq, velocity):
        """Send note-on via the active transport."""
        freq = clamp_frequency(freq, self.freq_min, self.freq_max)
        velocity = float(np.clip(velocity, self.vel_min, self.vel_max))
        self.note_id += 1
        return self.transport.send_note_on(freq, velocity, self.note_id)

    def send_note_off(self):
        """Send note-off via the active transport."""
        self.transport.send_note_off(self.note_id)

    def send_panic(self):
        """Send panic stop via the active transport."""
        self.transport.send_panic()

    def update_power_range(self, power):
        """Adaptively track power range for normalization."""
        self.power_history.append(power)
        if len(self.power_history) > 50:
            self.power_history.pop(0)
        if len(self.power_history) > 5:
            self.power_percentile_low = float(np.percentile(self.power_history, 10))
            self.power_percentile_high = float(np.percentile(self.power_history, 90))

    # ─── Mode: Spectral ───

    def process_spectral(self, epoch_data, sfreq, ch_names):
        """Dominant EEG frequency × multiplier → /fnote frequency."""
        # Average across all channels for spectral analysis
        avg_signal = np.mean(epoch_data, axis=0)
        peak_freq, peak_power = find_dominant_frequency(avg_signal, sfreq)

        # Map EEG frequency to actuator frequency via harmonic multiplier
        actuator_freq = peak_freq * self.harmonic_multiplier

        # Map power to velocity
        self.update_power_range(peak_power)
        velocity = map_to_velocity(
            peak_power, self.power_percentile_low, self.power_percentile_high,
            30, 127  # minimum velocity 30 so you always feel something
        )

        self.send_note_off()
        freq_sent, vel_sent = self.send_note_on(actuator_freq, velocity)
        return {
            "mode": "spectral",
            "eeg_freq": peak_freq,
            "eeg_power": peak_power,
            "actuator_freq": freq_sent,
            "velocity": vel_sent,
        }

    # ─── Mode: Band Power ───

    def process_band_power(self, epoch_data, sfreq, ch_names):
        """Dominant EEG band → matching harmonic tine, band strength → velocity."""
        avg_signal = np.mean(epoch_data, axis=0)
        powers = compute_band_powers(avg_signal, sfreq)

        # Find dominant band
        dominant_band = max(powers, key=powers.get)
        dominant_power = powers[dominant_band]

        # Map band center frequency to harmonic
        band_center = np.mean(BANDS[dominant_band])
        actuator_freq = band_center * self.harmonic_multiplier

        # Map power to velocity
        self.update_power_range(dominant_power)
        velocity = map_to_velocity(
            dominant_power, self.power_percentile_low, self.power_percentile_high,
            30, 127
        )

        self.send_note_off()
        freq_sent, vel_sent = self.send_note_on(actuator_freq, velocity)
        return {
            "mode": "band_power",
            "dominant_band": dominant_band,
            "band_power": dominant_power,
            "all_powers": powers,
            "actuator_freq": freq_sent,
            "velocity": vel_sent,
        }

    # ─── Mode: Concentration ───

    def process_concentration(self, epoch_data, sfreq, ch_names):
        """Concentration score → velocity on fundamental harmonic."""
        # Use frontal channels if available
        frontal_indices = [i for i, ch in enumerate(ch_names) if ch in ("AF7", "AF8", "Fp1", "Fp2", "F3", "F4")]
        if not frontal_indices:
            frontal_indices = list(range(min(2, len(ch_names))))

        frontal_avg = np.mean(epoch_data[frontal_indices], axis=0)
        powers = compute_band_powers(frontal_avg, sfreq)
        score = compute_concentration(powers)

        # Fixed frequency (fundamental × 5 = H5), velocity from score
        actuator_freq = self.fundamental * 5  # H5 = 320 Hz by default
        velocity = map_to_velocity(score, 20, 80, 30, 127)

        self.send_note_off()
        freq_sent, vel_sent = self.send_note_on(actuator_freq, velocity)
        return {
            "mode": "concentration",
            "score": score,
            "band_powers": powers,
            "actuator_freq": freq_sent,
            "velocity": vel_sent,
        }

    # ─── Mode: Multi-Tine ───

    def process_multi_tine(self, epoch_data, sfreq, ch_names):
        """Different brain regions → different tines (sequential, since actuator is monophonic)."""
        # Group channels by brain region
        regions = {
            "frontal": [i for i, ch in enumerate(ch_names)
                        if any(ch.startswith(p) for p in ("AF", "Fp", "F"))],
            "central": [i for i, ch in enumerate(ch_names)
                        if any(ch.startswith(p) for p in ("C", "Cz"))],
            "temporal": [i for i, ch in enumerate(ch_names)
                         if any(ch.startswith(p) for p in ("TP", "T"))],
            "parietal": [i for i, ch in enumerate(ch_names)
                         if any(ch.startswith(p) for p in ("P",))],
        }

        # Find the most active region
        region_powers = {}
        for region_name, indices in regions.items():
            if not indices:
                continue
            region_signal = np.mean(epoch_data[indices], axis=0)
            powers = compute_band_powers(region_signal, sfreq)
            region_powers[region_name] = sum(powers.values())

        if not region_powers:
            return {"mode": "multi_tine", "status": "no_regions"}

        # Map region to harmonic
        region_harmonics = {
            "temporal": 2,   # H2 = 128 Hz (deep, low)
            "central": 3,    # H3 = 192 Hz
            "parietal": 4,   # H4 = 256 Hz
            "frontal": 5,    # H5 = 320 Hz (attention, high)
        }

        dominant_region = max(region_powers, key=region_powers.get)
        harmonic = region_harmonics.get(dominant_region, 3)
        actuator_freq = self.fundamental * harmonic
        power = region_powers[dominant_region]

        self.update_power_range(power)
        velocity = map_to_velocity(
            power, self.power_percentile_low, self.power_percentile_high,
            30, 127
        )

        self.send_note_off()
        freq_sent, vel_sent = self.send_note_on(actuator_freq, velocity)
        return {
            "mode": "multi_tine",
            "dominant_region": dominant_region,
            "region_powers": region_powers,
            "harmonic": harmonic,
            "actuator_freq": freq_sent,
            "velocity": vel_sent,
        }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Play back enhanced EEG as vibration to the actuator"
    )
    parser.add_argument(
        "--input", required=True,
        help="Input .fif file (ZUNA-enhanced or raw recording)"
    )
    parser.add_argument(
        "--ip", default=PB.get("actuator_ip", "127.0.0.1"),
        help="Actuator IP address"
    )
    parser.add_argument(
        "--port", type=int, default=PB.get("actuator_port", 53280),
        help="Actuator OSC port (default: 53280, only used with --transport osc)"
    )
    parser.add_argument(
        "--transport", choices=["http", "osc"], default="http",
        help="Transport: http (master branch) or osc (feature/musical-controls)"
    )
    parser.add_argument(
        "--mode", choices=["spectral", "band_power", "concentration", "multi_tine"],
        default=PB.get("mode", "spectral"),
        help="Playback mode"
    )
    parser.add_argument(
        "--speed", type=float, default=PB.get("speed", 1.0),
        help="Playback speed multiplier (default: 1.0)"
    )
    parser.add_argument(
        "--epoch-length", type=float,
        default=PB.get("epoch_duration", 5.0),
        help="Analysis window length in seconds (default: 5.0)"
    )
    parser.add_argument(
        "--loop", action="store_true", default=False,
        help="Loop playback continuously"
    )
    args = parser.parse_args()

    # Validate input
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    # Load EEG data
    print(f"\n{'='*50}")
    print(f"  📡 EEG Playback → Harmonic Surface")
    print(f"{'='*50}")
    print(f"  Input:     {input_path}")
    print(f"  Target:    {args.ip}")
    print(f"  Transport: {args.transport.upper()}")
    print(f"  Mode:      {args.mode}")
    print(f"  Speed:     {args.speed}x")
    print(f"  Epoch:     {args.epoch_length}s")
    print(f"  Loop:      {args.loop}")
    print(f"{'='*50}")

    # Create transport
    if args.transport == "http":
        print(f"\n  Connecting via HTTP to {args.ip}...")
        transport = HttpTransport(args.ip, PB.get("fundamental_hz", 64.0))
    else:
        print(f"\n  Using OSC transport to {args.ip}:{args.port}")
        transport = OscTransport(args.ip, args.port, PB)

    print(f"\n  Loading EEG data...", end="", flush=True)

    raw = mne.io.read_raw_fif(str(input_path), preload=True, verbose=False)
    sfreq = raw.info["sfreq"]
    ch_names = raw.ch_names
    data = raw.get_data()  # (n_channels, n_samples)
    total_duration = data.shape[1] / sfreq

    print(f" done")
    print(f"  Channels: {len(ch_names)} ({', '.join(ch_names[:6])}{'...' if len(ch_names) > 6 else ''})")
    print(f"  Duration: {total_duration:.1f}s ({total_duration/60:.1f} min)")
    print(f"  Rate:     {sfreq} Hz")
    print()

    # Create engine
    engine = PlaybackEngine(transport, PB)

    # Select processing function
    mode_func = {
        "spectral": engine.process_spectral,
        "band_power": engine.process_band_power,
        "concentration": engine.process_concentration,
        "multi_tine": engine.process_multi_tine,
    }[args.mode]

    # Epoch parameters
    epoch_samples = int(args.epoch_length * sfreq)
    epoch_interval = args.epoch_length / args.speed
    total_epochs = int(np.ceil(data.shape[1] / epoch_samples))

    # Handle Ctrl+C
    running = True
    def signal_handler(sig, frame):
        nonlocal running
        running = False
        print("\n\n  Stopping playback...")
        engine.send_panic()

    signal.signal(signal.SIGINT, signal_handler)

    # ─── Playback Loop ───
    print(f"  ▶  Playing ({total_epochs} epochs, {epoch_interval:.1f}s each)...")
    print(f"  Press Ctrl+C to stop\n")

    pass_number = 0
    while running:
        pass_number += 1
        if pass_number > 1:
            print(f"\n  🔁 Loop pass {pass_number}")

        for epoch_idx in range(total_epochs):
            if not running:
                break

            # Extract epoch
            start_sample = epoch_idx * epoch_samples
            end_sample = min(start_sample + epoch_samples, data.shape[1])
            epoch_data = data[:, start_sample:end_sample]

            # Convert from volts back to microvolts for analysis
            epoch_data_uv = epoch_data * 1e6

            # Process epoch through selected mode
            result = mode_func(epoch_data_uv, sfreq, ch_names)

            # Display
            t = start_sample / sfreq
            if args.mode == "spectral":
                print(
                    f"  [{t:6.1f}s] "
                    f"EEG: {result.get('eeg_freq', 0):.1f} Hz → "
                    f"Actuator: {result.get('actuator_freq', 0):.0f} Hz, "
                    f"vel={result.get('velocity', 0):.0f}"
                )
            elif args.mode == "band_power":
                print(
                    f"  [{t:6.1f}s] "
                    f"Band: {result.get('dominant_band', '?'):>5s} → "
                    f"Actuator: {result.get('actuator_freq', 0):.0f} Hz, "
                    f"vel={result.get('velocity', 0):.0f}"
                )
            elif args.mode == "concentration":
                score = result.get("score", 0)
                bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                print(
                    f"  [{t:6.1f}s] "
                    f"Focus: [{bar}] {score:.0f}% → "
                    f"vel={result.get('velocity', 0):.0f}"
                )
            elif args.mode == "multi_tine":
                print(
                    f"  [{t:6.1f}s] "
                    f"Region: {result.get('dominant_region', '?'):>8s} "
                    f"H{result.get('harmonic', '?')} → "
                    f"{result.get('actuator_freq', 0):.0f} Hz, "
                    f"vel={result.get('velocity', 0):.0f}"
                )

            # Wait for next epoch (adjusted by speed)
            time.sleep(epoch_interval)

        if not args.loop:
            break

    # Stop
    engine.send_panic()
    print(f"\n  ■  Playback complete.\n")


if __name__ == "__main__":
    main()
