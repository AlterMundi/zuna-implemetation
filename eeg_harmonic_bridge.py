"""
EEG Harmonic Bridge — Muse 2 sensors as harmonic series controller

Maps each EEG sensor to a harmonic voice in the natural series:
  TP9  (left temporal)  → H2 = 128 Hz  — modulated by theta
  AF7  (left frontal)   → H3 = 192 Hz  — modulated by alpha
  AF8  (right frontal)  → H4 = 256 Hz  — modulated by beta
  TP10 (right temporal) → H5 = 320 Hz  — modulated by gamma
  Derived (coherence)   → H1 =  64 Hz

Features:
  1. Per-band velocity: each sensor's gain is driven by a specific EEG band
  2. Filter modulation: alpha/beta ratio → Surge XT filter cutoff (/param)
  3. Asymmetry mapping: L/R brain balance → stereo pan (--stereo mode)

Usage:
    python eeg_harmonic_bridge.py --surge-ip 127.0.0.1 --actuator-ip 192.168.4.176
    python eeg_harmonic_bridge.py --surge-ip 127.0.0.1 --stereo
    python eeg_harmonic_bridge.py --actuator-ip 192.168.4.176 --mono
"""

import argparse
import json
import signal
import sys
import time
import threading
from pathlib import Path

import numpy as np
from pythonosc import dispatcher, osc_server, udp_client

from osc_playback import compute_band_powers, map_to_velocity, BANDS

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

CHANNELS = REC.get("channels", ["TP9", "AF7", "AF8", "TP10"])
SAMPLING_RATE = REC.get("sampling_rate", 256)

# ─── Option 1: Per-band mapping ───
# Each sensor is driven by the EEG band most relevant to its brain region
SENSOR_CONFIG = {
    "TP9":  {"harmonic": 2, "band": "theta", "side": "L", "label": "L-temp"},
    "AF7":  {"harmonic": 3, "band": "alpha", "side": "L", "label": "L-front"},
    "AF8":  {"harmonic": 4, "band": "beta",  "side": "R", "label": "R-front"},
    "TP10": {"harmonic": 5, "band": "gamma", "side": "R", "label": "R-temp"},
}

# ─── Option 3: Surge XT parameter paths ───
SURGE_PARAMS = {
    "filter_cutoff": "/param/a/filt/cutoff",
    "filter_resonance": "/param/a/filt/resonance",
    "reverb_mix": "/param/fx/reverb/mix",
}

# ─────────────────────────────────────────────
# Multi-target OSC sender
# ─────────────────────────────────────────────

class MultiOscSender:
    """Sends /fnote messages to multiple OSC targets simultaneously."""

    def __init__(self):
        self.targets = []

    def add_target(self, name, ip, port):
        client = udp_client.SimpleUDPClient(ip, port)
        self.targets.append({"name": name, "client": client, "ip": ip, "port": port})

    def send(self, address, args):
        for t in self.targets:
            t["client"].send_message(address, args)

    def send_to(self, name, address, args):
        """Send to a specific named target only."""
        for t in self.targets:
            if t["name"] == name:
                t["client"].send_message(address, args)

    def describe(self):
        return ", ".join(f"{t['name']}={t['ip']}:{t['port']}" for t in self.targets)


# ─────────────────────────────────────────────
# EEG Harmonic Bridge
# ─────────────────────────────────────────────

class EEGHarmonicBridge:
    """Maps Muse 2 EEG sensors to natural harmonic series voices."""

    def __init__(self, osc_sender, f1, update_rate, stereo=False,
                 window_seconds=1.0):
        self.sender = osc_sender
        self.f1 = f1
        self.stereo = stereo
        self.update_interval = 1.0 / update_rate
        self.window_size = int(window_seconds * SAMPLING_RATE)

        # Ring buffer per channel
        self.buffers = {ch: np.zeros(self.window_size) for ch in CHANNELS}
        self.write_pos = 0
        self.samples_received = 0
        self.running = False

        # Voice IDs: one per harmonic (H1=1, H2=2, etc.)
        self.voice_active = {n: False for n in range(1, 6)}

        # Power normalization per channel
        self.power_history = {ch: [] for ch in CHANNELS}

        # Signal quality tracking
        self.contact_quality = {ch: 4.0 for ch in CHANNELS}
        self.channel_muted = {ch: False for ch in CHANNELS}

        # Asymmetry tracking (Option 6)
        self.asymmetry = 0.0      # -1.0 = full left, +1.0 = full right
        self.asymmetry_smooth = 0.0

        # Filter state (Option 3)
        self.filter_value = 0.5   # 0.0 = closed/warm, 1.0 = open/bright
        self.filter_smooth = 0.5

        self._lock = threading.Lock()
        self.updates_sent = 0

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

    # ─── Signal Quality ───

    def get_channel_window(self, ch):
        with self._lock:
            return np.roll(self.buffers[ch], -self.write_pos % self.window_size).copy()

    def is_saturated(self, ch):
        window = self.get_channel_window(ch)
        variance = np.var(window)
        amplitude = np.max(np.abs(window))
        if variance < 1.0 and amplitude > 500:
            return True
        if amplitude > 1500:
            return True
        return False

    def check_channel_quality(self, ch):
        with self._lock:
            contact = self.contact_quality[ch]
        if contact >= 4.0:
            self.channel_muted[ch] = True
            return False
        if self.is_saturated(ch):
            self.channel_muted[ch] = True
            return False
        self.channel_muted[ch] = False
        return True

    # ─── Analysis ───

    def get_band_power(self, ch, band):
        """Get power for a specific EEG band from a channel."""
        window = self.get_channel_window(ch)
        powers = compute_band_powers(window, SAMPLING_RATE)
        return powers.get(band, 0.0)

    def get_all_band_powers(self, ch):
        """Get all band powers for a channel."""
        window = self.get_channel_window(ch)
        return compute_band_powers(window, SAMPLING_RATE)

    def update_power_range(self, ch, power):
        hist = self.power_history[ch]
        hist.append(power)
        if len(hist) > 50:
            hist.pop(0)
        if len(hist) > 5:
            return float(np.percentile(hist, 10)), float(np.percentile(hist, 90))
        return 0.0, max(power, 1e-10)

    def compute_coherence(self):
        windows = [self.get_channel_window(ch) for ch in CHANNELS]
        if len(windows) < 2:
            return 0.5
        correlations = []
        for i in range(len(windows)):
            for j in range(i + 1, len(windows)):
                if np.std(windows[i]) > 0 and np.std(windows[j]) > 0:
                    corr = abs(np.corrcoef(windows[i], windows[j])[0, 1])
                    correlations.append(corr)
        return float(np.mean(correlations)) if correlations else 0.5

    # ─── Option 6: Asymmetry ───

    def compute_asymmetry(self):
        """Compute L/R brain asymmetry. Returns -1 (left) to +1 (right)."""
        left_channels = [ch for ch in CHANNELS if SENSOR_CONFIG[ch]["side"] == "L"]
        right_channels = [ch for ch in CHANNELS if SENSOR_CONFIG[ch]["side"] == "R"]

        left_power = 0.0
        left_count = 0
        for ch in left_channels:
            if not self.channel_muted.get(ch, True):
                left_power += sum(self.get_all_band_powers(ch).values())
                left_count += 1

        right_power = 0.0
        right_count = 0
        for ch in right_channels:
            if not self.channel_muted.get(ch, True):
                right_power += sum(self.get_all_band_powers(ch).values())
                right_count += 1

        if left_count > 0:
            left_power /= left_count
        if right_count > 0:
            right_power /= right_count

        total = left_power + right_power
        if total < 1e-10:
            return 0.0

        # -1 = full left, +1 = full right
        return float((right_power - left_power) / total)

    # ─── Option 3: Filter Modulation ───

    def compute_filter_value(self):
        """Compute filter cutoff from frontal alpha/beta ratio.
        Relaxed (high alpha) → closed/warm, Focused (high beta) → open/bright."""
        alpha_total = 0.0
        beta_total = 0.0
        count = 0

        for ch in ["AF7", "AF8"]:
            if not self.channel_muted.get(ch, True):
                powers = self.get_all_band_powers(ch)
                alpha_total += powers.get("alpha", 0.0)
                beta_total += powers.get("beta", 0.0)
                count += 1

        if count == 0 or (alpha_total + beta_total) < 1e-10:
            return 0.5  # neutral

        # beta / (alpha + beta) → 0 when relaxed, 1 when focused
        return float(beta_total / (alpha_total + beta_total))

    # ─── Main Update ───

    def update(self):
        if self.samples_received < self.window_size // 2:
            return None

        results = {}

        # ─── H2-H5: Per-band velocity (Option 1) ───
        for ch in CHANNELS:
            cfg = SENSOR_CONFIG[ch]
            harmonic_n = cfg["harmonic"]
            band = cfg["band"]
            freq = self.f1 * harmonic_n
            voice_id = harmonic_n

            good_signal = self.check_channel_quality(ch)

            if good_signal:
                power = self.get_band_power(ch, band)
                p_low, p_high = self.update_power_range(ch, power)
                velocity = map_to_velocity(power, p_low, p_high, 10, 127)

                # ─── Option 6: Stereo gain scaling ───
                if self.stereo:
                    side = cfg["side"]
                    # Boost the dominant side, attenuate the other
                    if side == "L":
                        stereo_scale = max(0.2, 1.0 - max(0.0, self.asymmetry_smooth))
                    else:
                        stereo_scale = max(0.2, 1.0 + min(0.0, self.asymmetry_smooth))
                    velocity = velocity * stereo_scale
            else:
                power = 0.0
                velocity = 0.0

            # Release previous, send new
            if self.voice_active[harmonic_n]:
                self.sender.send("/fnote/rel", [0.0, 0.0, float(voice_id)])

            if velocity > 0:
                self.sender.send("/fnote", [float(freq), float(velocity), float(voice_id)])
                self.voice_active[harmonic_n] = True
            else:
                self.voice_active[harmonic_n] = False

            results[ch] = {
                "harmonic": harmonic_n,
                "band": band,
                "freq": freq,
                "vel": velocity,
                "power": power,
                "muted": not good_signal,
            }

        # ─── H1: Coherence ───
        coherence = self.compute_coherence()
        h1_vel = map_to_velocity(coherence, 0.2, 0.8, 10, 127)

        if self.voice_active[1]:
            self.sender.send("/fnote/rel", [0.0, 0.0, 1.0])
        self.sender.send("/fnote", [float(self.f1), float(h1_vel), 1.0])
        self.voice_active[1] = True

        results["coherence"] = {
            "harmonic": 1, "freq": self.f1, "vel": h1_vel, "coherence": coherence,
        }

        # ─── Option 3: Filter cutoff ───
        raw_filter = self.compute_filter_value()
        self.filter_smooth += (raw_filter - self.filter_smooth) * 0.3  # smooth
        self.sender.send_to("Surge XT", SURGE_PARAMS["filter_cutoff"],
                            [float(self.filter_smooth)])
        results["filter"] = self.filter_smooth

        # ─── Option 6: Asymmetry ───
        raw_asym = self.compute_asymmetry()
        self.asymmetry_smooth += (raw_asym - self.asymmetry_smooth) * 0.3
        results["asymmetry"] = self.asymmetry_smooth

        self.updates_sent += 1
        return results

    def panic(self):
        self.sender.send("/allnotesoff", [])
        for n in self.voice_active:
            self.voice_active[n] = False

    # ─── Display ───

    def format_status(self, results):
        if not results:
            return f"  Buffering... {self.samples_received}/{self.window_size // 2}"

        parts = []

        # H1 coherence
        coh = results["coherence"]
        coh_bar = self._bar(coh["vel"], 127, 6)
        parts.append(f"H1{coh_bar}")

        # H2-H5 per-band
        for ch in CHANNELS:
            r = results[ch]
            cfg = SENSOR_CONFIG[ch]
            if r.get("muted"):
                parts.append(f"H{r['harmonic']}[MUTED]")
            else:
                bar = self._bar(r["vel"], 127, 6)
                parts.append(f"H{r['harmonic']}{bar}{cfg['band'][:3]}")

        # Filter & asymmetry
        filt = results.get("filter", 0.5)
        asym = results.get("asymmetry", 0.0)
        asym_indicator = "◄" if asym < -0.1 else ("►" if asym > 0.1 else "●")

        parts.append(f"filt={filt:.2f}")
        parts.append(f"{asym_indicator}{asym:+.2f}")

        return "  " + " ".join(parts)

    @staticmethod
    def _bar(value, maximum, width):
        n = int(value / maximum * width) if maximum > 0 else 0
        return "[" + "█" * n + "░" * (width - n) + "]"

    # ─── Loop ───

    def run_loop(self):
        while self.running:
            results = self.update()
            status = self.format_status(results)
            print(f"\r{status}  [{self.updates_sent}]", end="", flush=True)
            time.sleep(self.update_interval)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="EEG Harmonic Bridge — Muse 2 sensors as harmonic series controller"
    )
    parser.add_argument("--surge-ip", default=None,
                        help="Surge XT IP address")
    parser.add_argument("--surge-port", type=int, default=53280,
                        help="Surge XT OSC port (default: 53280)")
    parser.add_argument("--actuator-ip", default=None,
                        help="ESP32 Beacon IP address")
    parser.add_argument("--actuator-port", type=int, default=53280,
                        help="ESP32 OSC port (default: 53280)")
    parser.add_argument("--listen-port", type=int,
                        default=REC.get("osc_port", 5000),
                        help="Port to listen for Muse 2 OSC (default: 5000)")
    parser.add_argument("--f1", type=float, default=64.0,
                        help="Fundamental frequency in Hz (default: 64.0)")
    parser.add_argument("--stereo", action="store_true", default=False,
                        help="Enable stereo: L/R brain asymmetry scales harmonic gain")
    parser.add_argument("--mono", action="store_true", default=False,
                        help="Force mono mode (default)")
    parser.add_argument("--update-rate", type=float, default=2.0,
                        help="Updates per second (default: 2 Hz)")
    parser.add_argument("--window", type=float, default=1.0,
                        help="Analysis window in seconds (default: 1.0)")
    args = parser.parse_args()

    stereo = args.stereo and not args.mono

    if not args.surge_ip and not args.actuator_ip:
        print("ERROR: Specify at least one target: --surge-ip and/or --actuator-ip")
        sys.exit(1)

    # Build sender
    sender = MultiOscSender()
    if args.surge_ip:
        sender.add_target("Surge XT", args.surge_ip, args.surge_port)
    if args.actuator_ip:
        sender.add_target("ESP32", args.actuator_ip, args.actuator_port)

    # Bridge
    bridge = EEGHarmonicBridge(
        osc_sender=sender,
        f1=args.f1,
        update_rate=args.update_rate,
        stereo=stereo,
        window_seconds=args.window,
    )

    # OSC input
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", bridge.eeg_handler)
    disp.map("/muse/elements/horseshoe", bridge.horseshoe_handler)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.listen_port), disp)

    def signal_handler(sig, frame):
        print("\n\n  Stopping bridge...")
        bridge.running = False
        bridge.panic()
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # Banner
    mode_label = "STEREO" if stereo else "MONO"
    print(f"\n{'='*65}")
    print(f"  🧠 EEG Harmonic Bridge — Natural Harmonic Series [{mode_label}]")
    print(f"{'='*65}")
    print(f"  IN:      /muse/eeg @ 0.0.0.0:{args.listen_port}")
    print(f"  OUT:     {sender.describe()}")
    print(f"  f₁:      {args.f1} Hz")
    print(f"  Rate:    {args.update_rate} Hz")
    print(f"  Mapping:")
    for ch in CHANNELS:
        cfg = SENSOR_CONFIG[ch]
        h = cfg["harmonic"]
        b = cfg["band"]
        s = cfg["side"]
        print(f"    {ch:5s} ({s}) → H{h} = {args.f1 * h:6.0f} Hz  gain ← {b}")
    print(f"    {'Coher':5s}     → H1 = {args.f1:6.0f} Hz  gain ← coherence")
    print(f"  Filter:  alpha/beta ratio → Surge XT cutoff")
    if stereo:
        print(f"  Stereo:  L/R asymmetry scales harmonic gain")
    print(f"{'='*65}")
    print(f"  Waiting for EEG data... (Ctrl+C to stop)\n")

    bridge.running = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        bridge.run_loop()
    except KeyboardInterrupt:
        pass

    bridge.panic()
    server.shutdown()
    print(f"\n  Done. Sent {bridge.updates_sent} updates.\n")


if __name__ == "__main__":
    main()
