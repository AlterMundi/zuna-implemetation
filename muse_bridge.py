"""
Muse Bridge — EEG-driven gain modulation for the Harmonic Shaper

Reads live Muse 2 EEG via OSC, computes a brain-state metric, and
modulates the gain of each harmonic in harmonic_shaper within bounded
limits around a user-set base curve.

Mode: tilt
    A single alpha/beta ratio tilts the gain curve — relaxation boosts
    lower harmonics (warmer), focus boosts upper harmonics (brighter).
    H3 is the neutral pivot.

The base gains are fetched from harmonic_shaper's HTTP API on startup.
EEG modulation is bounded by a configurable depth (default ±20%).

Usage:
    python muse_bridge.py --shaper-ip 127.0.0.1
    python muse_bridge.py --depth 0.3 --update-rate 4
    python muse_bridge.py --shaper-ip 127.0.0.1 --listen-port 5000

Test with the EEG simulator (no Muse 2 needed):
    python simulate_eeg.py &
    python muse_bridge.py --shaper-ip 127.0.0.1
"""

import argparse
import json
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


# ─────────────────────────────────────────────
# Shaper API client
# ─────────────────────────────────────────────

def fetch_shaper_state(api_base):
    """GET harmonic_shaper state to read current base gains."""
    try:
        resp = urllib.request.urlopen(f"{api_base}/api/state", timeout=3)
        data = json.loads(resp.read())
        gains = {}
        for k, v in data.get("voices", {}).items():
            gains[int(k)] = v.get("gain", 0.8)
        return gains
    except Exception as e:
        print(f"  WARNING: Could not fetch shaper state ({e}), using defaults")
        return None


# ─────────────────────────────────────────────
# Muse Gain Bridge
# ─────────────────────────────────────────────

class MuseGainBridge:
    """Modulates harmonic_shaper gains from Muse 2 EEG using spectral tilt."""

    def __init__(self, osc_out, shaper_api, depth, update_rate,
                 window_seconds=1.0, smoothing_alpha=0.25):
        self.osc_out = osc_out
        self.shaper_api = shaper_api
        self.depth = depth
        self.update_interval = 1.0 / update_rate
        self.smoothing_alpha = smoothing_alpha
        self.window_size = int(window_seconds * SAMPLING_RATE)

        # Ring buffer per channel
        self.buffers = {ch: np.zeros(self.window_size) for ch in CHANNELS}
        self.write_pos = 0
        self.samples_received = 0
        self.running = False

        # Signal quality
        self.contact_quality = {ch: 4.0 for ch in CHANNELS}

        # Base gains (fetched from shaper or defaults)
        self.base_gains = dict(DEFAULT_BASE_GAINS)
        self.active_harmonics = list(range(1, 6))

        # Tilt state
        self.tilt_smooth = 0.0
        self.last_gains_sent = {}

        self._lock = threading.Lock()
        self.updates_sent = 0

    # ─── Base Gain Management ───

    def refresh_base_gains(self):
        """Fetch current gains from harmonic_shaper as the new base."""
        fetched = fetch_shaper_state(self.shaper_api)
        if fetched:
            self.base_gains = fetched
            self.active_harmonics = sorted(fetched.keys())
            if not self.active_harmonics:
                self.active_harmonics = list(range(1, 6))
            print(f"  Base gains refreshed: {self._format_base()}")
        else:
            print(f"  Using default base gains: {self._format_base()}")

    def _format_base(self):
        parts = []
        for n in self.active_harmonics[:5]:
            parts.append(f"H{n}={self.base_gains.get(n, 0.8):.2f}")
        return " ".join(parts)

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

    # ─── Analysis ───

    def _get_channel_window(self, ch):
        with self._lock:
            return np.roll(self.buffers[ch], -self.write_pos % self.window_size).copy()

    def _has_good_frontal(self):
        """Check if at least one frontal channel has good contact."""
        with self._lock:
            af7_ok = self.contact_quality.get("AF7", 4.0) < 4.0
            af8_ok = self.contact_quality.get("AF8", 4.0) < 4.0
        return af7_ok or af8_ok

    def _compute_frontal_powers(self):
        """Average alpha and beta power from frontal channels with good contact."""
        alpha_total = 0.0
        beta_total = 0.0
        count = 0

        for ch in ["AF7", "AF8"]:
            with self._lock:
                contact_ok = self.contact_quality.get(ch, 4.0) < 4.0
            if not contact_ok:
                continue
            window = self._get_channel_window(ch)
            powers = compute_band_powers(window, SAMPLING_RATE)
            alpha_total += powers.get("alpha", 0.0)
            beta_total += powers.get("beta", 0.0)
            count += 1

        if count == 0:
            return 0.0, 0.0
        return alpha_total / count, beta_total / count

    # ─── Tilt Mode ───

    def compute_tilt(self):
        """Compute spectral tilt from frontal alpha/beta ratio.

        Returns value in [-1, +1]:
            +1 = fully relaxed (alpha dominant) → boost lower harmonics
            -1 = fully focused (beta dominant)  → boost upper harmonics
        """
        alpha, beta = self._compute_frontal_powers()
        total = alpha + beta
        if total < 1e-10:
            return 0.0
        return float((alpha - beta) / total)

    def compute_gain_modulation(self):
        """Compute per-harmonic gain values using the tilt mode."""
        raw_tilt = self.compute_tilt()
        self.tilt_smooth += (raw_tilt - self.tilt_smooth) * self.smoothing_alpha

        gains = {}
        harmonics = self.active_harmonics[:5]
        weights = TILT_WEIGHTS[:len(harmonics)]

        for i, n in enumerate(harmonics):
            w = weights[i] if i < len(weights) else 0.0
            modulator = float(np.clip(self.tilt_smooth * w, -1.0, 1.0))
            base = self.base_gains.get(n, 0.8)
            effective = base * (1.0 + self.depth * modulator)
            gains[n] = float(np.clip(effective, 0.0, 1.0))

        return gains

    # ─── OSC Output ───

    def send_gains(self, gains):
        """Send /shaper/harmonic/<n>/gain to harmonic_shaper."""
        for n, gain in gains.items():
            self.osc_out.send_message(f"/shaper/harmonic/{n}/gain", [gain])
        self.last_gains_sent = gains

    # ─── Main Update ───

    def update(self):
        if self.samples_received < self.window_size // 2:
            return None

        if not self._has_good_frontal():
            return {"status": "no_signal"}

        gains = self.compute_gain_modulation()
        self.send_gains(gains)
        self.updates_sent += 1

        return {
            "tilt": self.tilt_smooth,
            "gains": gains,
        }

    # ─── Display ───

    def format_status(self, result):
        if not result:
            return f"  Buffering... {self.samples_received}/{self.window_size // 2}"

        if result.get("status") == "no_signal":
            return "  Waiting for frontal contact... (AF7/AF8 horseshoe < 4)"

        tilt = result.get("tilt", 0.0)
        gains = result.get("gains", {})

        # Tilt indicator
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
            bar_width = 8
            fill = int(g / 1.0 * bar_width)
            bar = "\u2588" * fill + "\u2591" * (bar_width - fill)
            sign = "+" if delta >= 0 else ""
            parts.append(f"H{n}[{bar}]{g:.2f}({sign}{delta:.2f})")

        return "  " + "  ".join(parts)

    # ─── Loop ───

    def run_loop(self):
        while self.running:
            result = self.update()
            status = self.format_status(result)
            print(f"\r{status}  [{self.updates_sent}]", end="", flush=True)
            time.sleep(self.update_interval)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Muse Bridge — EEG gain modulation for Harmonic Shaper"
    )
    parser.add_argument("--shaper-ip", default=MB.get("shaper_ip", "127.0.0.1"),
                        help="Harmonic Shaper IP (default: 127.0.0.1)")
    parser.add_argument("--shaper-port", type=int,
                        default=MB.get("shaper_port", 9002),
                        help="Shaper OSC port (default: 9002)")
    parser.add_argument("--shaper-api", default=MB.get("shaper_api", "http://127.0.0.1:8080"),
                        help="Shaper HTTP API base URL (default: http://127.0.0.1:8080)")
    parser.add_argument("--listen-port", type=int,
                        default=REC.get("osc_port", 5000),
                        help="Port to listen for Muse 2 OSC (default: 5000)")
    parser.add_argument("--depth", type=float,
                        default=MB.get("default_depth", 0.20),
                        help="Modulation depth as fraction (default: 0.20 = +/-20%%)")
    parser.add_argument("--update-rate", type=float,
                        default=MB.get("update_rate_hz", 4.0),
                        help="Updates per second (default: 4 Hz)")
    parser.add_argument("--window", type=float,
                        default=MB.get("window_seconds", 1.0),
                        help="EEG analysis window in seconds (default: 1.0)")
    parser.add_argument("--smoothing", type=float,
                        default=MB.get("smoothing_alpha", 0.25),
                        help="EMA smoothing factor 0-1 (default: 0.25)")
    args = parser.parse_args()

    # OSC output to harmonic_shaper
    osc_out = udp_client.SimpleUDPClient(args.shaper_ip, args.shaper_port)

    # Bridge
    bridge = MuseGainBridge(
        osc_out=osc_out,
        shaper_api=args.shaper_api,
        depth=args.depth,
        update_rate=args.update_rate,
        window_seconds=args.window,
        smoothing_alpha=args.smoothing,
    )

    # Fetch base gains from shaper
    print(f"\n  Fetching base gains from {args.shaper_api}...")
    bridge.refresh_base_gains()

    # OSC input from Muse 2
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", bridge.eeg_handler)
    disp.map("/muse/elements/horseshoe", bridge.horseshoe_handler)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.listen_port), disp)

    def signal_handler(sig, frame):
        print("\n\n  Stopping bridge...")
        bridge.running = False
        # Restore base gains before exiting
        print("  Restoring base gains...")
        bridge.send_gains(bridge.base_gains)
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    depth_pct = int(args.depth * 100)
    print(f"\n{'='*65}")
    print(f"  Muse Bridge — EEG Gain Modulation [TILT mode]")
    print(f"{'='*65}")
    print(f"  IN:       /muse/eeg @ 0.0.0.0:{args.listen_port}")
    print(f"  OUT:      /shaper/harmonic/N/gain @ {args.shaper_ip}:{args.shaper_port}")
    print(f"  API:      {args.shaper_api}")
    print(f"  Depth:    +/-{depth_pct}% of base gain")
    print(f"  Rate:     {args.update_rate} Hz")
    print(f"  Window:   {args.window}s ({int(args.window * SAMPLING_RATE)} samples)")
    print(f"  Smooth:   alpha={args.smoothing}")
    print(f"  Base:     {bridge._format_base()}")
    print(f"")
    print(f"  Tilt mapping (alpha/beta ratio):")
    print(f"    Relaxed (alpha high) --> boost lower harmonics (warm)")
    print(f"    Focused (beta high)  --> boost upper harmonics (bright)")
    print(f"    Weights: {TILT_WEIGHTS}")
    print(f"{'='*65}")
    print(f"  Waiting for EEG data... (Ctrl+C to stop)\n")

    bridge.running = True
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        bridge.run_loop()
    except KeyboardInterrupt:
        pass

    # Restore base gains on clean exit
    bridge.send_gains(bridge.base_gains)
    server.shutdown()
    print(f"\n  Done. Sent {bridge.updates_sent} updates. Base gains restored.\n")


if __name__ == "__main__":
    main()
