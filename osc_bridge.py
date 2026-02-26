"""
OSC Bridge — Real-time Muse 2 EEG → Harmonic Surface Actuator

Receives live EEG from Mind Monitor via OSC, analyzes it in a sliding
window, and immediately forwards mapped vibration commands to the
ESP32 Beacon via OSC /fnote.

    Muse 2 → /muse/eeg :5000 → [analyze] → /fnote :53280 → ESP32

Usage:
    python osc_bridge.py --actuator-ip 192.168.4.176 --mode spectral
    python osc_bridge.py --actuator-ip 192.168.4.176 --mode concentration --update-rate 4
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

# Reuse analysis functions from the playback module
from osc_playback import (
    compute_band_powers,
    find_dominant_frequency,
    compute_concentration,
    map_to_velocity,
    clamp_frequency,
    BANDS,
)

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
PB = CONFIG.get("playback", {})
REC = CONFIG.get("recorder", {})

CHANNELS = REC.get("channels", ["TP9", "AF7", "AF8", "TP10"])
SAMPLING_RATE = REC.get("sampling_rate", 256)

# ─────────────────────────────────────────────
# Bridge
# ─────────────────────────────────────────────

class EEGBridge:
    """Real-time OSC-to-OSC bridge: Muse 2 → analysis → actuator."""

    def __init__(self, osc_client, mode, harmonic_multiplier, update_rate,
                 window_seconds=1.0):
        self.client = osc_client
        self.mode = mode
        self.harmonic_multiplier = harmonic_multiplier
        self.update_interval = 1.0 / update_rate
        self.window_size = int(window_seconds * SAMPLING_RATE)
        self.fundamental = PB.get("fundamental_hz", 64.0)

        # Ring buffer per channel
        self.buffers = {ch: np.zeros(self.window_size) for ch in CHANNELS}
        self.write_pos = 0
        self.samples_received = 0
        self.running = False

        # Note state
        self.note_id = 0
        self.note_active = False

        # Adaptive power normalization
        self.power_history = []

        # Lock for thread safety
        self._lock = threading.Lock()

        # Stats
        self.last_result = {}
        self.updates_sent = 0

    # ─── OSC Input Handler ───

    def eeg_handler(self, address, *args):
        """Handle incoming /muse/eeg samples."""
        if not self.running:
            return
        with self._lock:
            for i, ch in enumerate(CHANNELS):
                if i < len(args):
                    self.buffers[ch][self.write_pos % self.window_size] = float(args[i])
            self.write_pos += 1
            self.samples_received += 1

    # ─── OSC Output ───

    def send_note(self, freq, velocity):
        """Send /fnote to actuator."""
        freq = clamp_frequency(freq)
        velocity = float(np.clip(velocity, 0, 127))
        self.note_id += 1

        # Release previous note
        if self.note_active:
            self.client.send_message("/fnote/rel", [0.0, 0.0, self.note_id - 1])

        self.client.send_message("/fnote", [freq, velocity, self.note_id])
        self.note_active = True
        return freq, velocity

    def send_panic(self):
        """Send /allnotesoff."""
        self.client.send_message("/allnotesoff", [])
        self.note_active = False

    # ─── Analysis ───

    def get_window(self):
        """Get the current analysis window as (n_channels, n_samples) array."""
        with self._lock:
            data = np.array([
                np.roll(self.buffers[ch], -self.write_pos % self.window_size)
                for ch in CHANNELS
            ])
        return data

    def update_power_range(self, power):
        self.power_history.append(power)
        if len(self.power_history) > 50:
            self.power_history.pop(0)
        if len(self.power_history) > 5:
            self.p_low = float(np.percentile(self.power_history, 10))
            self.p_high = float(np.percentile(self.power_history, 90))
        else:
            self.p_low, self.p_high = 0.0, max(power, 1.0)

    def analyze_and_send(self):
        """Run analysis on current window and send to actuator."""
        if self.samples_received < self.window_size // 2:
            return None  # Not enough data yet

        data = self.get_window()
        avg = np.mean(data, axis=0)

        if self.mode == "spectral":
            peak_freq, peak_power = find_dominant_frequency(avg, SAMPLING_RATE)
            actuator_freq = peak_freq * self.harmonic_multiplier
            self.update_power_range(peak_power)
            vel = map_to_velocity(peak_power, self.p_low, self.p_high, 30, 127)
            freq_sent, vel_sent = self.send_note(actuator_freq, vel)
            self.last_result = {
                "eeg_freq": peak_freq, "actuator_freq": freq_sent, "vel": vel_sent
            }

        elif self.mode == "band_power":
            powers = compute_band_powers(avg, SAMPLING_RATE)
            dominant = max(powers, key=powers.get)
            band_center = np.mean(BANDS[dominant])
            actuator_freq = band_center * self.harmonic_multiplier
            self.update_power_range(powers[dominant])
            vel = map_to_velocity(powers[dominant], self.p_low, self.p_high, 30, 127)
            freq_sent, vel_sent = self.send_note(actuator_freq, vel)
            self.last_result = {
                "band": dominant, "actuator_freq": freq_sent, "vel": vel_sent
            }

        elif self.mode == "concentration":
            # Use frontal channels (AF7=idx1, AF8=idx2)
            frontal = np.mean(data[1:3], axis=0)
            powers = compute_band_powers(frontal, SAMPLING_RATE)
            score = compute_concentration(powers)
            actuator_freq = self.fundamental * 5  # H5
            vel = map_to_velocity(score, 20, 80, 30, 127)
            freq_sent, vel_sent = self.send_note(actuator_freq, vel)
            self.last_result = {
                "score": score, "actuator_freq": freq_sent, "vel": vel_sent
            }

        self.updates_sent += 1
        return self.last_result

    # ─── Main loop ───

    def run_loop(self):
        """Analysis/send loop running at update_rate Hz."""
        while self.running:
            result = self.analyze_and_send()
            if result:
                if self.mode == "spectral":
                    print(
                        f"\r  ⚡ EEG {result.get('eeg_freq', 0):5.1f} Hz → "
                        f"Actuator {result.get('actuator_freq', 0):6.0f} Hz  "
                        f"vel={result.get('vel', 0):3.0f}  "
                        f"[{self.updates_sent} updates]",
                        end="", flush=True
                    )
                elif self.mode == "band_power":
                    print(
                        f"\r  ⚡ {result.get('band', '?'):>5s} → "
                        f"Actuator {result.get('actuator_freq', 0):6.0f} Hz  "
                        f"vel={result.get('vel', 0):3.0f}  "
                        f"[{self.updates_sent} updates]",
                        end="", flush=True
                    )
                elif self.mode == "concentration":
                    score = result.get("score", 0)
                    bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
                    print(
                        f"\r  ⚡ Focus [{bar}] {score:3.0f}%  "
                        f"vel={result.get('vel', 0):3.0f}  "
                        f"[{self.updates_sent} updates]",
                        end="", flush=True
                    )
            elif self.samples_received > 0:
                print(
                    f"\r  Buffering... {self.samples_received}/{self.window_size // 2} samples",
                    end="", flush=True
                )
            time.sleep(self.update_interval)


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Real-time Muse 2 EEG → harmonic surface actuator bridge"
    )
    parser.add_argument(
        "--actuator-ip", required=True,
        help="ESP32 Beacon IP address"
    )
    parser.add_argument(
        "--actuator-port", type=int, default=53280,
        help="ESP32 OSC port (default: 53280)"
    )
    parser.add_argument(
        "--listen-port", type=int,
        default=REC.get("osc_port", 5000),
        help="Port to listen for Muse 2 OSC (default: 5000)"
    )
    parser.add_argument(
        "--mode", choices=["spectral", "band_power", "concentration"],
        default="spectral",
        help="Mapping mode (default: spectral)"
    )
    parser.add_argument(
        "--update-rate", type=float, default=2.0,
        help="How many times per second to update the actuator (default: 2 Hz)"
    )
    parser.add_argument(
        "--harmonic-multiplier", type=int,
        default=PB.get("harmonic_multiplier", 32),
        help="EEG freq × this = actuator freq (default: 32)"
    )
    parser.add_argument(
        "--window", type=float, default=1.0,
        help="Analysis window in seconds (default: 1.0)"
    )
    args = parser.parse_args()

    # OSC output client (to actuator)
    osc_out = udp_client.SimpleUDPClient(args.actuator_ip, args.actuator_port)

    # Bridge engine
    bridge = EEGBridge(
        osc_client=osc_out,
        mode=args.mode,
        harmonic_multiplier=args.harmonic_multiplier,
        update_rate=args.update_rate,
        window_seconds=args.window,
    )

    # OSC input server (from Muse 2)
    disp = dispatcher.Dispatcher()
    disp.map("/muse/eeg", bridge.eeg_handler)
    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.listen_port), disp)

    # Ctrl+C
    def signal_handler(sig, frame):
        print("\n\n  Stopping bridge...")
        bridge.running = False
        bridge.send_panic()
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # Banner
    print(f"\n{'='*55}")
    print(f"  ⚡ EEG Bridge — Muse 2 → Harmonic Surface (LIVE)")
    print(f"{'='*55}")
    print(f"  IN:   /muse/eeg @ 0.0.0.0:{args.listen_port}")
    print(f"  OUT:  /fnote    @ {args.actuator_ip}:{args.actuator_port}")
    print(f"  Mode: {args.mode}")
    print(f"  Rate: {args.update_rate} Hz ({1/args.update_rate*1000:.0f}ms)")
    print(f"  Window: {args.window}s ({int(args.window * SAMPLING_RATE)} samples)")
    print(f"{'='*55}")
    print(f"  Waiting for EEG data... (Ctrl+C to stop)\n")

    # Start
    bridge.running = True

    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    try:
        bridge.run_loop()
    except KeyboardInterrupt:
        pass

    bridge.send_panic()
    server.shutdown()
    print(f"\n  Done. Sent {bridge.updates_sent} updates.\n")


if __name__ == "__main__":
    main()
