"""
OSC Recorder — Captures Muse 2 EEG stream and saves to MNE .fif format

Records raw EEG data from Mind Monitor's OSC stream and saves it as
an MNE-compatible .fif file with proper Muse 2 electrode positions.
The .fif file can then be transferred to a GPU machine for ZUNA processing.

Usage:
    python osc_recorder.py --duration 300 --output recordings/session_001.fif
    python osc_recorder.py  # Records until Ctrl+C
"""

import argparse
import json
import signal
import sys
import time
import threading
from datetime import datetime
from pathlib import Path

import numpy as np
import mne
from pythonosc import dispatcher, osc_server

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

# ─────────────────────────────────────────────
# Muse 2 electrode positions (3D, MNE head coords)
# ─────────────────────────────────────────────

MUSE2_POSITIONS = CONFIG.get("muse2_positions", {
    "TP9":  [-0.0694, -0.0326, -0.0140],
    "AF7":  [-0.0482,  0.0566,  0.0274],
    "AF8":  [ 0.0482,  0.0566,  0.0274],
    "TP10": [ 0.0694, -0.0326, -0.0140],
})

CHANNELS = CONFIG.get("recorder", {}).get("channels", ["TP9", "AF7", "AF8", "TP10"])
SAMPLING_RATE = CONFIG.get("recorder", {}).get("sampling_rate", 256)
OSC_ADDRESS = CONFIG.get("recorder", {}).get("osc_address", "/muse/eeg")

# ─────────────────────────────────────────────
# Recorder
# ─────────────────────────────────────────────

class EEGRecorder:
    """Records incoming OSC EEG data and saves to .fif format."""

    def __init__(self):
        self.data = {ch: [] for ch in CHANNELS}
        self.sample_count = 0
        self.start_time = None
        self.running = False
        self._lock = threading.Lock()

    def osc_handler(self, address, *args):
        """Handle incoming /muse/eeg OSC messages."""
        if not self.running:
            return

        if self.start_time is None:
            self.start_time = time.time()

        with self._lock:
            for ch, val in zip(CHANNELS, args[:len(CHANNELS)]):
                self.data[ch].append(float(val))
            self.sample_count += 1

    def start(self):
        """Mark recorder as active."""
        self.running = True
        self.start_time = None
        self.sample_count = 0
        self.data = {ch: [] for ch in CHANNELS}
        print(f"Recording started — waiting for OSC data on {OSC_ADDRESS}...")

    def stop(self):
        """Mark recorder as stopped."""
        self.running = False

    def get_duration(self):
        """Get recording duration in seconds."""
        if self.sample_count == 0:
            return 0.0
        return self.sample_count / SAMPLING_RATE

    def save_fif(self, output_path):
        """Save recorded data as MNE .fif file with Muse 2 montage."""
        with self._lock:
            if self.sample_count == 0:
                print("ERROR: No data recorded. Check OSC connection.")
                return False

            # Build channel data array (n_channels × n_samples)
            data_array = np.array([self.data[ch] for ch in CHANNELS])

        # Ensure all channels have the same length (trim to shortest)
        min_len = min(len(self.data[ch]) for ch in CHANNELS)
        data_array = data_array[:, :min_len]

        # Convert from microvolts to volts (MNE expects SI units)
        data_array = data_array * 1e-6

        # Create MNE Info object
        info = mne.create_info(
            ch_names=CHANNELS,
            sfreq=SAMPLING_RATE,
            ch_types="eeg"
        )

        # Create RawArray
        raw = mne.io.RawArray(data_array, info, verbose=False)

        # Set Muse 2 montage with 3D positions
        montage_positions = {
            ch: pos for ch, pos in MUSE2_POSITIONS.items()
            if ch in CHANNELS
        }
        montage = mne.channels.make_dig_montage(
            ch_pos=montage_positions,
            coord_frame="head"
        )
        raw.set_montage(montage)

        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw.save(str(output_path), overwrite=True, verbose=False)

        duration = min_len / SAMPLING_RATE
        print(f"\n{'='*50}")
        print(f"  Recording saved successfully!")
        print(f"  File:     {output_path}")
        print(f"  Channels: {len(CHANNELS)} ({', '.join(CHANNELS)})")
        print(f"  Samples:  {min_len}")
        print(f"  Duration: {duration:.1f}s ({duration/60:.1f} min)")
        print(f"  Rate:     {SAMPLING_RATE} Hz")
        print(f"  Size:     {output_path.stat().st_size / 1024:.1f} KB")
        print(f"{'='*50}")
        return True


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Record Muse 2 EEG via OSC to MNE .fif format"
    )
    parser.add_argument(
        "--port", type=int,
        default=CONFIG.get("recorder", {}).get("osc_port", 5000),
        help="OSC listen port (default: 5000)"
    )
    parser.add_argument(
        "--duration", type=int, default=0,
        help="Recording duration in seconds (0 = until Ctrl+C)"
    )
    parser.add_argument(
        "--output", type=str,
        default=f"recordings/session_{datetime.now().strftime('%Y%m%d_%H%M%S')}.fif",
        help="Output .fif file path"
    )
    args = parser.parse_args()

    recorder = EEGRecorder()

    # Setup OSC server
    disp = dispatcher.Dispatcher()
    disp.map(OSC_ADDRESS, recorder.osc_handler)

    server = osc_server.ThreadingOSCUDPServer(("0.0.0.0", args.port), disp)

    # Handle Ctrl+C gracefully
    def signal_handler(sig, frame):
        print("\n\nStopping recording...")
        recorder.stop()
        server.shutdown()

    signal.signal(signal.SIGINT, signal_handler)

    # Start
    print(f"\n{'='*50}")
    print(f"  🧠 EEG Recorder — Muse 2 → .fif")
    print(f"{'='*50}")
    print(f"  Listening:  0.0.0.0:{args.port}")
    print(f"  OSC addr:   {OSC_ADDRESS}")
    print(f"  Channels:   {', '.join(CHANNELS)}")
    print(f"  Rate:       {SAMPLING_RATE} Hz")
    print(f"  Duration:   {'until Ctrl+C' if args.duration == 0 else f'{args.duration}s'}")
    print(f"  Output:     {args.output}")
    print(f"{'='*50}")
    print()

    recorder.start()

    # Run OSC server in background
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()

    # Progress display
    try:
        elapsed = 0
        while recorder.running:
            time.sleep(1)
            elapsed += 1

            if recorder.sample_count > 0:
                duration = recorder.get_duration()
                rate = recorder.sample_count / elapsed if elapsed > 0 else 0
                print(
                    f"\r  ⏺  {duration:.1f}s recorded | "
                    f"{recorder.sample_count} samples | "
                    f"{rate:.0f} samples/s",
                    end="", flush=True
                )

            # Auto-stop after duration
            if args.duration > 0 and elapsed >= args.duration:
                print("\n\nDuration reached, stopping...")
                recorder.stop()
                break
    except KeyboardInterrupt:
        recorder.stop()

    # Shutdown server
    server.shutdown()

    # Save
    if recorder.sample_count > 0:
        recorder.save_fif(args.output)
    else:
        print("\nNo data was received. Ensure Mind Monitor is streaming to this IP/port.")


if __name__ == "__main__":
    main()
