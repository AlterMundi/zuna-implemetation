"""
Generate a mock ZUNA-enhanced .fif file for testing the playback pipeline.

Creates a realistic 60-second EEG recording that simulates what ZUNA would
output: 4 Muse 2 channels (denoised) cycling through distinct brain states
so you can hear/feel different vibration patterns on the harmonic surface.

Brain state sequence (each ~15 seconds):
  1. Relaxed alpha   (10 Hz dominant)  → actuator ~320 Hz (H5)
  2. Focused beta    (20 Hz dominant)  → actuator ~640 Hz
  3. Meditative theta (6 Hz dominant)  → actuator ~192 Hz (H3)
  4. Alert gamma     (35 Hz dominant)  → actuator ~1120 Hz

Usage:
    python generate_mock_eeg.py
    python generate_mock_eeg.py --duration 30 --output enhanced/test_session.fif
"""

import argparse
from pathlib import Path
import numpy as np
import mne


CHANNELS = ["TP9", "AF7", "AF8", "TP10"]
SFREQ = 256

MUSE2_POSITIONS = {
    "TP9":  [-0.0694, -0.0326, -0.0140],
    "AF7":  [-0.0482,  0.0566,  0.0274],
    "AF8":  [ 0.0482,  0.0566,  0.0274],
    "TP10": [ 0.0694, -0.0326, -0.0140],
}

# Brain states: (name, dominant_freq_hz, amplitude_uv, secondary_freq, sec_amp)
STATES = [
    ("Relaxed (alpha)",    10.0, 60, 2.0, 15),
    ("Focused (beta)",     20.0, 50, 13.0, 30),
    ("Meditative (theta)",  6.0, 70, 10.0, 25),
    ("Alert (gamma)",      35.0, 35, 20.0, 20),
]


def generate_state_signal(duration_s, state_idx, channel_idx):
    """Generate EEG signal for a given brain state."""
    name, dom_freq, dom_amp, sec_freq, sec_amp = STATES[state_idx % len(STATES)]
    n_samples = int(duration_s * SFREQ)
    t = np.arange(n_samples) / SFREQ

    # Phase offset per channel for realism
    phase = channel_idx * 0.7

    signal = (
        dom_amp * np.sin(2 * np.pi * dom_freq * t + phase)
        + sec_amp * np.sin(2 * np.pi * sec_freq * t + phase * 1.3)
        + 10 * np.sin(2 * np.pi * 2.0 * t + phase * 0.5)   # delta background
        + 5 * np.random.randn(n_samples)                      # low noise (ZUNA-cleaned)
    )

    # Frontal channels slightly stronger
    if channel_idx in (1, 2):
        signal *= 1.15

    return signal


def main():
    parser = argparse.ArgumentParser(description="Generate mock ZUNA-enhanced EEG")
    parser.add_argument("--duration", type=int, default=60, help="Duration in seconds")
    parser.add_argument("--output", default="enhanced/mock_enhanced_eeg_raw.fif")
    args = parser.parse_args()

    state_duration = args.duration / len(STATES)
    total_samples = int(args.duration * SFREQ)

    print(f"\n  Generating {args.duration}s mock EEG ({len(STATES)} states × {state_duration:.0f}s each)\n")

    # Build signal for each channel
    data = np.zeros((len(CHANNELS), total_samples))
    for ch_idx in range(len(CHANNELS)):
        for state_idx in range(len(STATES)):
            start = int(state_idx * state_duration * SFREQ)
            end = int((state_idx + 1) * state_duration * SFREQ)
            end = min(end, total_samples)
            segment_dur = (end - start) / SFREQ

            segment = generate_state_signal(segment_dur, state_idx, ch_idx)
            data[ch_idx, start:start + len(segment)] = segment

    # Smooth transitions between states (1s crossfade)
    fade_samples = SFREQ  # 1 second
    for state_idx in range(1, len(STATES)):
        boundary = int(state_idx * state_duration * SFREQ)
        start = max(0, boundary - fade_samples // 2)
        end = min(total_samples, boundary + fade_samples // 2)
        window = np.linspace(0, 1, end - start)
        # Already concatenated, just smooth with a mild filter
        for ch_idx in range(len(CHANNELS)):
            data[ch_idx, start:end] *= (0.7 + 0.3 * np.sin(np.pi * window))

    # Convert µV to V (MNE expects SI)
    data *= 1e-6

    # Create MNE Raw object
    info = mne.create_info(CHANNELS, SFREQ, ch_types="eeg")
    raw = mne.io.RawArray(data, info, verbose=False)

    montage = mne.channels.make_dig_montage(ch_pos=MUSE2_POSITIONS, coord_frame="head")
    raw.set_montage(montage)

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw.save(str(output_path), overwrite=True, verbose=False)

    print(f"  State sequence:")
    for i, (name, freq, amp, _, _) in enumerate(STATES):
        t_start = i * state_duration
        t_end = (i + 1) * state_duration
        act_freq = freq * 32
        print(f"    [{t_start:5.0f}s - {t_end:5.0f}s]  {name:<22s}  {freq:5.1f} Hz → actuator {act_freq:.0f} Hz")

    print(f"\n  ✓ Saved: {output_path}")
    print(f"    {total_samples} samples, {args.duration}s, {len(CHANNELS)} channels")
    print(f"\n  Test with:")
    print(f"    python osc_playback.py --input {output_path} --ip <ESP32_IP> --mode spectral --epoch-length 2.5\n")


if __name__ == "__main__":
    main()
