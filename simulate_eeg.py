"""
EEG Simulator — Sends mock /muse/eeg OSC to test the harmonic bridge

Generates realistic EEG-like data that cycles through brain states so
you can hear the harmonic fluctuations on Surge XT without the Muse 2.

Brain states cycle every ~15 seconds:
  1. Relaxed    — strong alpha (10 Hz), eyes-closed calm
  2. Focused    — strong beta (20 Hz), concentrated attention
  3. Meditative — strong theta (6 Hz), deep relaxation
  4. Alert      — strong gamma (40 Hz), peak awareness
  5. Mixed      — balanced bands, normal waking state

Usage:
    python simulate_eeg.py                     # default: send to localhost:5000
    python simulate_eeg.py --ip 192.168.1.100  # send to another machine
    python simulate_eeg.py --speed 0.5         # slower transitions
"""

import argparse
import math
import time
import sys
import numpy as np
from pythonosc import udp_client

SAMPLING_RATE = 256
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

# ─── Brain state definitions ───
# Each state has per-channel band amplitudes (µV)
# Real EEG is ~10-100 µV; we exaggerate dominant bands for clarity

BRAIN_STATES = [
    {
        "name": "😌 Relaxed (alpha dominant)",
        "duration": 15.0,
        "channels": {
            #             delta  theta  alpha   beta  gamma
            "TP9":  {"delta": 15, "theta": 10, "alpha": 60, "beta": 8,  "gamma": 3},
            "AF7":  {"delta": 12, "theta": 8,  "alpha": 70, "beta": 10, "gamma": 3},
            "AF8":  {"delta": 12, "theta": 8,  "alpha": 65, "beta": 12, "gamma": 3},
            "TP10": {"delta": 15, "theta": 10, "alpha": 55, "beta": 8,  "gamma": 3},
        }
    },
    {
        "name": "🎯 Focused (beta dominant)",
        "duration": 15.0,
        "channels": {
            "TP9":  {"delta": 8,  "theta": 5,  "alpha": 15, "beta": 55, "gamma": 12},
            "AF7":  {"delta": 6,  "theta": 5,  "alpha": 10, "beta": 70, "gamma": 15},
            "AF8":  {"delta": 6,  "theta": 5,  "alpha": 12, "beta": 65, "gamma": 18},
            "TP10": {"delta": 8,  "theta": 5,  "alpha": 15, "beta": 50, "gamma": 10},
        }
    },
    {
        "name": "🧘 Meditative (theta dominant)",
        "duration": 15.0,
        "channels": {
            "TP9":  {"delta": 20, "theta": 60, "alpha": 25, "beta": 5,  "gamma": 2},
            "AF7":  {"delta": 18, "theta": 55, "alpha": 30, "beta": 5,  "gamma": 2},
            "AF8":  {"delta": 18, "theta": 50, "alpha": 28, "beta": 6,  "gamma": 2},
            "TP10": {"delta": 20, "theta": 65, "alpha": 22, "beta": 5,  "gamma": 2},
        }
    },
    {
        "name": "⚡ Alert (gamma burst)",
        "duration": 10.0,
        "channels": {
            "TP9":  {"delta": 5,  "theta": 5,  "alpha": 10, "beta": 30, "gamma": 50},
            "AF7":  {"delta": 5,  "theta": 5,  "alpha": 8,  "beta": 25, "gamma": 60},
            "AF8":  {"delta": 5,  "theta": 5,  "alpha": 8,  "beta": 25, "gamma": 55},
            "TP10": {"delta": 5,  "theta": 5,  "alpha": 10, "beta": 30, "gamma": 45},
        }
    },
    {
        "name": "🌊 Mixed (balanced, drifting)",
        "duration": 15.0,
        "channels": {
            "TP9":  {"delta": 20, "theta": 20, "alpha": 25, "beta": 20, "gamma": 10},
            "AF7":  {"delta": 18, "theta": 22, "alpha": 20, "beta": 25, "gamma": 12},
            "AF8":  {"delta": 18, "theta": 18, "alpha": 22, "beta": 28, "gamma": 15},
            "TP10": {"delta": 20, "theta": 20, "alpha": 23, "beta": 18, "gamma": 8},
        }
    },
    {
        "name": "🧠→👈 Left hemisphere dominant",
        "duration": 12.0,
        "channels": {
            "TP9":  {"delta": 10, "theta": 15, "alpha": 60, "beta": 50, "gamma": 20},
            "AF7":  {"delta": 8,  "theta": 12, "alpha": 65, "beta": 55, "gamma": 22},
            "AF8":  {"delta": 10, "theta": 10, "alpha": 15, "beta": 12, "gamma": 5},
            "TP10": {"delta": 12, "theta": 10, "alpha": 12, "beta": 10, "gamma": 3},
        }
    },
    {
        "name": "🧠→👉 Right hemisphere dominant",
        "duration": 12.0,
        "channels": {
            "TP9":  {"delta": 10, "theta": 10, "alpha": 12, "beta": 10, "gamma": 3},
            "AF7":  {"delta": 12, "theta": 10, "alpha": 15, "beta": 12, "gamma": 5},
            "AF8":  {"delta": 8,  "theta": 12, "alpha": 65, "beta": 55, "gamma": 22},
            "TP10": {"delta": 10, "theta": 15, "alpha": 60, "beta": 50, "gamma": 20},
        }
    },
]

# EEG band center frequencies
BAND_FREQS = {
    "delta": 2.5,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 40.0,
}


def generate_sample(t, channel_amps, noise_level=5.0):
    """Generate one EEG sample as a sum of band oscillations + noise."""
    value = 0.0
    for band, freq in BAND_FREQS.items():
        amp = channel_amps.get(band, 0.0)
        # Each band is a sine with slight random phase wobble
        phase_wobble = np.random.uniform(-0.1, 0.1)
        value += amp * math.sin(2 * math.pi * freq * t + phase_wobble)
    # Add realistic pink noise
    value += np.random.normal(0, noise_level)
    return value


def interpolate_state(state_a, state_b, t):
    """Smoothly interpolate between two brain states."""
    result = {}
    for ch in CHANNELS:
        result[ch] = {}
        for band in BAND_FREQS:
            a = state_a["channels"][ch].get(band, 0)
            b = state_b["channels"][ch].get(band, 0)
            # Smooth cosine interpolation
            factor = 0.5 * (1 - math.cos(math.pi * t))
            result[ch][band] = a + (b - a) * factor
    return result


def main():
    parser = argparse.ArgumentParser(
        description="EEG Simulator — mock Muse 2 brain activity for testing"
    )
    parser.add_argument("--ip", default="127.0.0.1",
                        help="Target IP for /muse/eeg (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Target port (default: 5000)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier for state transitions (default: 1.0)")
    parser.add_argument("--transition", type=float, default=3.0,
                        help="Seconds to blend between states (default: 3.0)")
    args = parser.parse_args()

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    dt = 1.0 / SAMPLING_RATE

    print(f"\n{'='*55}")
    print(f"  🧠 EEG Simulator — Mock Brain Activity")
    print(f"{'='*55}")
    print(f"  Target:     {args.ip}:{args.port}")
    print(f"  Rate:       {SAMPLING_RATE} Hz")
    print(f"  Speed:      {args.speed}x")
    print(f"  Transition: {args.transition}s blend")
    print(f"  States:     {len(BRAIN_STATES)} brain states cycling")
    print(f"{'='*55}")

    state_idx = 0
    t = 0.0
    state_timer = 0.0
    sample_count = 0

    try:
        while True:
            current_state = BRAIN_STATES[state_idx]
            next_state = BRAIN_STATES[(state_idx + 1) % len(BRAIN_STATES)]
            state_duration = current_state["duration"] / args.speed
            transition_time = args.transition / args.speed

            # Are we in the transition zone at the end of this state?
            remaining = state_duration - state_timer
            if remaining < transition_time and remaining > 0:
                blend = 1.0 - (remaining / transition_time)
                amps = interpolate_state(current_state, next_state, blend)
            else:
                amps = {ch: current_state["channels"][ch] for ch in CHANNELS}

            # Generate samples for all 4 channels
            values = [generate_sample(t, amps[ch]) for ch in CHANNELS]
            client.send_message("/muse/eeg", values)

            # Also send good horseshoe quality (all sensors connected)
            if sample_count % 64 == 0:
                client.send_message("/muse/elements/horseshoe", [1.0, 1.0, 1.0, 1.0])

            t += dt
            state_timer += dt
            sample_count += 1

            # State transition
            if state_timer >= state_duration:
                state_timer = 0.0
                state_idx = (state_idx + 1) % len(BRAIN_STATES)
                new_state = BRAIN_STATES[state_idx]
                print(f"\n  → {new_state['name']}  ({new_state['duration']/args.speed:.0f}s)")

            # Status display
            if sample_count % 256 == 0:
                band_display = "  ".join(
                    f"{b}={amps['AF7'].get(b, 0):3.0f}"
                    for b in ["theta", "alpha", "beta", "gamma"]
                )
                print(f"\r  t={t:6.1f}s  {band_display}", end="", flush=True)

            time.sleep(dt)

    except KeyboardInterrupt:
        print(f"\n\n  Done. Sent {sample_count} samples ({t:.1f}s).\n")


if __name__ == "__main__":
    main()
