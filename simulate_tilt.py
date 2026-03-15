"""
Tilt Simulator — Cycles through alpha/beta stages for cymatic observation

Designed to pair with muse_bridge.py in tilt mode. Sends mock /muse/eeg
OSC that moves through clear brain states, showing how the harmonic gain
curve tilts from warm (lower harmonics boosted) to bright (upper boosted).

Stages:
  1. DEEP RELAXED  — alpha=80, beta=10  → tilt ≈ +0.78  (warm, H1-H2 boosted)
  2. MILD RELAXED  — alpha=50, beta=25  → tilt ≈ +0.33  (slightly warm)
  3. NEUTRAL       — alpha=35, beta=35  → tilt ≈  0.00  (flat, no modulation)
  4. MILD FOCUSED  — alpha=25, beta=50  → tilt ≈ -0.33  (slightly bright)
  5. DEEP FOCUSED  — alpha=10, beta=80  → tilt ≈ -0.78  (bright, H4-H5 boosted)
  6. SWEEP UP      — continuous ramp from focused → relaxed (10s)
  7. SWEEP DOWN    — continuous ramp from relaxed → focused (10s)
  8. OSCILLATION   — alpha/beta seesaw at ~0.2 Hz (breathing rhythm)

The display shows expected tilt value and a gain curve preview so you
can correlate what you see on the cymatic mirror with brain state.

Usage:
    python simulate_tilt.py                        # defaults
    python simulate_tilt.py --speed 0.5            # slower for careful observation
    python simulate_tilt.py --depth 0.3            # match your bridge depth for preview
    python simulate_tilt.py --stages 1,2,3,4,5     # only specific stages
"""

import argparse
import math
import time
import numpy as np
from pythonosc import udp_client

SAMPLING_RATE = 256
CHANNELS = ["TP9", "AF7", "AF8", "TP10"]

BAND_FREQS = {
    "delta": 2.5,
    "theta": 6.0,
    "alpha": 10.0,
    "beta": 20.0,
    "gamma": 40.0,
}

DEFAULT_TILT_WEIGHTS = [-0.8, -0.4, 0.0, 0.4, 0.8]


# ─── Stage Definitions ───
# Only alpha and beta matter for tilt mode; other bands provide realistic background.

STAGES = [
    {
        "id": 1,
        "name": "DEEP RELAXED",
        "description": "Eyes closed, calm — strong alpha, minimal beta",
        "duration": 12.0,
        "frontal": {"delta": 12, "theta": 10, "alpha": 80, "beta": 10, "gamma": 3},
        "temporal": {"delta": 15, "theta": 12, "alpha": 60, "beta": 8, "gamma": 3},
    },
    {
        "id": 2,
        "name": "MILD RELAXED",
        "description": "Resting aware — alpha still dominant",
        "duration": 10.0,
        "frontal": {"delta": 12, "theta": 10, "alpha": 50, "beta": 25, "gamma": 5},
        "temporal": {"delta": 15, "theta": 12, "alpha": 40, "beta": 20, "gamma": 4},
    },
    {
        "id": 3,
        "name": "NEUTRAL",
        "description": "Balanced — alpha and beta roughly equal",
        "duration": 10.0,
        "frontal": {"delta": 12, "theta": 10, "alpha": 35, "beta": 35, "gamma": 8},
        "temporal": {"delta": 15, "theta": 12, "alpha": 30, "beta": 28, "gamma": 6},
    },
    {
        "id": 4,
        "name": "MILD FOCUSED",
        "description": "Light concentration — beta rising",
        "duration": 10.0,
        "frontal": {"delta": 10, "theta": 8, "alpha": 25, "beta": 50, "gamma": 12},
        "temporal": {"delta": 12, "theta": 8, "alpha": 20, "beta": 40, "gamma": 10},
    },
    {
        "id": 5,
        "name": "DEEP FOCUSED",
        "description": "Intense focus — strong beta, minimal alpha",
        "duration": 12.0,
        "frontal": {"delta": 8, "theta": 5, "alpha": 10, "beta": 80, "gamma": 15},
        "temporal": {"delta": 10, "theta": 6, "alpha": 12, "beta": 55, "gamma": 12},
    },
    {
        "id": 6,
        "name": "SWEEP UP",
        "description": "Continuous ramp: focused --> relaxed (10s linear)",
        "duration": 10.0,
        "sweep": True,
        "sweep_from": {"delta": 8, "theta": 5, "alpha": 10, "beta": 80, "gamma": 15},
        "sweep_to": {"delta": 12, "theta": 10, "alpha": 80, "beta": 10, "gamma": 3},
    },
    {
        "id": 7,
        "name": "SWEEP DOWN",
        "description": "Continuous ramp: relaxed --> focused (10s linear)",
        "duration": 10.0,
        "sweep": True,
        "sweep_from": {"delta": 12, "theta": 10, "alpha": 80, "beta": 10, "gamma": 3},
        "sweep_to": {"delta": 8, "theta": 5, "alpha": 10, "beta": 80, "gamma": 15},
    },
    {
        "id": 8,
        "name": "OSCILLATION",
        "description": "Alpha/beta seesaw at ~0.2 Hz (breathing rhythm)",
        "duration": 20.0,
        "oscillating": True,
        "osc_freq": 0.2,
        "osc_center_alpha": 40,
        "osc_center_beta": 40,
        "osc_swing": 35,
    },
]


def generate_sample(t, channel_amps, noise_level=5.0):
    """Generate one EEG sample as sum of band oscillations + noise."""
    value = 0.0
    for band, freq in BAND_FREQS.items():
        amp = channel_amps.get(band, 0.0)
        phase_wobble = np.random.uniform(-0.1, 0.1)
        value += amp * math.sin(2 * math.pi * freq * t + phase_wobble)
    value += np.random.normal(0, noise_level)
    return value


def get_stage_amps(stage, stage_progress):
    """Get per-channel amplitudes for current stage and progress (0-1)."""
    if stage.get("sweep"):
        # Linear interpolation between sweep_from and sweep_to
        amps_from = stage["sweep_from"]
        amps_to = stage["sweep_to"]
        frontal = {b: amps_from[b] + (amps_to[b] - amps_from[b]) * stage_progress
                   for b in BAND_FREQS}
        temporal = dict(frontal)
        temporal["theta"] += 2
        temporal["gamma"] -= 2

    elif stage.get("oscillating"):
        # Sinusoidal seesaw between alpha and beta
        freq = stage["osc_freq"]
        center_a = stage["osc_center_alpha"]
        center_b = stage["osc_center_beta"]
        swing = stage["osc_swing"]
        osc_phase = stage_progress * stage["duration"] * freq * 2 * math.pi
        osc_val = math.sin(osc_phase)
        frontal = {
            "delta": 10,
            "theta": 8,
            "alpha": center_a + swing * osc_val,
            "beta": center_b - swing * osc_val,
            "gamma": 8,
        }
        temporal = {
            "delta": 12,
            "theta": 10,
            "alpha": frontal["alpha"] * 0.75,
            "beta": frontal["beta"] * 0.75,
            "gamma": 5,
        }

    else:
        frontal = stage["frontal"]
        temporal = stage["temporal"]

    return {
        "AF7": frontal,
        "AF8": {b: v * (0.95 + np.random.uniform(0, 0.1)) for b, v in frontal.items()},
        "TP9": temporal,
        "TP10": {b: v * (0.95 + np.random.uniform(0, 0.1)) for b, v in temporal.items()},
    }


def expected_tilt(frontal_amps):
    """Compute expected tilt from frontal alpha/beta."""
    alpha = frontal_amps.get("alpha", 0)
    beta = frontal_amps.get("beta", 0)
    total = alpha + beta
    if total < 1e-10:
        return 0.0
    return (alpha - beta) / total


def preview_gains(tilt, depth, base_gains, weights):
    """Preview what the gain curve looks like at this tilt."""
    gains = []
    for i in range(5):
        base = base_gains[i]
        mod = max(-1.0, min(1.0, tilt * weights[i]))
        g = base * (1.0 + depth * mod)
        gains.append(max(0.0, min(1.0, g)))
    return gains


def gain_bar(gain, width=10):
    fill = int(gain * width)
    return "\u2588" * fill + "\u2591" * (width - fill)


def main():
    parser = argparse.ArgumentParser(
        description="Tilt Simulator — stage-based EEG mock for cymatic observation"
    )
    parser.add_argument("--ip", default="127.0.0.1",
                        help="Target IP for /muse/eeg (default: 127.0.0.1)")
    parser.add_argument("--port", type=int, default=5000,
                        help="Target port (default: 5000)")
    parser.add_argument("--speed", type=float, default=1.0,
                        help="Speed multiplier (default: 1.0)")
    parser.add_argument("--depth", type=float, default=0.20,
                        help="Modulation depth for gain preview (default: 0.20)")
    parser.add_argument("--base-gains", type=str, default="0.80,0.67,0.68,0.75,1.00",
                        help="Base gains H1-H5 for preview (default: 0.80,0.67,0.68,0.75,1.00)")
    parser.add_argument("--stages", type=str, default=None,
                        help="Comma-separated stage IDs to run (default: all)")
    parser.add_argument("--loop", action="store_true", default=True,
                        help="Loop through stages (default: true)")
    parser.add_argument("--no-loop", action="store_true", default=False,
                        help="Run through stages once then stop")
    parser.add_argument("--transition", type=float, default=2.0,
                        help="Blend time between stages in seconds (default: 2.0)")
    args = parser.parse_args()

    base_gains = [float(x) for x in args.base_gains.split(",")]
    while len(base_gains) < 5:
        base_gains.append(0.8)

    do_loop = args.loop and not args.no_loop

    # Filter stages if requested
    if args.stages:
        stage_ids = [int(x) for x in args.stages.split(",")]
        stages = [s for s in STAGES if s["id"] in stage_ids]
    else:
        stages = list(STAGES)

    if not stages:
        print("ERROR: No valid stages selected")
        return

    client = udp_client.SimpleUDPClient(args.ip, args.port)
    dt = 1.0 / SAMPLING_RATE
    weights = DEFAULT_TILT_WEIGHTS

    print(f"\n{'='*70}")
    print(f"  Tilt Simulator — Cymatic Observation Stages")
    print(f"{'='*70}")
    print(f"  Target:     {args.ip}:{args.port}")
    print(f"  Speed:      {args.speed}x")
    print(f"  Depth:      +/-{int(args.depth*100)}% (for gain preview)")
    print(f"  Base:       {' '.join(f'H{i+1}={g:.2f}' for i, g in enumerate(base_gains))}")
    print(f"  Stages:     {len(stages)}")
    print(f"  Loop:       {do_loop}")
    print(f"{'='*70}")
    print(f"  Stages to cycle:")
    for s in stages:
        print(f"    [{s['id']}] {s['name']:16s} — {s['description']}")
    print(f"{'='*70}\n")

    stage_idx = 0
    t = 0.0
    stage_timer = 0.0
    sample_count = 0
    prev_amps = None

    try:
        while True:
            stage = stages[stage_idx]
            stage_duration = stage["duration"] / args.speed
            transition_time = args.transition / args.speed
            stage_progress = min(stage_timer / stage_duration, 1.0)

            # Get target amplitudes for this stage
            target_amps = get_stage_amps(stage, stage_progress)

            # Crossfade from previous stage during transition window
            if prev_amps and stage_timer < transition_time:
                blend = stage_timer / transition_time
                factor = 0.5 * (1 - math.cos(math.pi * blend))
                amps = {}
                for ch in CHANNELS:
                    amps[ch] = {}
                    for band in BAND_FREQS:
                        a = prev_amps[ch].get(band, 0)
                        b = target_amps[ch].get(band, 0)
                        amps[ch][band] = a + (b - a) * factor
            else:
                amps = target_amps

            # Generate and send samples
            values = [generate_sample(t, amps[ch]) for ch in CHANNELS]
            client.send_message("/muse/eeg", values)

            if sample_count % 64 == 0:
                client.send_message("/muse/elements/horseshoe", [1.0, 1.0, 1.0, 1.0])

            t += dt
            stage_timer += dt
            sample_count += 1

            # Status display every 256 samples (~1s)
            if sample_count % 256 == 0:
                frontal = amps.get("AF7", {})
                alpha_val = frontal.get("alpha", 0)
                beta_val = frontal.get("beta", 0)
                tilt = expected_tilt(frontal)
                gains = preview_gains(tilt, args.depth, base_gains, weights)

                # Direction indicator
                if tilt > 0.15:
                    direction = "<<< WARM "
                elif tilt < -0.15:
                    direction = " BRIGHT>>>"
                else:
                    direction = "  neutral "

                # Gain bars
                gain_display = "  ".join(
                    f"H{i+1}[{gain_bar(g, 6)}]{g:.2f}"
                    for i, g in enumerate(gains)
                )

                stage_pct = int(stage_progress * 100)
                print(
                    f"\r  [{stage['id']}] {stage['name']:16s} {stage_pct:3d}%  "
                    f"a={alpha_val:4.0f} b={beta_val:4.0f}  "
                    f"tilt={tilt:+.2f} {direction}  "
                    f"{gain_display}",
                    end="", flush=True
                )

            # Stage transition
            if stage_timer >= stage_duration:
                prev_amps = amps
                stage_timer = 0.0
                stage_idx += 1

                if stage_idx >= len(stages):
                    if not do_loop:
                        break
                    stage_idx = 0
                    print(f"\n\n  --- Looping ---")

                next_stage = stages[stage_idx]
                print(f"\n\n  >>> Stage {next_stage['id']}: {next_stage['name']}")
                print(f"      {next_stage['description']}")
                print(f"      Duration: {next_stage['duration']/args.speed:.0f}s")
                print()

            time.sleep(dt)

    except KeyboardInterrupt:
        pass

    print(f"\n\n  Done. Sent {sample_count} samples ({t:.1f}s).\n")


if __name__ == "__main__":
    main()
