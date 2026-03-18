# EEG-Driven Gain Modulation: Spectral Tilt

## Overview

The Muse 2 EEG headband controls the **gain (amplitude)** of each harmonic in
the harmonic_shaper synthesizer. The alpha/beta power ratio tilts the gain
curve across the harmonic series — relaxation emphasizes one end, focus
emphasizes the other. A Launchpad MIDI slider controls how much the EEG data
is allowed to modify the gains.

This is separate from and complementary to [phase control](PHASE_CONTROL_ANALYSIS.md).
Phase controls *how harmonics align in time* (the shape of the interference
pattern). Gain controls *how loud each harmonic is* (the balance of the tonal
spectrum and the intensity of the cymatic figure).

---

## The Tilt Model

### Alpha/Beta Ratio

The frontal sensors (AF7 and AF8) measure two key brain rhythms:

- **Alpha (8-13 Hz)**: Dominant during relaxation, calm, eyes-closed rest
- **Beta (13-30 Hz)**: Dominant during concentration, active thinking, focus

The bridge computes a single scalar "tilt" from -1 to +1:

```
alpha_power = mean(AF7_alpha, AF8_alpha)
beta_power  = mean(AF7_beta, AF8_beta)

tilt = (alpha - beta) / (alpha + beta)
```

| Brain state | Alpha | Beta | Tilt | Meaning |
|-------------|-------|------|------|---------|
| Deep relaxation | High | Low | +0.7 to +1.0 | Strongly relaxed |
| Calm, resting | Moderate | Low | +0.2 to +0.5 | Mildly relaxed |
| Neutral / balanced | Equal | Equal | ~0.0 | Neither state |
| Light focus | Low | Moderate | -0.2 to -0.5 | Mildly focused |
| Intense focus | Low | High | -0.7 to -1.0 | Strongly focused |

The tilt is smoothed with an exponential moving average (default alpha=0.25)
to prevent jitter from EEG artifacts.

### Tilt Weights

Each harmonic has a fixed weight that determines how the tilt affects it:

```
TILT_WEIGHTS = [-0.8, -0.4, 0.0, 0.4, 0.8]
                 H1    H2    H3    H4    H5
```

The weight and the tilt multiply to produce a **modulator** for each harmonic:

```
modulator = tilt × weight
```

| | H1 (w=-0.8) | H2 (w=-0.4) | H3 (w=0.0) | H4 (w=+0.4) | H5 (w=+0.8) |
|---|---|---|---|---|---|
| Relaxed (tilt=+0.6) | -0.48 | -0.24 | 0.0 | +0.24 | +0.48 |
| Focused (tilt=-0.6) | +0.48 | +0.24 | 0.0 | -0.24 | -0.48 |
| Neutral (tilt=0.0) | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

H3 is the pivot point (weight=0, never changes). The curve tilts around it.

---

## The Gain Formula

```
effective_gain = base × (1 + gain_depth × modulator)
effective_gain = clamp(effective_gain, 0.0, 1.0)
```

Where:
- `base` = Launchpad-set gain for this harmonic (always respected)
- `gain_depth` = 0.0 to 1.0, controlled by the Launchpad slider
- `modulator` = `clamp(tilt × weight, -1.0, 1.0)`

### Example: Relaxed State, Depth = 50%

With `base = 0.8`, `gain_depth = 0.5`, and `tilt = +0.6`:

| Harmonic | Weight | Modulator | Gain Factor | Effective Gain |
|----------|--------|-----------|-------------|----------------|
| H1 | -0.8 | -0.48 | 1 + 0.5×(-0.48) = 0.76 | 0.8 × 0.76 = **0.61** |
| H2 | -0.4 | -0.24 | 1 + 0.5×(-0.24) = 0.88 | 0.8 × 0.88 = **0.70** |
| H3 | 0.0 | 0.0 | 1 + 0.5×(0.0) = 1.00 | 0.8 × 1.00 = **0.80** |
| H4 | +0.4 | +0.24 | 1 + 0.5×(+0.24) = 1.12 | 0.8 × 1.12 = **0.90** |
| H5 | +0.8 | +0.48 | 1 + 0.5×(+0.48) = 1.24 | 0.8 × 1.24 = **0.99** |

The gain curve tilts upward toward higher harmonics when relaxed.

### Example: Focused State, Depth = 50%

Same setup but `tilt = -0.6`:

| Harmonic | Weight | Modulator | Effective Gain |
|----------|--------|-----------|----------------|
| H1 | -0.8 | +0.48 | 0.8 × 1.24 = **0.99** |
| H2 | -0.4 | +0.24 | 0.8 × 1.12 = **0.90** |
| H3 | 0.0 | 0.0 | 0.8 × 1.00 = **0.80** |
| H4 | +0.4 | -0.24 | 0.8 × 0.88 = **0.70** |
| H5 | +0.8 | -0.48 | 0.8 × 0.76 = **0.61** |

The tilt reverses: lower harmonics are boosted, upper harmonics attenuated.

---

## The Launchpad Slider: Dynamic Gain Depth

### Why a Slider?

The `gain_depth` parameter controls *how much* the EEG data is allowed to
modify the gains. At depth=0, the Launchpad-set gains are untouched. At
depth=1.0, the EEG can swing gains up to ±80% from base (at the extreme
harmonics).

In performance, you may want to:
- Start with the slider at 0 (pure Launchpad control)
- Gradually introduce EEG influence during the session
- Pull back to 0 when adjusting base gains
- Push to maximum for a more responsive, brain-driven experience

### Signal Chain

```
Launchpad Mini
     |
     | MIDI CC1 (mod wheel, 0-127)
     v
midi_relay.py
     |
     | OSC /bridge/gain_depth [0-127]
     v
muse_bridge.py
     |
     | normalizes to 0.0-1.0
     | sets self.gain_depth
     v
gain formula: base × (1 + gain_depth × modulator)
```

The `midi_relay.py` script translates MIDI CC messages from the Launchpad
into OSC messages that `muse_bridge.py` understands. It's a thin bridge —
auto-detects the MIDI port, reads one CC number, forwards the raw value.

### What the Slider Percentage Means

| Slider % | gain_depth | Max gain swing (H1/H5) | Experience |
|----------|------------|------------------------|------------|
| 0% | 0.0 | No EEG effect | Pure manual control |
| 25% | ~0.25 | ±20% of base | Subtle, barely noticeable |
| 50% | ~0.50 | ±40% of base | Clear, responsive |
| 75% | ~0.75 | ±60% of base | Strong, dynamic |
| 100% | 1.0 | ±80% of base | Maximum EEG influence |

### Why Bounded Modulation

The gains always stay within a range defined by the base value and the depth.
If the Launchpad sets H3 to 0.6 and the slider is at 50%, the EEG can only
move H3 between ~0.36 and ~0.84 (depending on the tilt weight for H3, which
is 0.0 — so H3 actually doesn't move at all). The performer's base settings
are always respected.

---

## Cymatic Impact

### What You See

Unlike phase rotation (which changes the *shape* of the interference pattern),
gain modulation changes the *intensity* and *balance* of the harmonics:

- **Relaxed (upper harmonics boosted)**: The cymatic pattern shows more
  fine detail and complexity. Higher harmonics create tighter spatial
  features on the mirror.

- **Focused (lower harmonics boosted)**: The pattern simplifies. Lower
  harmonics create broad, flowing structures. The fundamental (H1) dominates.

- **Neutral**: The balance stays as the performer set it on the Launchpad.

The transitions are smooth (EMA-smoothed tilt, continuous gain output) and
the effect scales linearly with the slider position.

### Combined with Phase

When running in `--param both` mode, the gain tilt and phase rotation work
simultaneously. The cymatic figure's shape evolves (phase) while its tonal
balance shifts (gain). The two effects are independent:

- Phase velocity comes from individual band powers (theta, alpha, beta, gamma)
- Gain tilt comes from the alpha/beta ratio (a single frontal measure)

This means a person can have strong theta (rotating H2's phase) while being
alpha-dominant (tilting gains toward upper harmonics) — the effects don't
conflict.

---

## Configuration

### CLI Arguments (gain-related)

| Argument | Default | Description |
|----------|---------|-------------|
| `--param gain` | both | Gain-only mode |
| `--param both` | both | Combined phase + gain |
| `--depth 0.20` | 0.20 | Gain fraction in gain-only mode |
| `--gain-depth 0.20` | 0.0 (both) / 0.20 (gain) | Initial gain depth |

### OSC Messages

| Direction | Address | Payload | Description |
|-----------|---------|---------|-------------|
| **IN** | `/bridge/gain_depth` | `[0-127]` or `[0.0-1.0]` | Slider sets gain depth |
| **OUT** | `/shaper/harmonic/N/gain` | `[0.0-1.0]` | Effective gain per harmonic |

### Output Rate

Gains are sent at the EEG analysis rate (4 Hz) in normal operation. When a
heartbeat pulse is active (see [HEARTBEAT_PULSE.md](HEARTBEAT_PULSE.md)),
gains are sent at the fast output rate (30 Hz) for smooth pulse rendering.
