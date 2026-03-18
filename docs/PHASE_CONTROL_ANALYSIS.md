# EEG-Driven Phase Control: How the Muse 2 Shapes Cymatic Patterns

## Overview

The Muse 2 EEG headband controls the **phase offset** of each harmonic sine wave
in the harmonic_shaper synthesizer. Phase determines the *timing relationship*
between harmonics — when two sine waves have a specific phase relationship, they
constructively or destructively interfere, creating the distinct figures visible
on the cymatic mirror. The Muse turns brain rhythms into a slowly evolving
interference field.

---

## The Physical Setup

```
Muse 2 headband                    Launchpad Mini
(4 EEG sensors)                    (latches harmonics)
       |                                  |
  Mind Monitor app                        |
       |                                  |
  OSC /muse/eeg :5000              harmonic_beacon
       |                           (MIDI -> OSC)
       v                                  |
  muse_bridge.py ←──── hr_relay.py        |
  (EEG analysis +      (heartbeat)        |
   phase interp +                         |
   gain pulse)     midi_relay.py          |
       |           (Launchpad slider)     |
       |                                  |
       |  /shaper/harmonic/N/phase        |  /beacon/voice/on,off
       |  /shaper/harmonic/N/gain         |  OSC :9001
       |  OSC :9002                       |
       |                                  |
       +----------> harmonic_shaper <-----+
                    (additive synth)
                          |
                    sounddevice
                    (PortAudio)
                          |
                    two speakers
                          |
                    sealed tube
                          |
                    balloon + mirror
                          |
                    laser dot on wall
                    (cymatic figure)
```

The harmonic_shaper synthesizes pure sine waves. The Launchpad selects *which*
harmonics play (and at what frequency). The Muse controls *how those harmonics
relate to each other in phase* — which is what determines the shape of the
interference pattern. Optionally, a Fitbit heartbeat can overlay a rhythmic
gain pulse and a Launchpad slider can control EEG gain depth — see
[SESSION_GUIDE.md](SESSION_GUIDE.md) for all combinations.

---

## The Muse 2 Sensors

The Muse 2 has four dry EEG electrodes positioned on the forehead and behind
the ears:

```
            ┌──────────────────────┐
            │        SCALP         │
            │                      │
            │   AF7 ●      ● AF8  │   Frontal (forehead)
            │   left        right  │
            │                      │
            │                      │
     TP9 ●  │                      │  ● TP10   Temporal (behind ears)
     left   │                      │  right
            └──────────────────────┘
```

Each sensor picks up electrical activity from the brain region beneath it.
Different brain regions generate different dominant rhythms:

| Sensor | Position | Brain Region | Dominant EEG Band | Frequency Range | What It Reflects |
|--------|----------|-------------|-------------------|-----------------|-----------------|
| **TP9** | Left ear | Left temporal lobe | **Theta** | 4 -- 8 Hz | Memory retrieval, emotional processing, drowsiness, daydreaming |
| **AF7** | Left forehead | Left prefrontal cortex | **Alpha** | 8 -- 13 Hz | Relaxation, calm inhibition, eyes-closed rest, internal focus |
| **AF8** | Right forehead | Right prefrontal cortex | **Beta** | 13 -- 30 Hz | Active concentration, analytical thinking, problem solving |
| **TP10** | Right ear | Right temporal lobe | **Gamma** | 30 -- 44 Hz | High-level perception, cross-modal binding, peak awareness |

The Muse streams 256 samples/second per channel as raw microvolts via OSC
(`/muse/eeg` with 4 float values). It also sends contact quality via
`/muse/elements/horseshoe` (values 1-4 per sensor, where 1 = good, 4 = no contact).

---

## From Raw EEG to Band Power

The bridge collects EEG samples into a sliding window (default 1 second = 256
samples per channel) stored in a ring buffer. On each update tick (default 4 Hz),
it extracts the relevant band power for each sensor:

**Step 1: Welch Power Spectral Density**

```
raw EEG window (256 samples, 1 second)
           |
           v
    scipy.signal.welch(window, fs=256, nperseg=512)
           |
           v
    frequency axis (0 to 128 Hz) + power values (µV²/Hz)
```

**Step 2: Band Power Extraction**

For each sensor, the bridge extracts the power in its target band by averaging
the PSD values within that frequency range:

```
TP9:  theta_power = mean(PSD[4 Hz .. 8 Hz])
AF7:  alpha_power = mean(PSD[8 Hz .. 13 Hz])
AF8:  beta_power  = mean(PSD[13 Hz .. 30 Hz])
TP10: gamma_power = mean(PSD[30 Hz .. 44 Hz])
```

**Step 3: Adaptive Normalization**

Raw band power values vary enormously between people, sessions, and even minutes
within a session. The bridge uses adaptive percentile normalization over a
rolling window of 50 past measurements:

```
p10 = 10th percentile of last 50 power values for this channel
p90 = 90th percentile of last 50 power values for this channel

normalized = (current_power - p10) / (p90 - p10)
normalized = clamp(normalized, 0.0, 1.0)
```

This means 0.0 represents "low activity for this person right now" and 1.0
represents "high activity for this person right now" — regardless of absolute
microvolt values. The system self-calibrates over the first ~12 seconds
(50 updates at 4 Hz).

---

## From Band Power to Phase Velocity

Each normalized band power value (0 to 1) is converted to a **phase rotation
velocity** in degrees per second:

```
target_velocity = normalized_power * depth
```

Where `depth` is the `--depth` CLI argument (default 30.0, meaning max 30 deg/s).

The velocity is smoothed with an exponential moving average to prevent jitter:

```
velocity += (target_velocity - velocity) * smoothing_alpha
```

With default `smoothing_alpha = 0.25`, the velocity adapts in roughly 4 update
cycles (~1 second). This prevents sudden EEG artifacts from causing visible
phase jumps.

### Velocity Examples (at depth=30)

| Brain State | Theta (TP9) | Alpha (AF7) | Beta (AF8) | Gamma (TP10) |
|-------------|-------------|-------------|------------|--------------|
| Deep relaxation (eyes closed) | ~5 deg/s | **~25 deg/s** | ~3 deg/s | ~2 deg/s |
| Focused concentration | ~3 deg/s | ~5 deg/s | **~25 deg/s** | ~8 deg/s |
| Meditation (deep) | ~8 deg/s | ~15 deg/s | ~2 deg/s | ~1 deg/s |
| Drowsy / daydreaming | **~22 deg/s** | ~12 deg/s | ~3 deg/s | ~2 deg/s |
| Alert peak awareness | ~3 deg/s | ~5 deg/s | ~15 deg/s | **~25 deg/s** |
| Resting neutral | ~8 deg/s | ~10 deg/s | ~10 deg/s | ~5 deg/s |

---

## From Phase Velocity to Phase Accumulation

The velocity is integrated over time to produce a continuously advancing phase:

```
phase_accumulator += velocity * dt
phase_accumulator %= 360                (wrap around)

effective_phase = base_phase + phase_accumulator
```

This is the key behavior: **phase accumulates**. Unlike gain modulation (which
snaps back when the brain state changes), phase keeps advancing. If theta was
strong for 10 seconds, H2's phase has moved forward by some amount and stays
there even after theta drops. The next time theta rises, it continues from where
it left off.

### Full Cycle Timing

At maximum velocity (depth=30 deg/s), a harmonic completes a full 360-degree
rotation in:

```
360 / 30 = 12 seconds
```

At half power: 24 seconds. At low power: it barely moves.

With `depth=45`: full cycle in 8 seconds at max power.
With `depth=15`: full cycle in 24 seconds at max power.

---

## Phase Interpolation: The Dual-Rate Loop

### The Problem

EEG analysis is computationally expensive (Welch PSD on 256-sample windows)
and the Muse 2 only provides 256 samples/second, so analysis runs at 4 Hz
(every 250 ms). If phase output also runs at 4 Hz, the cymatic pattern
updates in visible jumps — a 15 deg/s rotation sends 3.75-degree steps
every 250 ms, which looks "jumpy" on the cymatic mirror.

### The Solution: Split Analysis and Output

The bridge runs two clocks:

| Clock | Rate | What it does |
|-------|------|--------------|
| **Analysis** (slow) | 4 Hz | Reads EEG, computes band power, updates target velocity (EMA-smoothed) |
| **Output** (fast) | 30 Hz (default) | Advances phase accumulators by `velocity * dt` and sends OSC |

```
Analysis tick (4 Hz):                Output tick (30 Hz):
  EEG window → PSD →                  accumulator += velocity × (1/30)
  band power → normalize →            accumulator %= 360
  target velocity →                   phase = base + accumulator
  EMA smooth → velocity               OSC send → shaper
```

At 30 Hz, a 15 deg/s rotation sends 0.5-degree steps every 33 ms — below
the threshold of visual perception. The cymatic pattern moves fluidly.

### dt and Phase Resolution

The `dt` in the output tick is `1 / osc_rate`:

| osc_rate | dt | Step at 15 deg/s | Step at 30 deg/s |
|----------|------|-------------------|-------------------|
| 4 Hz | 250 ms | 3.75 degrees | 7.5 degrees |
| **30 Hz** | **33 ms** | **0.5 degrees** | **1.0 degrees** |
| 60 Hz | 17 ms | 0.25 degrees | 0.5 degrees |

The default 30 Hz is chosen as a balance between smoothness and OSC traffic.
Tune with `--osc-rate` if needed. The EEG analysis rate (`--update-rate`)
is independent.

### Implementation Detail

The output tick calls `advance_phases(dt)` which reads the **current
velocity** (as set by the last analysis tick) and accumulates. Between
analysis ticks, velocity stays constant, producing linear interpolation.
When the next analysis tick updates velocity (via EMA smoothing), the
change is gradual — there is no discontinuity.

---

## The Harmonic-to-Sensor Mapping

```
┌──────────────────────────────────────────────────────────────────────┐
│                                                                      │
│  H1 (fundamental, e.g. 65 Hz)                                       │
│  ⚓ ANCHORED — never rotates, serves as phase reference              │
│                                                                      │
│  H2 (2nd harmonic, e.g. 130 Hz)                                     │
│  ← TP9 theta power (4-8 Hz)                                         │
│  Drowsy, dreamy, emotional brain states rotate H2                    │
│                                                                      │
│  H3 (3rd harmonic, e.g. 195 Hz)                                     │
│  ← AF7 alpha power (8-13 Hz)                                        │
│  Relaxation, calm, eyes-closed states rotate H3                      │
│                                                                      │
│  H4 (4th harmonic, e.g. 260 Hz)                                     │
│  ← AF8 beta power (13-30 Hz)                                        │
│  Active focus, analysis, problem-solving rotates H4                  │
│                                                                      │
│  H5 (5th harmonic, e.g. 325 Hz)                                     │
│  ← TP10 gamma power (30-44 Hz)                                      │
│  Peak awareness, binding, "aha" moments rotate H5                    │
│                                                                      │
└──────────────────────────────────────────────────────────────────────┘
```

### Why H1 Is Anchored

H1 (the fundamental) serves as the **phase reference** for all other harmonics.
If H1 also rotated, there would be no stable reference point — all phase
relationships would be relative and the accumulation would be meaningless.
By anchoring H1, the cymatic pattern has a fixed "home" orientation that
the other harmonics drift away from and return to.

### Why This Mapping

The mapping follows a neurological logic:

- **Lower harmonics ← slower brain rhythms.** Theta (4-8 Hz) is the slowest
  active rhythm and maps to H2, the lowest overtone. Theta states (drowsiness,
  memory, emotion) create slow, gentle pattern evolution.

- **Higher harmonics ← faster brain rhythms.** Gamma (30-44 Hz) is the fastest
  rhythm and maps to H5, the highest overtone. Gamma states (peak awareness,
  perceptual binding) create rapid, complex pattern changes.

- **Frontal sensors ← middle harmonics.** AF7 (alpha) and AF8 (beta) represent
  the relaxation/focus axis and map to H3 and H4, the middle of the series.
  These are the most commonly active bands in awake humans.

---

## What Happens in the Audio Engine

When the muse_bridge sends `/shaper/harmonic/3/phase` with value `147.5`, the
harmonic_shaper's `VoiceParameterStore` receives it:

```python
# osc_receiver.py
def _on_phase(self, addr, value, *_):
    n = self._parse_n(addr)     # extracts 3 from "/shaper/harmonic/3/phase"
    self._store.set_phase(n, float(value))

# state.py
def set_phase(self, harmonic_n, phase_deg):
    self._voices[harmonic_n].phase = math.radians(phase_deg % 360)
    # phase_deg=147.5 → 2.575 radians
```

On the next audio callback (~5.8 ms later), the audio engine reads this phase:

```python
# audio_engine.py — runs in PortAudio's C-level thread at 44100 Hz
carrier_phases = 2π × freq × t + start_phase   # continuous carrier
sine = sin(carrier_phases + params.phase)       # <-- phase offset applied here
sine *= gain
```

The `params.phase` term shifts the entire sine wave in time. For H3 at 195 Hz
with phase offset 147.5 degrees (2.575 radians):

```
sample = sin(2π × 195 × t + carrier_accumulator + 2.575) × gain
```

This is equivalent to shifting H3 in time by:

```
time_shift = 147.5° / 360° × (1/195 Hz) = 2.10 ms
```

At audio frequencies, 2.10 ms is significant — it changes how the peaks and
troughs of H3 align with H1, H2, H4, and H5 in the speaker output.

---

## Cymatic Impact: Phase and Interference Patterns

### Why Phase Matters for Cymatics

In the cymatic tube setup, the sound from two speakers creates a standing wave
pattern. The balloon membrane vibrates according to the **composite waveform** —
the sum of all active harmonics. The mirror deflects the laser proportionally
to the membrane's displacement and velocity.

The shape of the composite waveform depends on both the **amplitudes** and the
**phase relationships** between harmonics. Two harmonics at the same amplitudes
but different phase offsets produce completely different waveform shapes:

```
H1 + H2 (phase 0°):      H1 + H2 (phase 180°):

    ╱╲                        ╱╲
   ╱  ╲╱╲                   ╱  ╲
  ╱      ╲                 ╱    ╲
 ╱        ╲               ╱      ╲╱╲
                                     ╲
Tall peaks, sharp          Broader peaks, more
                           symmetric valleys
```

With 5 harmonics, the phase space is enormous. Each harmonic has 360 degrees
of freedom, giving 360^4 = 16.8 billion possible combinations (H1 is anchored).
The Muse navigates this space in real time based on brain activity.

### What You'll See

**Relaxed state (alpha dominant, H3 rotating fastest):**
The cymatic pattern slowly morphs as H3 drifts relative to the other harmonics.
Since alpha is the dominant resting rhythm, this is the "breathing" of the
pattern — slow, gentle, continuous evolution. H2, H4, H5 barely move.

**Focused state (beta dominant, H4 rotating fastest):**
H4 rotates faster, creating a more active pattern evolution. Since H4 is the
4th harmonic (two octaves above H1), its phase changes create higher-frequency
spatial features in the cymatic figure.

**Drowsy/meditative state (theta dominant, H2 rotating fastest):**
H2 (one octave above H1) rotates. Since H2 has the simplest harmonic
relationship with H1 (2:1), its phase changes create the most dramatic
large-scale pattern shifts — the whole figure seems to slowly "breathe" or
"pulse."

**Alert/gamma state (H5 rotating fastest):**
H5 adds fine detail and complexity. Its fast rotation creates rapid shimmer
in the pattern edges while the large-scale structure (driven by H2/H3) stays
relatively stable.

**Mixed state (all bands active):**
All harmonics rotate simultaneously at different speeds. The pattern becomes
a living, multi-scale interference field — large structures shift slowly
(theta/alpha → H2/H3) while fine details evolve quickly (beta/gamma → H4/H5).

---

## Signal Safety and Quality

### Contact Quality Gating

Each sensor is independently gated by the Muse's horseshoe quality indicator:

```
horseshoe value:  1 = good contact  →  sensor active, phase rotates
                  2 = okay contact  →  sensor active
                  3 = poor contact  →  sensor active (noisy)
                  4 = no contact    →  sensor MUTED, phase velocity → 0
```

When a sensor has no contact, its harmonic's phase **holds at its current
position** rather than snapping back to base. This prevents jarring phase jumps
when the headband shifts.

### EMA Smoothing

Phase velocity is smoothed with an exponential moving average
(`alpha = 0.25` default). This means:

- A sudden EEG spike takes ~4 updates (~1 second) to reach full effect
- A sudden drop takes ~4 updates to decay
- Muscle artifacts (which appear as broadband spikes) are dampened before
  they reach the phase accumulator
- The cymatic pattern transitions smoothly rather than jittering

### Shutdown Restoration

When the bridge exits (Ctrl+C or signal), it sends the **base phases** back
to the shaper, restoring the original phase configuration. This prevents
the shaper from being left in a random phase state.

---

## Configuration Reference

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--param phase` | phase | Select phase rotation mode |
| `--depth 30` | 30.0 | Maximum rotation speed in degrees/second |
| `--update-rate 4` | 4.0 | EEG analysis rate in Hz (slow clock) |
| `--osc-rate 30` | 30.0 | Phase output rate in Hz (fast clock, for interpolation) |
| `--window 1.0` | 1.0 | EEG analysis window in seconds |
| `--smoothing 0.25` | 0.25 | EMA smoothing factor (0 = no smoothing, 1 = no filtering) |
| `--pulse 0.15` | 0.15 | Heartbeat gain pulse amplitude (0 = disabled) |
| `--shaper-ip` | 127.0.0.1 | Harmonic shaper IP address |
| `--shaper-port` | 9002 | Shaper OSC control port |
| `--shaper-api` | http://127.0.0.1:8080 | Shaper HTTP API for base value fetch |
| `--listen-port` | 5000 | Port for incoming Muse 2 OSC and /bridge/* control |

### config.json

```json
"muse_bridge": {
    "shaper_ip": "127.0.0.1",
    "shaper_port": 9002,
    "shaper_api": "http://127.0.0.1:8080",
    "default_depth": 0.20,
    "smoothing_alpha": 0.25,
    "update_rate_hz": 4,
    "osc_rate_hz": 30,
    "window_seconds": 1.0,
    "tilt_weights": [-0.8, -0.4, 0.0, 0.4, 0.8]
}
```

### OSC Messages Sent (to harmonic_shaper :9002)

| Address | Payload | Range | Rate |
|---------|---------|-------|------|
| `/shaper/harmonic/1/phase` | `[degrees]` | Always base phase (anchored) | 30 Hz |
| `/shaper/harmonic/2/phase` | `[degrees]` | 0 -- 360, wrapping | 30 Hz |
| `/shaper/harmonic/3/phase` | `[degrees]` | 0 -- 360, wrapping | 30 Hz |
| `/shaper/harmonic/4/phase` | `[degrees]` | 0 -- 360, wrapping | 30 Hz |
| `/shaper/harmonic/5/phase` | `[degrees]` | 0 -- 360, wrapping | 30 Hz |

### Depth Tuning Guide

| Depth | Full rotation at max power | Character |
|-------|---------------------------|-----------|
| 10 | 36 seconds | Very slow, barely perceptible drift |
| 15 | 24 seconds | Gentle, meditative pace |
| **30** | **12 seconds** | **Default — visible but not frantic** |
| 45 | 8 seconds | Active, clearly evolving |
| 60 | 6 seconds | Fast, dynamic pattern changes |
| 90 | 4 seconds | Rapid, possibly disorienting |

---

## Complete Data Flow

```
1. OSC /muse/eeg arrives (256 Hz, 4 floats)
         |
         v
2. Ring buffer stores sample per channel
   buffers["TP9"][write_pos] = value[0]
   buffers["AF7"][write_pos] = value[1]
   buffers["AF8"][write_pos] = value[2]
   buffers["TP10"][write_pos] = value[3]
         |
         v  [SLOW CLOCK: every 250ms at 4 Hz analysis rate]
3. Extract 1-second window per channel
         |
         v
4. Welch PSD → band power extraction
   TP9:  theta_power = mean(PSD[4-8 Hz])
   AF7:  alpha_power = mean(PSD[8-13 Hz])
   AF8:  beta_power  = mean(PSD[13-30 Hz])
   TP10: gamma_power = mean(PSD[30-44 Hz])
         |
         v
5. Adaptive normalization (percentile p10/p90 over 50 samples)
   normalized = clamp((power - p10) / (p90 - p10), 0, 1)
         |
         v
6. Target velocity = normalized × depth (deg/s)
         |
         v
7. EMA smoothing: velocity += (target - velocity) × 0.25
         |
         |  [FAST CLOCK: every 33ms at 30 Hz output rate]
         v
8. Phase accumulation: accumulator += velocity × (1/30)
   accumulator %= 360
         |
         v
9. Effective phase = base_phase + accumulator
         |
         v
10. OSC send: /shaper/harmonic/N/phase → harmonic_shaper :9002
          |
          v
11. VoiceParameterStore.set_phase(n, degrees)
    → stores as radians internally
          |
          v
12. Audio callback reads phase (next ~5.8ms block)
    sine = sin(2π × freq × t + carrier_phase + phase_offset) × gain
          |
          v
13. Stereo mix → speakers → tube → balloon → mirror → laser → wall
```

Steps 1-7 run on the **slow clock** (4 Hz). Steps 8-10 run on the
**fast clock** (30 Hz), reading the most recent velocity from step 7.
This dual-rate design produces smooth, jitter-free cymatic movement
while keeping EEG computation efficient.
