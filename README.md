# 🧠→🔊 EEG-to-Vibration Pipeline

Map brain activity to sound and vibration using Muse 2, Surge XT, and the Harmonic Beacon actuator.

## Architecture

```
                         ┌──────────────────────────────────────────────────────┐
                         │               Use Case 1: Batch (ZUNA)              │
Muse 2 ──OSC──→ osc_recorder.py ──.fif──→ zuna_processor.py ──→ osc_playback.py ──→ Actuator
                         │               Use Case 2: Direct                    │
Muse 2 ──OSC──→ osc_bridge.py ─────────────────────────────────/fnote──→ Actuator
                         │               Use Case 3: Harmonic Series           │
Muse 2 ──OSC──→ eeg_harmonic_bridge.py ──→ Surge XT (/fnote + /param)
                         │                                     └──→ Actuator (HTTP /play)
                         │               Use Case 4: Gain Modulation           │
Muse 2 ──OSC──→ muse_bridge.py ──/shaper/harmonic/N/phase|gain──→ harmonic_shaper
                         └──────────────────────────────────────────────────────┘
```

## Quick Start

### Use Case 1: ZUNA-Enhanced Playback (batch)
Record → denoise with ZUNA on GPU → play back enhanced EEG.

```bash
# 1. Record from Muse 2 (Mind Monitor streaming to port 5000)
python osc_recorder.py --duration 60

# 2. Transfer to GPU machine, run ZUNA
scp recordings/*.fif gpu-server:~/eeg/
ssh gpu-server "python zuna_processor.py --input ~/eeg/recording.fif --output ~/eeg/enhanced/ --gpu"
scp gpu-server:~/eeg/enhanced/*.fif enhanced/

# 3. Play back to actuator
python osc_playback.py --input enhanced/recording_eeg.fif --ip 192.168.4.176 --mode spectral
```

### Use Case 2: Real-Time Direct Bridge
Live Muse 2 → actuator with no processing. Proof-of-concept.

```bash
python osc_bridge.py --actuator-ip 192.168.4.176 --mode spectral
```

### Use Case 3: Harmonic Series Controller ⭐
Each Muse 2 sensor drives a harmonic voice in the natural series.

```bash
# Surge XT only (audio)
python eeg_harmonic_bridge.py --surge-ip 127.0.0.1

# Actuator only (vibration)
python eeg_harmonic_bridge.py --actuator-ip 192.168.4.176

# Both simultaneously (audio + vibration)
python eeg_harmonic_bridge.py --surge-ip 127.0.0.1 --actuator-ip 192.168.4.176 --stereo

# Test without Muse 2 (simulated brain states)
python simulate_eeg.py &
python eeg_harmonic_bridge.py --surge-ip 127.0.0.1
```

**Sensor → Harmonic Mapping (f₁ = 64 Hz):**

| Sensor | Region | Harmonic | Freq | Gain driven by |
|--------|-----------|----------|------|----------------|
| TP9    | L temporal| H2       | 128  | theta power    |
| AF7    | L frontal | H3       | 192  | alpha power    |
| AF8    | R frontal | H4       | 256  | beta power     |
| TP10   | R temporal| H5       | 320  | gamma power    |
| Derived| Coherence | H1       | 64   | cross-corr.    |

**Modulation features:**
- Filter cutoff: alpha/beta ratio → Surge XT filter (relaxed=warm, focused=bright)
- Stereo asymmetry: `--stereo` scales L/R harmonic gain by hemisphere dominance
- Saturation detection: auto-mutes railing sensors (no skin contact)

### Use Case 4: EEG Modulation for Harmonic Shaper
Muse 2 brain activity modulates the
[NaturalHarmony](https://github.com/AlterMundi/NaturalHarmony) harmonic_shaper
via OSC. Two parameter modes:

**Phase rotation** (`--param phase`, default): Each sensor's band power controls
the rotation speed of its matched harmonic's phase. Stronger brain activity in
a band = faster phase drift for that harmonic. H1 stays anchored. The cymatic
pattern evolves continuously.

**Gain tilt** (`--param gain`): Alpha/beta ratio tilts the gain curve.
Relaxation boosts lower harmonics, focus boosts upper harmonics.

```bash
# Start harmonic_shaper (in the NaturalHarmony repo)
python -m harmonic_shaper.main

# Phase rotation (default) — max 30 deg/s
python muse_bridge.py --param phase --shaper-ip 127.0.0.1 --depth 30

# Gain tilt — ±20% modulation
python muse_bridge.py --param gain --shaper-ip 127.0.0.1 --depth 0.2

# Test without Muse 2
python simulate_eeg.py &
python muse_bridge.py --param phase --shaper-ip 127.0.0.1
```

**Phase rotation mapping:**

| Harmonic | Sensor | Band | Effect |
|----------|--------|------|--------|
| H1       | ---    | ---  | Anchored (no rotation) |
| H2       | TP9    | theta | Rotation speed ← theta power |
| H3       | AF7    | alpha | Rotation speed ← alpha power |
| H4       | AF8    | beta  | Rotation speed ← beta power |
| H5       | TP10   | gamma | Rotation speed ← gamma power |

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `osc_recorder.py` | Capture Muse 2 OSC → MNE `.fif` file |
| `zuna_processor.py` | ZUNA denoise/upsample (GPU machine) |
| `osc_playback.py` | Play `.fif` → actuator (HTTP + OSC, 4 modes) |
| `osc_bridge.py` | Live direct bridge: Muse 2 → actuator |
| `eeg_harmonic_bridge.py` | Harmonic series mapper → Surge XT + actuator |
| `muse_bridge.py` | EEG → harmonic_shaper (phase rotation or gain tilt) |
| `simulate_eeg.py` | Mock brain activity for testing (7 states) |
| `generate_mock_eeg.py` | Generate synthetic `.fif` test files |

## Requirements

```bash
pip install -r requirements.txt   # mne, python-osc, numpy, scipy
```

**Surge XT setup:** Enable OSC input in Settings, port 53280.

## Related Projects

- [NaturalHarmony](https://github.com/AlterMundi/NaturalHarmony) — MIDI-to-harmonic-series engine (same OSC protocol)
- [BeaconMagnetActuator](https://github.com/Pablomonte/BeaconMagnetActuator) — ESP32 harmonic surface firmware

## License

MIT
