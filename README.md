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

### Use Case 4: Cymatic Modulation for Harmonic Shaper

Muse 2 brain activity, heartbeat from a Fitbit, and a Launchpad MIDI
controller modulate the
[NaturalHarmony](https://github.com/AlterMundi/NaturalHarmony) harmonic_shaper
via OSC.  **Every input is independent and optional** — use one, two, or all
three together.

#### Modular Inputs

| Input | Script | What it controls | Optional? |
|-------|--------|------------------|-----------|
| **Muse 2 EEG** | `muse_bridge.py` | Phase rotation and/or gain tilt | Yes — if no EEG stream, bridge idles |
| **Fitbit / HR sensor** | `hr_relay.py` | Heartbeat gain pulse ("breathing" in cymatics) | Yes — if no relay running, no pulse |
| **Launchpad slider** | `midi_relay.py` | EEG gain modulation depth (0-100%) | Yes — gain depth stays at CLI default |
| **EEG simulator** | `simulate_eeg.py` | Synthetic Muse 2 data for testing | Replaces Muse 2 hardware |

#### Parameter Modes (`--param`)

**Phase rotation** (`--param phase`): Each sensor's band power controls the
rotation speed of its matched harmonic's phase.  H1 stays anchored.  Output is
interpolated at 30 Hz for smooth cymatic movement.

**Gain tilt** (`--param gain`): Alpha/beta ratio tilts the gain curve.
Relaxation boosts lower harmonics, focus boosts upper.

**Combined** (`--param both`, default): Phase rotation + gain tilt
simultaneously.  Gain depth is controlled by the Launchpad slider.

#### Heartbeat Pulse (`--pulse`)

When `hr_relay.py` sends heartbeat data, a short exponential gain envelope
fires on each beat — visible as a rhythmic "breathing" in the cymatic pattern.

```
final_gain = base * (1 + tilt * gain_depth) * (1 + envelope)
```

- `base` = Launchpad-set gain (always respected)
- `tilt * gain_depth` = EEG influence (slider-controlled, optional)
- `envelope` = heartbeat pulse (0 at rest, spikes on each beat, optional)

Works in **any** param mode.  Set `--pulse 0` to disable.

#### Session Configurations

```bash
# Always start harmonic_shaper first (in the NaturalHarmony repo)
python -m harmonic_shaper.main
```

**1. Muse-only: phase control**
```bash
python muse_bridge.py --param phase --depth 30
```

**2. Muse-only: gain tilt**
```bash
python muse_bridge.py --param gain --depth 0.20
```

**3. Muse phase + Launchpad gain (slider controls EEG gain depth)**
```bash
python midi_relay.py --target-port 5000 &
python muse_bridge.py --param both --depth 30
```

**4. Muse phase + Fitbit gain pulse (heartbeat drives cymatics, no EEG gain)**
```bash
python hr_relay.py --mode simulate --bpm 72 &
python muse_bridge.py --param phase --depth 30
```

**5. All three: Muse phase + Launchpad gain + Fitbit pulse**
```bash
python midi_relay.py --target-port 5000 &
python hr_relay.py --mode ble &
python muse_bridge.py --param both --depth 30
```

**6. Fitbit pulse only (no Muse, no Launchpad)**
```bash
python hr_relay.py --mode simulate --bpm 72 &
python muse_bridge.py --param phase --depth 0
```

**7. Test without any hardware**
```bash
python simulate_eeg.py &
python hr_relay.py --mode simulate --bpm 72 &
python muse_bridge.py --param both --depth 30
```

#### HR Relay Modes (`hr_relay.py`)

| Mode | Hardware | Latency | Beat-accurate? |
|------|----------|---------|----------------|
| `--mode simulate` | None | Zero | Synthetic (for testing/tuning) |
| `--mode ble` | Fitbit Charge 6 / any BLE HR sensor | 50-200ms | Yes (RR intervals) |
| `--mode fitbit-api` | Any Fitbit (Web API) | 2-15 min | No (BPM synthesized locally) |

```bash
# Simulate heartbeats at 72 BPM with +/-3 drift
python hr_relay.py --mode simulate --bpm 72 --variation 3

# Real-time from Fitbit Charge 6 (start an exercise on the watch first)
python hr_relay.py --mode ble

# Fitbit Web API (one-time OAuth2 setup)
python hr_relay.py --mode fitbit-api --client-id YOUR_ID --client-secret YOUR_SECRET
```

#### Phase Rotation Mapping

| Harmonic | Sensor | Band | Effect |
|----------|--------|------|--------|
| H1       | ---    | ---  | Anchored (no rotation) |
| H2       | TP9    | theta | Rotation speed ← theta power |
| H3       | AF7    | alpha | Rotation speed ← alpha power |
| H4       | AF8    | beta  | Rotation speed ← beta power |
| H5       | TP10   | gamma | Rotation speed ← gamma power |

#### Data Flow

```
                    ┌─────────────────┐
Muse 2 ──OSC──────→│                 │──/shaper/harmonic/N/phase──→ harmonic_shaper
                    │   muse_bridge   │──/shaper/harmonic/N/gain───→ harmonic_shaper
Fitbit ──hr_relay──→│     .py         │
                    │                 │
Launchpad ──midi───→│                 │
  relay    .py      └─────────────────┘
```

## Scripts Reference

| Script | Purpose |
|--------|---------|
| `osc_recorder.py` | Capture Muse 2 OSC → MNE `.fif` file |
| `zuna_processor.py` | ZUNA denoise/upsample (GPU machine) |
| `osc_playback.py` | Play `.fif` → actuator (HTTP + OSC, 4 modes) |
| `osc_bridge.py` | Live direct bridge: Muse 2 → actuator |
| `eeg_harmonic_bridge.py` | Harmonic series mapper → Surge XT + actuator |
| `muse_bridge.py` | EEG + heartbeat + slider → harmonic_shaper modulation |
| `hr_relay.py` | Heart rate → OSC (simulate, BLE, or Fitbit Web API) |
| `midi_relay.py` | MIDI CC (Launchpad slider) → OSC for muse_bridge |
| `simulate_eeg.py` | Mock brain activity for testing (7 states) |
| `generate_mock_eeg.py` | Generate synthetic `.fif` test files |

## Requirements

```bash
pip install -r requirements.txt
```

Core: `numpy`, `scipy`, `mne`, `python-osc`, `mido`
Optional: `bleak` (BLE heart rate), `requests-oauthlib` (Fitbit Web API)

**Surge XT setup:** Enable OSC input in Settings, port 53280.

## Testing

```bash
python -m pytest tests/ -v
```

48 tests covering gain tilt, phase rotation, heartbeat envelope, OSC handlers,
and end-to-end integration with simulated data.  No hardware required.

## Related Projects

- [NaturalHarmony](https://github.com/AlterMundi/NaturalHarmony) — MIDI-to-harmonic-series engine (same OSC protocol)
- [BeaconMagnetActuator](https://github.com/Pablomonte/BeaconMagnetActuator) — ESP32 harmonic surface firmware

## License

MIT
