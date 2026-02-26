# 🧠→🔊 ZUNA EEG-to-Vibration Pipeline

Record brain activity from Muse 2 → enhance with ZUNA AI → play back as vibrations on the harmonic surface.

```
Muse 2 ──OSC──→ osc_recorder.py ──.fif──→ [scp to GPU] ──→ zuna_processor.py
                                                                    │
ESP32 Beacon ←──OSC /fnote──← osc_playback.py ←──.fif──← [scp back]
```

## Quick Start

### 1. Install Dependencies (local machine)

```bash
pip install -r requirements.txt
```

### 2. Record a Session

Connect Muse 2 via Mind Monitor (OSC target = this machine's IP, port 5000):

```bash
python osc_recorder.py --duration 300 --output recordings/session_001.fif
```

### 3. Process with ZUNA (GPU machine)

Transfer and process:

```bash
# Transfer to GPU machine
scp recordings/session_001.fif gpu-machine:~/zuna-work/

# SSH in and process
ssh gpu-machine
pip install zuna  # first time only
python zuna_processor.py --input ~/zuna-work/session_001.fif \
                         --output ~/zuna-work/enhanced/ \
                         --bad-channels TP10 --gpu

# Transfer back
scp gpu-machine:~/zuna-work/enhanced/session_001.fif enhanced/
```

### 4. Play Back to Harmonic Surface

Ensure ESP32 Beacon is running `feature/musical-controls` branch:

```bash
python osc_playback.py --input enhanced/session_001.fif \
                       --ip 192.168.1.50 --mode spectral
```

## Playback Modes

| Mode | What it does | Best for |
|------|-------------|----------|
| `spectral` | Dominant EEG frequency × 32 → vibration frequency | Most organic feel |
| `band_power` | Strongest EEG band → matching harmonic tine | Feeling state changes |
| `concentration` | Focus score → vibration intensity on H5 | Simple feedback |
| `multi_tine` | Brain region → specific harmonic tine | Spatial mapping |

```bash
python osc_playback.py --input enhanced/session.fif --ip 192.168.1.50 --mode band_power
python osc_playback.py --input enhanced/session.fif --ip 192.168.1.50 --mode concentration
python osc_playback.py --input enhanced/session.fif --ip 192.168.1.50 --mode multi_tine
python osc_playback.py --input enhanced/session.fif --ip 192.168.1.50 --mode spectral --speed 2.0 --loop
```

## Configuration

Edit `config.json` to change defaults:
- **recorder**: OSC port, channels, sampling rate
- **zuna**: filter settings, diffusion steps
- **playback**: actuator IP/port, mode, harmonic multiplier, velocity range
- **muse2_positions**: 3D electrode coordinates for .fif montage

## ESP32 Actuator Protocol

The playback script sends to the `feature/musical-controls` branch of [BeaconMagnetActuator](../BeaconMagnetActuator/):

| OSC Address | Args | Description |
|---|---|---|
| `/fnote` | freq (Hz), vel (0-127), noteID | Note-on |
| `/fnote/rel` | freq, vel, noteID | Note-off |
| `/allnotesoff` | — | Stop all |

Port: `53280` (configurable in ESP32 config.json `osc_port`)

## Project Structure

```
Zuna-Implementation/
├── osc_recorder.py      # Stage 1: Muse 2 OSC → .fif
├── zuna_processor.py    # Stage 2: ZUNA denoise/upsample (GPU machine)
├── osc_playback.py      # Stage 3: .fif → OSC /fnote → actuator
├── config.json          # Pipeline configuration
├── requirements.txt     # Python dependencies
├── recordings/          # Raw .fif recordings
└── enhanced/            # ZUNA-processed .fif files
```

## Requirements

- **Local machine**: Python 3.8+, `numpy`, `scipy`, `mne`, `python-osc`
- **GPU machine**: Above + `zuna` (pip install zuna)
- **Hardware**: Muse 2 + Mind Monitor, ESP32 Beacon (feature/musical-controls branch)

## License

MIT
