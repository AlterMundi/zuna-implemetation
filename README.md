# ZUNA EEG-to-Vibration Pipeline

Record, denoise, and play back Muse 2 EEG data to the
[BeaconMagnetActuator](https://github.com/Pablomonte/BeaconMagnetActuator)
harmonic surface.

> **Looking for live cymatic control?** The real-time Muse/Fitbit/Launchpad
> modulation system has moved to
> [cymatic-control](../cymatic-control).

## Architecture

```
Muse 2 ──OSC──→ osc_recorder.py ──.fif──→ zuna_processor.py ──.fif──→ osc_playback.py ──→ Actuator
                                              (GPU machine)
```

## Quick Start

### 1. Record from Muse 2

Stream from Mind Monitor to port 5000.

```bash
python osc_recorder.py --duration 60
```

### 2. Denoise with ZUNA (GPU)

Transfer to a GPU machine and run the ZUNA denoiser.

```bash
scp recordings/*.fif gpu-server:~/eeg/
ssh gpu-server "python zuna_processor.py --input ~/eeg/recording.fif --output ~/eeg/enhanced/ --gpu"
scp gpu-server:~/eeg/enhanced/*.fif enhanced/
```

### 3. Play Back to Actuator

```bash
# Spectral mode — dominant EEG freq * multiplier
python osc_playback.py --input enhanced/recording_eeg.fif --ip 192.168.4.176 --mode spectral

# Band power mode — dominant band drives velocity
python osc_playback.py --input enhanced/recording_eeg.fif --ip 192.168.4.176 --mode band_power

# Concentration mode — focus score on a single tine
python osc_playback.py --input enhanced/recording_eeg.fif --ip 192.168.4.176 --mode concentration

# Multi-tine mode — regions drive different tines
python osc_playback.py --input enhanced/recording_eeg.fif --ip 192.168.4.176 --mode multi_tine
```

## Playback Modes

| Mode | Description |
|------|-------------|
| `spectral` | Dominant EEG frequency x multiplier drives actuator frequency |
| `band_power` | Dominant band's power drives velocity on its matching tine |
| `concentration` | Composite focus score drives a single tine |
| `multi_tine` | Different brain regions drive different tines simultaneously |

## Scripts

| Script | Purpose |
|--------|---------|
| `osc_recorder.py` | Capture Muse 2 OSC stream to MNE `.fif` file |
| `zuna_processor.py` | ZUNA denoising and upsampling (GPU) |
| `osc_playback.py` | Play `.fif` file to actuator (HTTP + OSC, 4 modes) |
| `generate_mock_eeg.py` | Generate synthetic `.fif` test files |

### Also present (migrated to cymatic-control)

The following scripts are kept here for reference but are actively maintained
in the [cymatic-control](../cymatic-control) repo:

| Script | Purpose |
|--------|---------|
| `muse_bridge.py` | EEG + heartbeat + slider -> harmonic_shaper modulation |
| `hr_relay.py` | Heart rate -> OSC relay |
| `midi_relay.py` | MIDI CC -> OSC relay |
| `osc_bridge.py` | Live direct bridge: Muse 2 -> actuator |
| `eeg_harmonic_bridge.py` | Harmonic series mapper -> Surge XT + actuator |
| `simulate_eeg.py` | Mock brain activity generator |
| `simulate_tilt.py` | Mock alpha/beta tilt stages |

## Requirements

```bash
pip install -r requirements.txt
```

Core: `numpy`, `scipy`, `mne`, `python-osc`

## Related Projects

- [cymatic-control](../cymatic-control) -- Live cymatic modulation (EEG, heartbeat, MIDI)
- [NaturalHarmony](https://github.com/AlterMundi/NaturalHarmony) -- harmonic_shaper synthesizer
- [BeaconMagnetActuator](https://github.com/Pablomonte/BeaconMagnetActuator) -- ESP32 harmonic surface

## License

MIT
