# Heartbeat-Driven Cymatic Pulse

## Overview

A heart rate sensor (Fitbit or any BLE HR device) sends beat timing to
`muse_bridge.py`, which fires a short gain envelope on each heartbeat.
The result is a visible rhythmic "breathing" in the cymatic pattern —
the figure swells and decays in sync with the performer's heartbeat.

The heartbeat is **fully optional**. If no `hr_relay.py` is running or
`--pulse 0` is set, the system works exactly as before — only EEG and
the Launchpad control the output.

---

## The Envelope

On each heartbeat, an exponential decay envelope fires:

```
envelope(t) = pulse_amplitude × exp(-t / decay_tau)
```

Where `t` is time since the last beat. The envelope starts at
`pulse_amplitude` (default 0.15 = 15% gain boost) and decays toward zero.

### Auto-Scaling Decay

The decay constant scales with BPM so pulses don't overlap:

```
decay_tau = 0.3 × (60 / bpm)
```

| BPM | Beat interval | decay_tau | Envelope at 50% of interval |
|-----|---------------|-----------|----------------------------|
| 60 | 1000 ms | 300 ms | 0.028 (1.9% of peak) |
| 72 | 833 ms | 250 ms | 0.027 (1.8% of peak) |
| 90 | 667 ms | 200 ms | 0.026 (1.7% of peak) |
| 120 | 500 ms | 150 ms | 0.027 (1.8% of peak) |

At all heart rates, the pulse has decayed to ~2% by the time the next beat
arrives — there's no "stacking" of pulses. Higher BPM simply means faster,
shorter pulses.

### How It Multiplies Into Gains

The pulse overlays on the existing gain chain:

```
final_gain = base × (1 + tilt × gain_depth) × (1 + envelope)
```

- `base` — Launchpad-set gain (always respected)
- `tilt × gain_depth` — EEG influence (slider-controlled, optional)
- `envelope` — heartbeat pulse (0 at rest, spikes on each beat)

The multiplication is per-harmonic. All harmonics swell together on each
beat, preserving the current tonal balance.

---

## Two Trigger Modes

The bridge supports two triggering modes, selected automatically based on
which OSC message it receives:

### Beat-Triggered Mode (BLE / Simulate)

`hr_relay.py` sends `/bridge/heartbeat [bpm, rr_ms]` on **each detected
beat**. The bridge resets `beat_phase` to 0 immediately, firing the
envelope. This gives beat-accurate timing — the cymatic pulse is
phase-locked to the actual heartbeat.

```
BLE sensor → hr_relay → /bridge/heartbeat → beat_phase = 0 → envelope fires
                                                 ↑
                                          on each physical beat
```

### BPM-Synthesized Mode (Fitbit Web API)

`hr_relay.py` sends `/bridge/heartbeat_bpm [bpm]` periodically (every
~15 seconds). The bridge runs a local clock at `60/bpm` seconds and
auto-triggers the envelope on each tick. The rhythm matches the person's
rate but isn't phase-locked to actual beats.

```
Web API → hr_relay → /bridge/heartbeat_bpm → bridge internal clock
                           (every ~15s)          → fires at 60/bpm interval
```

This is the fallback for Fitbit models that don't support BLE HR broadcast.

---

## hr_relay.py: Three Modes

The relay script is the single entry point for all heart rate data sources.
It normalizes everything into the same OSC interface that `muse_bridge.py`
expects.

### Simulate Mode (`--mode simulate`)

**No hardware needed.** Generates synthetic heartbeats at a configurable BPM
with optional random variation. For development, testing, and tuning the
pulse parameters.

```bash
python hr_relay.py --mode simulate --bpm 72 --variation 3
```

Sends `/bridge/heartbeat [bpm, rr_ms]` on each synthetic beat. The
variation adds +/-N BPM random drift per beat, simulating natural heart
rate variability.

### BLE Mode (`--mode ble`)

**Fitbit Charge 6 or any standard BLE Heart Rate sensor** (chest straps,
smart watches that broadcast HR Profile UUID 0x180D).

```bash
python hr_relay.py --mode ble
python hr_relay.py --mode ble --device "Charge 6"
```

Uses `bleak` to scan for nearby BLE devices with the Heart Rate Service.
Subscribes to HR Measurement notifications (UUID 0x2A37). Extracts:

- Heart rate in BPM
- RR intervals (beat-to-beat timing in 1/1024 second resolution)

Sends `/bridge/heartbeat [bpm, rr_ms]` for each detected beat from the
RR interval data. This is the highest-fidelity path — ~50-200ms BLE
notification latency, beat-accurate.

**Fitbit Charge 6 setup:**
1. On the Charge 6, go to Settings > HR on Equipment > enable
2. Start an exercise (e.g. "Workout") on the watch
3. The Charge 6 now advertises as a standard BLE HR monitor
4. Run `hr_relay.py --mode ble` on the computer
5. The relay will find and connect to it automatically

**BLE Heart Rate Measurement Parsing:**

The BLE HR characteristic (0x2A37) encodes data in a compact binary format:

```
Byte 0: Flags
  bit 0: 0 = HR is uint8, 1 = HR is uint16
  bit 4: 0 = no RR intervals, 1 = RR intervals present

Byte 1 (or 1-2): Heart rate value
Remaining bytes: RR intervals (uint16, units of 1/1024 sec)
```

Most wrist sensors (including Charge 6) send 8-bit HR with RR intervals.
Chest straps may send 16-bit HR. The relay handles both formats.

### Fitbit Web API Mode (`--mode fitbit-api`)

**Any Fitbit model** — uses the Fitbit Web API to poll intraday heart rate
at 1-second resolution. Requires a Fitbit developer account and OAuth2 setup.

```bash
python hr_relay.py --mode fitbit-api \
    --client-id YOUR_CLIENT_ID \
    --client-secret YOUR_CLIENT_SECRET \
    --poll-interval 15
```

**One-time setup:**
1. Register an app at https://dev.fitbit.com/apps/new
   - OAuth 2.0 Application Type: "Personal"
   - Callback URL: `http://localhost:8189/callback`
   - Access type: Read-only
2. Copy the Client ID and Client Secret
3. Run the relay — it will print a URL to open in your browser
4. Authorize the app, paste the redirect URL back
5. Token is saved to `~/.fitbit_token.json` (auto-refreshes)

**Limitations:**
- Intraday data has 2-15 minute cloud sync delay
- No RR intervals, only BPM values
- The bridge synthesizes beats locally from BPM (rhythmically correct
  but not phase-locked to actual beats)

Sends `/bridge/heartbeat_bpm [bpm]` on each poll.

---

## Which Fitbit Do You Have?

| Model | BLE HR Broadcast? | Recommended Mode |
|-------|-------------------|------------------|
| **Charge 6** | Yes (exercise mode) | `--mode ble` (best) |
| Sense 2 | No | `--mode fitbit-api` |
| Versa 4 | No | `--mode fitbit-api` |
| Inspire 3 | No | `--mode fitbit-api` |
| Luxe / older | No | `--mode fitbit-api` |
| **Any BLE chest strap** | Yes (always on) | `--mode ble` |
| No device yet | N/A | `--mode simulate` |

---

## Visual Effect on Cymatics

### What You See

On each heartbeat, all harmonics swell in gain simultaneously:

```
  gain
  1.0 ┤
      │     ╱╲          ╱╲          ╱╲
  0.9 ┤    ╱  ╲        ╱  ╲        ╱  ╲
      │   ╱    ╲      ╱    ╲      ╱    ╲
  0.8 ┤──╱──────╲────╱──────╲────╱──────╲──── base gain
      │          ╲  ╱          ╲  ╱
  0.7 ┤           ╲╱            ╲╱
      │
      └──────────────────────────────────── time
           beat 1      beat 2      beat 3
```

On the cymatic mirror, this creates a rhythmic pulse in the overall
figure intensity. The pattern briefly becomes more defined (higher amplitude
= stronger standing wave = clearer interference pattern), then relaxes.

### Combined with EEG

The heartbeat pulse is multiplicative on top of any EEG gain tilt. If the
EEG is tilting gains toward upper harmonics (focused state), the heartbeat
pulse swells those already-boosted harmonics even further. The two effects
layer naturally:

- **EEG** controls the *balance* between harmonics (slow, brain-state-driven)
- **Heartbeat** creates a *rhythmic pulse* across all harmonics (fast, body-driven)

### Combined with Phase

When running alongside phase rotation, the heartbeat pulse adds a third
dimension to the cymatic evolution:

- **Phase** (Muse 2): The *shape* of the interference pattern slowly evolves
- **Gain tilt** (Muse 2 + slider): The *tonal balance* shifts with brain state
- **Gain pulse** (Fitbit): The *intensity* rhythmically breathes with the heart

---

## Configuration

### CLI Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--pulse 0.15` | 0.15 | Heartbeat pulse amplitude (0 = disabled) |

### OSC Messages (received by muse_bridge)

| Address | Payload | Source | Effect |
|---------|---------|--------|--------|
| `/bridge/heartbeat` | `[bpm, rr_ms]` | BLE / simulate | Fires envelope immediately |
| `/bridge/heartbeat_bpm` | `[bpm]` | Fitbit Web API | Updates local beat synthesizer |

### Tuning the Pulse

| pulse_amplitude | Effect |
|-----------------|--------|
| 0.0 | Disabled (no heartbeat effect) |
| 0.05 | Very subtle, barely visible |
| **0.15** | **Default — clear but not overwhelming** |
| 0.25 | Strong, prominent breathing |
| 0.40 | Very dramatic, may clip gains near 1.0 |

The decay shape cannot be tuned directly from CLI — it auto-scales with BPM.
If you need different envelope shapes, modify the `compute_heartbeat_envelope`
method in `muse_bridge.py`.

### Dependencies

| Package | Mode | Install |
|---------|------|---------|
| `bleak>=0.22.0` | BLE | `pip install bleak` |
| `requests-oauthlib>=1.3.0` | fitbit-api | `pip install requests-oauthlib` |
| (none extra) | simulate | Already in requirements |
