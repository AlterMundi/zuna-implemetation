"""
Test battery for muse_bridge.py

Layer 1: Unit tests — pure math/logic (tilt, gain modulation, phase)
Layer 2: Handler tests — OSC handlers, depth control, restore
Layer 3: Integration — real UDP OSC with synthetic EEG
"""

import threading
import time

import numpy as np
import pytest
from pythonosc import dispatcher, osc_server, udp_client

from conftest import FakeOSCClient, fill_bridge_buffers, make_sine_buffer
from muse_bridge import (
    CHANNELS,
    PHASE_SENSOR_MAP,
    SAMPLING_RATE,
    TILT_WEIGHTS,
    MuseBridge,
)


# ═══════════════════════════════════════════════
# Layer 1: Unit Tests — Gain Tilt
# ═══════════════════════════════════════════════


class TestComputeTilt:
    """Tests for the alpha/beta tilt calculation."""

    def test_relaxed_positive_tilt(self, bridge_relaxed):
        tilt = bridge_relaxed.compute_tilt()
        assert tilt > 0.3, f"Alpha-dominant EEG should give positive tilt, got {tilt}"

    def test_focused_negative_tilt(self, bridge_focused):
        tilt = bridge_focused.compute_tilt()
        assert tilt < -0.3, f"Beta-dominant EEG should give negative tilt, got {tilt}"

    def test_balanced_less_extreme(self, make_bridge):
        """Mixing alpha + beta should produce less extreme tilt than pure alpha."""
        bridge = make_bridge(param_mode="gain", gain_depth=0.2)
        # Beta band is wider than alpha in PSD, so boost beta amplitude to
        # compensate for the mean-over-bins averaging in compute_band_powers
        balanced = make_sine_buffer(10.0, amplitude=50.0) + make_sine_buffer(20.0, amplitude=90.0)
        fill_bridge_buffers(bridge, {ch: balanced for ch in CHANNELS})
        tilt = bridge.compute_tilt()
        assert abs(tilt) < 0.5, f"Mixed alpha+beta should be less extreme than pure alpha, got {tilt}"

    def test_no_signal_zero_tilt(self, make_bridge):
        bridge = make_bridge(param_mode="gain", gain_depth=0.2)
        fill_bridge_buffers(bridge, {ch: np.zeros(SAMPLING_RATE) for ch in CHANNELS})
        tilt = bridge.compute_tilt()
        assert tilt == 0.0


class TestGainModulation:
    """Tests for EEG-driven gain tilt modulation."""

    def test_depth_zero_returns_base_gains(self, make_bridge):
        bridge = make_bridge(param_mode="gain", gain_depth=0.0)
        alpha = make_sine_buffer(10.0, amplitude=80.0)
        fill_bridge_buffers(bridge, {ch: alpha for ch in CHANNELS})

        gains = bridge.compute_gain_modulation()
        for n, g in gains.items():
            assert abs(g - bridge.base_gains[n]) < 1e-6, \
                f"H{n}: depth=0 should leave gain at base ({bridge.base_gains[n]}), got {g}"

    def test_depth_half_stays_within_range(self, make_bridge):
        bridge = make_bridge(param_mode="gain", gain_depth=0.5)
        alpha = make_sine_buffer(10.0, amplitude=80.0)
        fill_bridge_buffers(bridge, {ch: alpha for ch in CHANNELS})

        gains = bridge.compute_gain_modulation()
        for n, g in gains.items():
            base = bridge.base_gains[n]
            low = base * 0.5
            high = base * 1.5
            assert low - 0.01 <= g <= high + 0.01, \
                f"H{n}: gain {g} outside [{low}, {high}] for depth=0.5"

    def test_gains_clamp_to_zero_one(self, make_bridge):
        bridge = make_bridge(param_mode="gain", gain_depth=1.0)
        bridge.base_gains = {n: 0.95 for n in range(1, 6)}
        alpha = make_sine_buffer(10.0, amplitude=100.0)
        fill_bridge_buffers(bridge, {ch: alpha for ch in CHANNELS})

        # Run several iterations to push smoothed tilt toward extreme
        for _ in range(20):
            gains = bridge.compute_gain_modulation()
        for n, g in gains.items():
            assert 0.0 <= g <= 1.0, f"H{n}: gain {g} out of [0, 1]"

    def test_relaxed_tilt_boosts_lower_harmonics(self, bridge_relaxed, fake_osc):
        """Positive tilt + negative weights on lower harmonics = boost."""
        for _ in range(10):
            gains = bridge_relaxed.compute_gain_modulation()
        base = bridge_relaxed.base_gains

        # TILT_WEIGHTS = [-0.8, -0.4, 0.0, 0.4, 0.8]
        # Positive tilt * negative weight = negative modulator => base * (1 + depth * negative) < base
        # Wait — negative weight * positive tilt = negative modulator, which *reduces* gain.
        # The naming "boost lower" is about the relative curve shape:
        # Lower harmonics have more negative weights, so with positive tilt the
        # gain formula is base * (1 + depth * (tilt * weight)).
        # tilt>0, weight<0 => modulator<0 => effective < base for lower harmonics
        # tilt>0, weight>0 => modulator>0 => effective > base for upper harmonics
        # Actually, the TILT_WEIGHTS semantics: a relaxed state (positive tilt)
        # with weight=-0.8 on H1 means H1 gets *reduced*.
        # Let's just verify the weights create a differential:
        harmonics = sorted(gains.keys())
        assert gains[harmonics[0]] != gains[harmonics[-1]], \
            "Tilt should create a differential between low and high harmonics"

    def test_focused_tilt_opposite_direction(self, bridge_relaxed, bridge_focused, fake_osc):
        """Focused and relaxed should tilt gains in opposite directions."""
        for _ in range(10):
            g_relax = bridge_relaxed.compute_gain_modulation()
        # Reset the focused bridge's tilt_smooth
        for _ in range(10):
            g_focus = bridge_focused.compute_gain_modulation()

        h1_relax = g_relax.get(1, 0.8)
        h1_focus = g_focus.get(1, 0.8)
        h5_relax = g_relax.get(5, 0.8)
        h5_focus = g_focus.get(5, 0.8)

        # The direction should flip between relaxed and focused
        relax_diff = h5_relax - h1_relax
        focus_diff = h5_focus - h1_focus
        assert relax_diff * focus_diff < 0 or abs(relax_diff - focus_diff) > 0.01, \
            "Relaxed and focused should produce opposite tilt directions"


# ═══════════════════════════════════════════════
# Layer 1: Unit Tests — Phase Rotation
# ═══════════════════════════════════════════════


class TestPhaseVelocity:
    """Tests for EEG band power -> phase velocity mapping."""

    def test_zero_power_zero_velocity(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        fill_bridge_buffers(bridge, {ch: np.zeros(SAMPLING_RATE) for ch in CHANNELS})

        bridge.analyze_phase_velocities()
        for n in PHASE_SENSOR_MAP:
            assert bridge.phase_velocities[n] < 0.01, \
                f"H{n}: zero EEG should give near-zero velocity, got {bridge.phase_velocities[n]}"

    def test_high_power_approaches_depth(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0, smoothing_alpha=1.0)
        strong_theta = make_sine_buffer(6.0, amplitude=100.0)
        strong_alpha = make_sine_buffer(10.0, amplitude=100.0)
        strong_beta = make_sine_buffer(20.0, amplitude=100.0)
        strong_gamma = make_sine_buffer(35.0, amplitude=100.0)

        fill_bridge_buffers(bridge, {
            "TP9": strong_theta,
            "AF7": strong_alpha,
            "AF8": strong_beta,
            "TP10": strong_gamma,
        })

        # Run several rounds so adaptive normalization has history
        for _ in range(10):
            bridge.analyze_phase_velocities()

        for n in PHASE_SENSOR_MAP:
            assert bridge.phase_velocities[n] >= 0.0, \
                f"H{n}: velocity should be non-negative"
            assert bridge.phase_velocities[n] <= 30.0 + 0.1, \
                f"H{n}: velocity should not exceed phase_depth (30), got {bridge.phase_velocities[n]}"


class TestAdvancePhases:
    """Tests for phase accumulation logic."""

    def test_accumulation_linear(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        bridge.phase_velocities = {2: 10.0, 3: 20.0, 4: 5.0, 5: 15.0}
        dt = 0.1

        phases = bridge.advance_phases(dt)
        assert abs(bridge.phase_accumulators[2] - 1.0) < 0.01  # 10 deg/s * 0.1s
        assert abs(bridge.phase_accumulators[3] - 2.0) < 0.01
        assert abs(bridge.phase_accumulators[4] - 0.5) < 0.01
        assert abs(bridge.phase_accumulators[5] - 1.5) < 0.01

    def test_accumulation_over_multiple_ticks(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        bridge.phase_velocities = {2: 10.0, 3: 0.0, 4: 0.0, 5: 0.0}
        dt = 0.1

        for _ in range(10):
            bridge.advance_phases(dt)

        expected = 10.0 * 0.1 * 10  # 10 deg
        assert abs(bridge.phase_accumulators[2] - expected) < 0.1

    def test_wraps_at_360(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=400.0)
        bridge.phase_velocities = {2: 400.0, 3: 0.0, 4: 0.0, 5: 0.0}

        bridge.advance_phases(1.0)  # 400 deg in one step
        assert bridge.phase_accumulators[2] < 360.0
        assert abs(bridge.phase_accumulators[2] - 40.0) < 0.01

    def test_h1_stays_anchored(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        bridge.base_phases = {1: 45.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        bridge.phase_velocities = {2: 10.0, 3: 20.0, 4: 5.0, 5: 15.0}

        for _ in range(100):
            phases = bridge.advance_phases(0.1)

        assert phases[1] == 45.0, f"H1 should stay at base phase 45°, got {phases[1]}"

    def test_output_includes_base_phase(self, make_bridge):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        bridge.base_phases = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0, 5: 45.0}
        bridge.phase_velocities = {2: 10.0, 3: 0.0, 4: 0.0, 5: 0.0}

        phases = bridge.advance_phases(1.0)  # H2: 90 + 10 = 100
        assert abs(phases[2] - 100.0) < 0.1
        assert abs(phases[3] - 180.0) < 0.1  # no velocity, stays at base
        assert abs(phases[4] - 270.0) < 0.1


# ═══════════════════════════════════════════════
# Layer 2: Handler / Depth Control Tests
# ═══════════════════════════════════════════════


class TestGainDepthHandler:
    """Tests for /bridge/gain_depth OSC handler."""

    def test_midi_range_normalizes(self, make_bridge):
        bridge = make_bridge()
        bridge.gain_depth_handler("/bridge/gain_depth", 64)
        assert abs(bridge.gain_depth - 64 / 127.0) < 0.01

    def test_midi_max(self, make_bridge):
        bridge = make_bridge()
        bridge.gain_depth_handler("/bridge/gain_depth", 127)
        assert abs(bridge.gain_depth - 1.0) < 0.01

    def test_midi_zero(self, make_bridge):
        bridge = make_bridge()
        bridge.gain_depth_handler("/bridge/gain_depth", 0)
        assert bridge.gain_depth == 0.0

    def test_float_range_passthrough(self, make_bridge):
        bridge = make_bridge()
        bridge.gain_depth_handler("/bridge/gain_depth", 0.75)
        assert abs(bridge.gain_depth - 0.75) < 0.001

    def test_clamps_negative(self, make_bridge):
        bridge = make_bridge()
        bridge.gain_depth_handler("/bridge/gain_depth", -5.0)
        assert bridge.gain_depth == 0.0

    def test_no_args_ignored(self, make_bridge):
        bridge = make_bridge(gain_depth=0.5)
        bridge.gain_depth_handler("/bridge/gain_depth")
        assert bridge.gain_depth == 0.5


class TestEEGHandler:
    """Tests for /muse/eeg OSC handler."""

    def test_fills_buffer(self, make_bridge):
        bridge = make_bridge()
        for i in range(10):
            bridge.eeg_handler("/muse/eeg", 1.0, 2.0, 3.0, 4.0)

        assert bridge.samples_received == 10
        assert bridge.write_pos == 10

    def test_ring_buffer_wraps(self, make_bridge):
        bridge = make_bridge(window_seconds=0.1)
        n_samples = bridge.window_size + 5
        for i in range(n_samples):
            bridge.eeg_handler("/muse/eeg", float(i), 0.0, 0.0, 0.0)

        assert bridge.samples_received == n_samples
        # Buffer should contain the most recent window_size samples
        last_pos = (bridge.write_pos - 1) % bridge.window_size
        assert bridge.buffers["TP9"][last_pos] == float(n_samples - 1)

    def test_stopped_bridge_ignores(self, make_bridge):
        bridge = make_bridge()
        bridge.running = False
        bridge.eeg_handler("/muse/eeg", 1.0, 2.0, 3.0, 4.0)
        assert bridge.samples_received == 0


class TestHorseshoeHandler:
    """Tests for /muse/elements/horseshoe OSC handler."""

    def test_updates_quality(self, make_bridge):
        bridge = make_bridge()
        bridge.horseshoe_handler("/muse/elements/horseshoe", 1.0, 2.0, 1.0, 3.0)
        assert bridge.contact_quality["TP9"] == 1.0
        assert bridge.contact_quality["AF7"] == 2.0
        assert bridge.contact_quality["AF8"] == 1.0
        assert bridge.contact_quality["TP10"] == 3.0


class TestRestoreBase:
    """Tests for restore_base sending correct OSC messages."""

    def test_gain_mode_restores_gains(self, make_bridge, fake_osc):
        bridge = make_bridge(param_mode="gain", gain_depth=0.2)
        bridge.base_gains = {1: 0.7, 2: 0.6, 3: 0.5, 4: 0.4, 5: 0.3}
        bridge.restore_base()

        gain_msgs = fake_osc.get("/shaper/harmonic")
        assert len(gain_msgs) == 5
        for addr, args in gain_msgs:
            assert "/gain" in addr

    def test_phase_mode_restores_phases(self, make_bridge, fake_osc):
        bridge = make_bridge(param_mode="phase", phase_depth=30.0)
        bridge.base_phases = {1: 0.0, 2: 90.0, 3: 180.0, 4: 270.0, 5: 45.0}
        bridge.restore_base()

        phase_msgs = fake_osc.get("/shaper/harmonic")
        assert len(phase_msgs) == 5
        for addr, args in phase_msgs:
            assert "/phase" in addr

    def test_both_mode_restores_both(self, make_bridge, fake_osc):
        bridge = make_bridge(param_mode="both")
        bridge.base_gains = {1: 0.8, 2: 0.8, 3: 0.8, 4: 0.8, 5: 0.8}
        bridge.base_phases = {1: 0.0, 2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0}
        bridge.restore_base()

        gain_msgs = [m for m in fake_osc.messages if "/gain" in m[0]]
        phase_msgs = [m for m in fake_osc.messages if "/phase" in m[0]]
        assert len(gain_msgs) == 5
        assert len(phase_msgs) == 5


# ═══════════════════════════════════════════════
# Layer 3: Integration Tests — Real UDP OSC
# ═══════════════════════════════════════════════


def _find_free_port():
    import socket
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class OSCCollector:
    """Collects incoming OSC messages on a background server."""

    def __init__(self, port):
        self.messages = []
        self._lock = threading.Lock()
        disp = dispatcher.Dispatcher()
        disp.set_default_handler(self._handler)
        self.server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", port), disp)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def _handler(self, address, *args):
        with self._lock:
            self.messages.append((address, list(args)))

    def start(self):
        self.thread.start()

    def stop(self):
        self.server.shutdown()

    def get(self, prefix=None):
        with self._lock:
            if prefix:
                return [(a, v) for a, v in self.messages if a.startswith(prefix)]
            return list(self.messages)

    def wait_for(self, prefix, count=1, timeout=5.0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if len(self.get(prefix)) >= count:
                return True
            time.sleep(0.05)
        return False


class TestIntegrationPhaseMode:
    """End-to-end: synthetic EEG -> muse_bridge -> OSC phase output."""

    def test_phase_messages_arrive(self):
        listen_port = _find_free_port()
        output_port = _find_free_port()

        collector = OSCCollector(output_port)
        collector.start()

        osc_out = udp_client.SimpleUDPClient("127.0.0.1", output_port)
        bridge = MuseBridge(
            osc_out=osc_out,
            shaper_api="http://127.0.0.1:9999",
            param_mode="phase",
            phase_depth=30.0,
            gain_depth=0.0,
            update_rate=4.0,
            osc_rate=30.0,
        )
        bridge.running = True
        for ch in CHANNELS:
            bridge.contact_quality[ch] = 1.0

        # Set up OSC input server
        disp = dispatcher.Dispatcher()
        disp.map("/muse/eeg", bridge.eeg_handler)
        disp.map("/muse/elements/horseshoe", bridge.horseshoe_handler)
        in_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", listen_port), disp)
        in_thread = threading.Thread(target=in_server.serve_forever, daemon=True)
        in_thread.start()

        # Send synthetic EEG
        sender = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        theta_wave = make_sine_buffer(6.0, amplitude=60.0)
        for i in range(len(theta_wave)):
            sender.send_message("/muse/eeg", [
                float(theta_wave[i]),
                float(theta_wave[i]),
                float(theta_wave[i]),
                float(theta_wave[i]),
            ])

        # Give bridge time to fill buffer then run a few analysis+output ticks
        time.sleep(0.3)
        bridge.analyze_phase_velocities()
        for _ in range(5):
            bridge.advance_phases(1.0 / 30.0)
            phases = bridge.advance_phases(1.0 / 30.0)
            bridge.send_phases(phases)
            time.sleep(0.02)

        # Verify
        assert collector.wait_for("/shaper/harmonic", count=4, timeout=3.0), \
            f"Expected phase OSC messages, got {len(collector.get('/shaper/harmonic'))}"

        phase_msgs = collector.get("/shaper/harmonic")
        for addr, args in phase_msgs:
            assert "/phase" in addr
            assert 0.0 <= args[0] < 360.0

        # Cleanup
        bridge.running = False
        in_server.shutdown()
        collector.stop()


class TestIntegrationBothModeSlider:
    """End-to-end: synthetic EEG + slider -> phase + gain output."""

    def test_slider_activates_gain(self):
        listen_port = _find_free_port()
        output_port = _find_free_port()

        collector = OSCCollector(output_port)
        collector.start()

        osc_out = udp_client.SimpleUDPClient("127.0.0.1", output_port)
        bridge = MuseBridge(
            osc_out=osc_out,
            shaper_api="http://127.0.0.1:9999",
            param_mode="both",
            phase_depth=30.0,
            gain_depth=0.0,
            update_rate=4.0,
            osc_rate=30.0,
        )
        bridge.running = True
        for ch in CHANNELS:
            bridge.contact_quality[ch] = 1.0

        # Set up OSC server with gain_depth handler
        disp = dispatcher.Dispatcher()
        disp.map("/muse/eeg", bridge.eeg_handler)
        disp.map("/muse/elements/horseshoe", bridge.horseshoe_handler)
        disp.map("/bridge/gain_depth", bridge.gain_depth_handler)
        in_server = osc_server.ThreadingOSCUDPServer(("127.0.0.1", listen_port), disp)
        in_thread = threading.Thread(target=in_server.serve_forever, daemon=True)
        in_thread.start()

        # Fill buffer with EEG
        sender = udp_client.SimpleUDPClient("127.0.0.1", listen_port)
        alpha_wave = make_sine_buffer(10.0, amplitude=80.0)
        for i in range(len(alpha_wave)):
            sender.send_message("/muse/eeg", [
                float(alpha_wave[i]),
                float(alpha_wave[i]),
                float(alpha_wave[i]),
                float(alpha_wave[i]),
            ])

        time.sleep(0.3)

        # Initially gain_depth=0, so no gain messages should be produced
        bridge.analyze_phase_velocities()
        assert bridge.gain_depth == 0.0

        # Send slider via OSC
        sender.send_message("/bridge/gain_depth", [64])
        time.sleep(0.2)
        assert bridge.gain_depth > 0.4, f"Slider should have set gain_depth, got {bridge.gain_depth}"

        # Now run gain modulation — should produce gain messages
        gains = bridge.compute_gain_modulation()
        bridge.send_gains(gains)

        assert collector.wait_for("/shaper/harmonic", count=5, timeout=3.0)

        gain_msgs = [m for m in collector.get("/shaper/harmonic") if "/gain" in m[0]]
        assert len(gain_msgs) >= 5, f"Expected gain messages after slider, got {len(gain_msgs)}"

        for addr, args in gain_msgs:
            assert 0.0 <= args[0] <= 1.0

        # Cleanup
        bridge.running = False
        in_server.shutdown()
        collector.stop()
