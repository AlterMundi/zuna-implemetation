"""Shared fixtures for muse_bridge tests."""

import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure project root is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from muse_bridge import MuseBridge, CHANNELS, SAMPLING_RATE


class FakeOSCClient:
    """Records OSC messages instead of sending UDP."""

    def __init__(self):
        self.messages = []

    def send_message(self, address, args):
        self.messages.append((address, list(args)))

    def clear(self):
        self.messages.clear()

    def get(self, address_prefix=None):
        if address_prefix is None:
            return list(self.messages)
        return [(a, v) for a, v in self.messages if a.startswith(address_prefix)]


def make_sine_buffer(freq_hz, amplitude=50.0, duration_s=1.0, sfreq=SAMPLING_RATE):
    """Generate a sine wave buffer suitable for EEG analysis."""
    t = np.arange(int(duration_s * sfreq)) / sfreq
    return amplitude * np.sin(2 * np.pi * freq_hz * t)


def fill_bridge_buffers(bridge, signals, sfreq=SAMPLING_RATE):
    """Fill all channel buffers with provided signals.

    signals: dict mapping channel name -> numpy array
    """
    for ch, data in signals.items():
        n = min(len(data), bridge.window_size)
        bridge.buffers[ch][:n] = data[:n]
    bridge.write_pos = bridge.window_size
    bridge.samples_received = bridge.window_size


@pytest.fixture
def fake_osc():
    return FakeOSCClient()


@pytest.fixture
def make_bridge(fake_osc):
    """Factory fixture: returns a function that creates a MuseBridge with sensible defaults."""

    def _make(param_mode="both", phase_depth=30.0, gain_depth=0.2,
              update_rate=4.0, osc_rate=30.0, **kwargs):
        bridge = MuseBridge(
            osc_out=fake_osc,
            shaper_api="http://127.0.0.1:9999",
            param_mode=param_mode,
            phase_depth=phase_depth,
            gain_depth=gain_depth,
            update_rate=update_rate,
            osc_rate=osc_rate,
            **kwargs,
        )
        bridge.running = True
        # Default: all sensors have good contact
        for ch in CHANNELS:
            bridge.contact_quality[ch] = 1.0
        return bridge

    return _make


@pytest.fixture
def bridge_relaxed(make_bridge, fake_osc):
    """A bridge pre-loaded with alpha-dominant EEG (relaxed state)."""
    bridge = make_bridge(param_mode="both", gain_depth=0.5)
    alpha_wave = make_sine_buffer(10.0, amplitude=80.0)
    beta_wave = make_sine_buffer(20.0, amplitude=10.0)
    fill_bridge_buffers(bridge, {
        "TP9": alpha_wave,
        "AF7": alpha_wave,
        "AF8": alpha_wave,
        "TP10": alpha_wave,
    })
    return bridge


@pytest.fixture
def bridge_focused(make_bridge, fake_osc):
    """A bridge pre-loaded with beta-dominant EEG (focused state)."""
    bridge = make_bridge(param_mode="both", gain_depth=0.5)
    beta_wave = make_sine_buffer(20.0, amplitude=80.0)
    alpha_wave = make_sine_buffer(10.0, amplitude=10.0)
    fill_bridge_buffers(bridge, {
        "TP9": beta_wave,
        "AF7": beta_wave,
        "AF8": beta_wave,
        "TP10": beta_wave,
    })
    return bridge
