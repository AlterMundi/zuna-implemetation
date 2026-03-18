"""
MIDI Relay — Forwards a MIDI CC (e.g. Launchpad slider) as OSC to muse_bridge.

Reads a single CC number from a MIDI controller and sends it as
/bridge/gain_depth to the muse_bridge OSC server. This lets the
Launchpad Mini's mod wheel control EEG gain modulation depth in real-time.

Usage:
    python midi_relay.py                          # auto-detect port, CC1, target localhost:5000
    python midi_relay.py --cc 1 --target-port 5000
    python midi_relay.py --port "Launchpad Mini" --cc 74
    python midi_relay.py --list                   # list available MIDI ports
"""

import argparse
import sys
import json
from pathlib import Path

import mido
from pythonosc import udp_client


def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


def find_port(pattern=None):
    """Find a MIDI input port, optionally matching a name pattern."""
    available = mido.get_input_names()
    if not available:
        return None
    if pattern:
        for name in available:
            if pattern.lower() in name.lower():
                return name
    return available[0]


def main():
    cfg = load_config()
    relay_cfg = cfg.get("midi_relay", {})

    parser = argparse.ArgumentParser(
        description="MIDI Relay — Forward CC slider to muse_bridge as OSC"
    )
    parser.add_argument("--list", action="store_true",
                        help="List available MIDI ports and exit")
    parser.add_argument("--port", default=relay_cfg.get("midi_port", None),
                        help="MIDI port name or substring (default: auto-detect)")
    parser.add_argument("--cc", type=int,
                        default=relay_cfg.get("cc_number", 1),
                        help="MIDI CC number to relay (default: 1 = mod wheel)")
    parser.add_argument("--target-ip", default=relay_cfg.get("target_ip", "127.0.0.1"),
                        help="Target IP for muse_bridge OSC (default: 127.0.0.1)")
    parser.add_argument("--target-port", type=int,
                        default=relay_cfg.get("target_port", 5000),
                        help="Target port for muse_bridge OSC (default: 5000)")
    parser.add_argument("--osc-address",
                        default=relay_cfg.get("osc_address", "/bridge/gain_depth"),
                        help="OSC address to send (default: /bridge/gain_depth)")
    args = parser.parse_args()

    if args.list:
        ports = mido.get_input_names()
        if not ports:
            print("  No MIDI input ports found.")
        else:
            print("  Available MIDI input ports:")
            for p in ports:
                print(f"    - {p}")
        return

    port_name = find_port(args.port)
    if not port_name:
        print("  ERROR: No MIDI input ports available.")
        print("  Connect a MIDI controller and try again.")
        sys.exit(1)

    osc = udp_client.SimpleUDPClient(args.target_ip, args.target_port)

    print(f"\n{'='*55}")
    print(f"  MIDI Relay")
    print(f"{'='*55}")
    print(f"  MIDI in:  {port_name}")
    print(f"  CC:       {args.cc}")
    print(f"  OSC out:  {args.osc_address} @ {args.target_ip}:{args.target_port}")
    print(f"{'='*55}")
    print(f"  Move the slider... (Ctrl+C to stop)\n")

    last_value = -1

    try:
        with mido.open_input(port_name) as midi_in:
            for msg in midi_in:
                if msg.type == "control_change" and msg.control == args.cc:
                    if msg.value != last_value:
                        last_value = msg.value
                        osc.send_message(args.osc_address, [msg.value])
                        pct = int(msg.value / 127.0 * 100)
                        bar_w = 20
                        fill = int(msg.value / 127.0 * bar_w)
                        bar = "\u2588" * fill + "\u2591" * (bar_w - fill)
                        print(f"\r  CC{args.cc}: {msg.value:3d}/127  [{bar}]  {pct:3d}%", end="", flush=True)
    except KeyboardInterrupt:
        pass

    print(f"\n\n  Relay stopped.\n")


if __name__ == "__main__":
    main()
