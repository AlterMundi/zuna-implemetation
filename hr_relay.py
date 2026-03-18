"""
HR Relay — Forwards heart rate data to muse_bridge as OSC.

Three modes for getting heartbeat data:

  simulate   — Generates beats at a configurable BPM with optional drift.
               No hardware needed. Good for development and tuning the
               cymatic pulse.

  ble        — Connects to a Fitbit Charge 6 or any BLE device that
               broadcasts the standard Heart Rate Profile (UUID 0x180D).
               Real-time beat events with RR intervals.

  fitbit-api — Polls the Fitbit Web API for intraday heart rate at
               1-second resolution. Requires OAuth2 browser auth (one-time).
               BPM updates with cloud sync delay.

All modes send to the muse_bridge OSC server:
  /bridge/heartbeat     [bpm, rr_ms]   (simulate, ble)
  /bridge/heartbeat_bpm [bpm]          (fitbit-api)

Usage:
    python hr_relay.py --mode simulate --bpm 72
    python hr_relay.py --mode simulate --bpm 72 --variation 5
    python hr_relay.py --mode ble
    python hr_relay.py --mode ble --device "Charge 6"
    python hr_relay.py --mode fitbit-api --client-id YOUR_ID --client-secret YOUR_SECRET
"""

import argparse
import json
import math
import random
import signal
import sys
import time
from pathlib import Path

from pythonosc import udp_client


def load_config():
    config_path = Path(__file__).parent / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            return json.load(f)
    return {}


# ─────────────────────────────────────────────
# Simulate Mode
# ─────────────────────────────────────────────

def run_simulate(osc, args):
    """Generate synthetic heartbeats with optional BPM drift."""
    base_bpm = args.bpm
    variation = args.variation
    current_bpm = base_bpm
    beat_count = 0

    print(f"  Generating beats at ~{base_bpm} BPM (variation: +/-{variation})...\n")

    try:
        while True:
            current_bpm = base_bpm + random.uniform(-variation, variation)
            current_bpm = max(30.0, min(200.0, current_bpm))
            rr_ms = 60000.0 / current_bpm
            interval = rr_ms / 1000.0

            osc.send_message("/bridge/heartbeat", [current_bpm, rr_ms])
            beat_count += 1

            bar = "\u2665" if beat_count % 2 == 0 else "\u2661"
            print(f"\r  {bar} beat #{beat_count}  {current_bpm:.1f} BPM  "
                  f"(RR: {rr_ms:.0f}ms)", end="", flush=True)

            time.sleep(interval)
    except KeyboardInterrupt:
        pass

    print(f"\n\n  Stopped after {beat_count} beats.\n")


# ─────────────────────────────────────────────
# BLE Mode
# ─────────────────────────────────────────────

HR_SERVICE_UUID = "0000180d-0000-1000-8000-00805f9b34fb"
HR_CHAR_UUID = "00002a37-0000-1000-8000-00805f9b34fb"


def parse_hr_measurement(data):
    """Parse a Heart Rate Measurement characteristic value (BLE spec).

    Returns (bpm, [rr_interval_ms, ...]).
    """
    flags = data[0]
    hr_16bit = flags & 0x01
    rr_present = (flags >> 4) & 0x01

    offset = 1
    if hr_16bit:
        bpm = int.from_bytes(data[offset:offset + 2], byteorder="little")
        offset += 2
    else:
        bpm = data[offset]
        offset += 1

    # Skip energy expended if present
    if (flags >> 3) & 0x01:
        offset += 2

    rr_intervals = []
    if rr_present:
        while offset + 1 < len(data):
            rr_raw = int.from_bytes(data[offset:offset + 2], byteorder="little")
            rr_intervals.append(rr_raw * 1000.0 / 1024.0)
            offset += 2

    return bpm, rr_intervals


async def run_ble_async(osc, args):
    """Connect to BLE HR sensor and stream beats."""
    import bleak

    device_filter = args.device

    print("  Scanning for BLE Heart Rate sensors...")
    scanner = bleak.BleakScanner()
    devices = await scanner.discover(timeout=args.scan_timeout)

    hr_devices = []
    for d in devices:
        name = d.name or ""
        if device_filter and device_filter.lower() not in name.lower():
            continue
        hr_devices.append(d)

    if not hr_devices:
        # If no filter matched, try all devices with HR service
        if device_filter:
            print(f"  No devices matching '{device_filter}' found.")
        print("  Checking all nearby devices for HR service...")
        hr_devices = devices

    target = None
    for d in hr_devices:
        try:
            async with bleak.BleakClient(d) as client:
                services = client.services
                for s in services:
                    if s.uuid.lower() == HR_SERVICE_UUID:
                        target = d
                        break
                if target:
                    break
        except Exception:
            continue

    if not target:
        print("  ERROR: No BLE device with Heart Rate Service found.")
        print("  Make sure your device is in exercise mode and nearby.")
        return

    name = target.name or target.address
    print(f"  Found HR sensor: {name} ({target.address})")
    print(f"  Connecting...\n")

    beat_count = 0

    def notification_handler(sender, data):
        nonlocal beat_count
        bpm, rr_intervals = parse_hr_measurement(bytearray(data))

        if rr_intervals:
            for rr_ms in rr_intervals:
                osc.send_message("/bridge/heartbeat", [float(bpm), rr_ms])
                beat_count += 1
                bar = "\u2665" if beat_count % 2 == 0 else "\u2661"
                print(f"\r  {bar} beat #{beat_count}  {bpm} BPM  "
                      f"(RR: {rr_ms:.0f}ms)", end="", flush=True)
        else:
            osc.send_message("/bridge/heartbeat", [float(bpm), 60000.0 / max(bpm, 1)])
            beat_count += 1
            bar = "\u2665" if beat_count % 2 == 0 else "\u2661"
            print(f"\r  {bar} beat #{beat_count}  {bpm} BPM", end="", flush=True)

    stop_event = None
    try:
        import asyncio
        stop_event = asyncio.Event()

        async with bleak.BleakClient(target) as client:
            await client.start_notify(HR_CHAR_UUID, notification_handler)
            print(f"  Streaming heart rate from {name}...")
            await stop_event.wait()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        print(f"\n  BLE error: {e}")

    print(f"\n\n  Stopped after {beat_count} beats.\n")


def run_ble(osc, args):
    """Entry point for BLE mode (wraps async)."""
    try:
        import bleak  # noqa: F401
    except ImportError:
        print("  ERROR: 'bleak' is required for BLE mode.")
        print("  Install it with: pip install bleak")
        sys.exit(1)

    import asyncio

    loop = asyncio.new_event_loop()
    try:
        loop.run_until_complete(run_ble_async(osc, args))
    except KeyboardInterrupt:
        pass
    finally:
        loop.close()


# ─────────────────────────────────────────────
# Fitbit Web API Mode
# ─────────────────────────────────────────────

TOKEN_PATH = Path.home() / ".fitbit_token.json"
FITBIT_AUTH_URL = "https://www.fitbit.com/oauth2/authorize"
FITBIT_TOKEN_URL = "https://api.fitbit.com/oauth2/token"
FITBIT_HR_URL = "https://api.fitbit.com/1/user/-/activities/heart/date/today/1d/1sec.json"


def load_token():
    if TOKEN_PATH.exists():
        with open(TOKEN_PATH) as f:
            return json.load(f)
    return None


def save_token(token):
    with open(TOKEN_PATH, "w") as f:
        json.dump(token, f)
    TOKEN_PATH.chmod(0o600)


def get_fitbit_session(client_id, client_secret):
    """Create or restore an OAuth2 session for the Fitbit Web API."""
    try:
        from requests_oauthlib import OAuth2Session
    except ImportError:
        print("  ERROR: 'requests-oauthlib' is required for fitbit-api mode.")
        print("  Install it with: pip install requests-oauthlib")
        sys.exit(1)

    scope = ["heartrate"]
    token = load_token()

    if token:
        extra = {"client_id": client_id, "client_secret": client_secret}
        session = OAuth2Session(
            client_id, token=token,
            auto_refresh_url=FITBIT_TOKEN_URL,
            auto_refresh_kwargs=extra,
            token_updater=save_token,
        )
        print("  Restored saved token from ~/.fitbit_token.json")
        return session

    # First-time authorization
    session = OAuth2Session(client_id, scope=scope,
                            redirect_uri="http://localhost:8189/callback")
    auth_url, state = session.authorization_url(FITBIT_AUTH_URL)

    print(f"\n  Open this URL in your browser to authorize:\n")
    print(f"  {auth_url}\n")
    print("  After authorizing, paste the full redirect URL below:")
    redirect_response = input("  > ").strip()

    token = session.fetch_token(
        FITBIT_TOKEN_URL,
        authorization_response=redirect_response,
        client_secret=client_secret,
    )
    save_token(token)
    print("  Token saved to ~/.fitbit_token.json")
    return session


def run_fitbit_api(osc, args):
    """Poll Fitbit Web API for intraday HR and send BPM updates."""
    if not args.client_id or not args.client_secret:
        print("  ERROR: --client-id and --client-secret are required for fitbit-api mode.")
        print("  Register an app at https://dev.fitbit.com/apps/new")
        sys.exit(1)

    session = get_fitbit_session(args.client_id, args.client_secret)
    poll_interval = args.poll_interval
    last_bpm = 0
    polls = 0

    print(f"  Polling Fitbit HR every {poll_interval}s...\n")

    try:
        while True:
            try:
                resp = session.get(FITBIT_HR_URL)
                resp.raise_for_status()
                data = resp.json()

                dataset = (data.get("activities-heart-intraday", {})
                           .get("dataset", []))
                if dataset:
                    latest = dataset[-1]
                    bpm = latest.get("value", 0)
                    ts = latest.get("time", "??:??:??")
                    if bpm > 0:
                        osc.send_message("/bridge/heartbeat_bpm", [float(bpm)])
                        last_bpm = bpm
                        polls += 1
                        print(f"\r  \u2661 {bpm} BPM @ {ts}  (poll #{polls})",
                              end="", flush=True)
                else:
                    print(f"\r  Waiting for HR data...", end="", flush=True)

            except Exception as e:
                print(f"\r  API error: {e}", end="", flush=True)

            time.sleep(poll_interval)
    except KeyboardInterrupt:
        pass

    print(f"\n\n  Stopped after {polls} polls. Last BPM: {last_bpm}\n")


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    cfg = load_config()
    hr_cfg = cfg.get("hr_relay", {})
    rec_cfg = cfg.get("recorder", {})

    parser = argparse.ArgumentParser(
        description="HR Relay — Forward heartbeat data to muse_bridge as OSC"
    )
    parser.add_argument("--mode", choices=["simulate", "ble", "fitbit-api"],
                        default=hr_cfg.get("mode", "simulate"),
                        help="Data source mode (default: simulate)")
    parser.add_argument("--target-ip",
                        default=hr_cfg.get("target_ip", "127.0.0.1"),
                        help="Target IP for muse_bridge (default: 127.0.0.1)")
    parser.add_argument("--target-port", type=int,
                        default=hr_cfg.get("target_port", rec_cfg.get("osc_port", 5000)),
                        help="Target port for muse_bridge (default: 5000)")

    # Simulate mode args
    parser.add_argument("--bpm", type=float,
                        default=hr_cfg.get("bpm", 72.0),
                        help="Base BPM for simulate mode (default: 72)")
    parser.add_argument("--variation", type=float,
                        default=hr_cfg.get("variation", 3.0),
                        help="BPM random variation +/- for simulate (default: 3)")

    # BLE mode args
    parser.add_argument("--device", default=hr_cfg.get("device", None),
                        help="BLE device name substring filter (default: any HR sensor)")
    parser.add_argument("--scan-timeout", type=float,
                        default=hr_cfg.get("scan_timeout", 10.0),
                        help="BLE scan timeout in seconds (default: 10)")

    # Fitbit API mode args
    parser.add_argument("--client-id", default=hr_cfg.get("client_id", None),
                        help="Fitbit OAuth2 client ID")
    parser.add_argument("--client-secret", default=hr_cfg.get("client_secret", None),
                        help="Fitbit OAuth2 client secret")
    parser.add_argument("--poll-interval", type=float,
                        default=hr_cfg.get("poll_interval", 15.0),
                        help="Fitbit API poll interval in seconds (default: 15)")

    args = parser.parse_args()

    osc = udp_client.SimpleUDPClient(args.target_ip, args.target_port)

    mode_labels = {
        "simulate": "SIMULATED BEATS",
        "ble": "BLE HEART RATE",
        "fitbit-api": "FITBIT WEB API",
    }

    print(f"\n{'='*55}")
    print(f"  HR Relay [{mode_labels[args.mode]}]")
    print(f"{'='*55}")
    print(f"  OSC out:  /bridge/heartbeat @ {args.target_ip}:{args.target_port}")
    if args.mode == "simulate":
        print(f"  BPM:      {args.bpm} +/- {args.variation}")
    elif args.mode == "ble":
        dev = args.device or "(any)"
        print(f"  Device:   {dev}")
    elif args.mode == "fitbit-api":
        print(f"  Poll:     every {args.poll_interval}s")
        print(f"  OSC out:  /bridge/heartbeat_bpm")
    print(f"{'='*55}\n")

    if args.mode == "simulate":
        run_simulate(osc, args)
    elif args.mode == "ble":
        run_ble(osc, args)
    elif args.mode == "fitbit-api":
        run_fitbit_api(osc, args)


if __name__ == "__main__":
    main()
