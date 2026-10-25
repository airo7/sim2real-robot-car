"""Smoke test for the factory-firmware JSON bridge: IR, ultrasonic and a 1 s motor pulse.

Usage:
    python3 -m hardware.test_factory --serial /dev/cu.usbmodem00001   # USB @ 9600
    python3 -m hardware.test_factory --ip 192.168.4.1                 # WiFi, needs ESP32 TCP :100
    python3 -m hardware.test_factory --ip 192.168.4.1 --diagnose      # WiFi diagnostics only
"""

import argparse
import socket
import subprocess
import time
import urllib.error
import urllib.request

from hardware.pc_bridge_factory import FactoryUSBLink, FactoryTCPLink


def _local_ip_toward(target: str) -> str | None:
    """Which local IP the OS would use to reach target."""
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect((target, 9))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except OSError:
        return None


def diagnose_wifi(ip: str):
    print(f"=== WiFi diagnose {ip} ===\n")

    local = _local_ip_toward(ip)
    print(f"Your Mac IP toward {ip}: {local or 'UNKNOWN'}")
    if local and not local.startswith("192.168.4."):
        print("  ⚠ Not on 192.168.4.x — Mac may still be routing via Ethernet/VPN.")
        print("  Fix: turn off VPN (Tailscale/NordVPN), disconnect Ethernet,")
        print("       System Settings → Network → ELEGOO WiFi → Details →")
        print("       set IPv4 Configure to 'Using DHCP' and renew.")
    print()

    try:
        out = subprocess.check_output(["route", "-n", "get", ip], text=True, stderr=subprocess.STDOUT)
        for line in out.splitlines():
            if "interface:" in line or "gateway:" in line:
                print(f"  {line.strip()}")
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    print()

    print("HTTP camera test (same as browser):")
    tests = [
        (f"http://{ip}/", 4),
        (f"http://{ip}/capture", 4),
        (f"http://{ip}/Test", 4),
    ]
    for url, tmo in tests:
        try:
            with urllib.request.urlopen(url, timeout=tmo) as r:
                print(f"  {url} → HTTP {r.status}")
        except urllib.error.HTTPError as e:
            print(f"  {url} → HTTP {e.code}")
        except (urllib.error.URLError, TimeoutError, OSError) as e:
            print(f"  {url} → FAIL ({e})")
    print("  (stream endpoints omitted — MJPEG hangs; use browser for live view)")
    print()

    print("TCP ports:")
    for port in (80, 81, 100, 8080):
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.settimeout(3)
        try:
            s.connect((ip, port))
            print(f"  port {port}: OPEN")
        except OSError as e:
            err = f"{e.__class__.__name__}"
            if getattr(e, "errno", None) == 65:
                err += " (No route to host — routing/VPN, not the car)"
            print(f"  port {port}: closed ({err})")
        finally:
            s.close()
    print()
    print("How to read results:")
    print("  • Browser works + terminal 'No route to host' → Mac routing/VPN issue")
    print("  • HTTP 80 OPEN, port 100 closed → camera OK, no WiFi command bridge")
    print("  • port 100 OPEN → run: python3 -m hardware.test_factory --ip", ip)
    print("  • USB always works: --serial /dev/cu.usbmodem00001")
    print()
    print("Car checks: battery ON, switch CAM, ESP32 LED blinking.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ip", default=None)
    parser.add_argument("--serial", default=None)
    parser.add_argument("--diagnose", action="store_true")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print JSON TX/RX (WiFi) to verify Arduino responses")
    args = parser.parse_args()

    if args.diagnose:
        if not args.ip:
            args.ip = "192.168.4.1"
        diagnose_wifi(args.ip)
        return

    if args.serial:
        link = FactoryUSBLink(args.serial)
        if args.verbose:
            link._debug = True
    elif args.ip:
        link = FactoryTCPLink(args.ip, debug=args.verbose)
    else:
        parser.error("Provide --serial PORT (recommended) or --ip ADDR")

    print("Init session (N:110) …")
    init_ack = link.init_session()
    if args.verbose or (args.ip and init_ack is None):
        print(f"  init ack: {init_ack or 'TIMEOUT (Arduino not replying)'}")

    print("Read IR sensors …")
    ir = link.read_tracking()
    print(f"  IR on-line bits (L,C,R): {ir}")

    print("Read ultrasonic …")
    dist = link.read_distance_mm()
    print(f"  distance: {dist} mm")

    if args.ip and init_ack is None and dist == 9999:
        print()
        print("⚠ TCP connects but Arduino never answers over WiFi.")
        print("  Camera/ESP32 OK — ESP32↔Arduino serial path is likely dead.")
        print("  Check: power switch on CAM (not USB), battery ON, USB unplugged.")
        print("  Compare: python3 -m hardware.test_factory --serial /dev/cu.usbmodem00001")
        print()

    print("Pulse motors 1s (slow forward) …")
    link.set_motors(0.15, 0.15)
    time.sleep(1.0)
    link.set_motors(0.0, 0.0)
    print("Stopped.")

    link.stop()
    print("Factory bridge OK.")


if __name__ == "__main__":
    main()