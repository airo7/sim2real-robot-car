"""List BLE devices, or connect to one and dump its services to find the UART characteristic.

macOS needs Bluetooth permission for the terminal app (Privacy & Security → Bluetooth).

Usage:
    python3 -m hardware.ble_scan                  # list all BLE devices
    python3 -m hardware.ble_scan --name HC        # connect to first match, dump services
    python3 -m hardware.ble_scan --address UUID   # inspect a specific device
"""

import argparse
import asyncio

from bleak import BleakScanner, BleakClient

HM10_SERVICE = "0000ffe0-0000-1000-8000-00805f9b34fb"


async def scan(timeout: float = 12.0):
    print(f"Scanning {timeout:.0f} s …")
    devs = await BleakScanner.discover(timeout=timeout, return_adv=True)
    if not devs:
        print("No BLE devices found. If this is a permission issue, enable Bluetooth "
              "for your terminal app and restart it.")
        return []
    rows = []
    for addr, (d, adv) in devs.items():
        name = d.name or adv.local_name or "(no name)"
        rows.append((adv.rssi, addr, name, adv.service_uuids))
    rows.sort(reverse=True)
    for rssi, addr, name, uuids in rows:
        print(f"  {addr}  RSSI={rssi:>4}  name={name!r}  services={uuids or '-'}")
    return rows


async def inspect(name_filter: str = None, address: str = None, timeout: float = 12.0):
    name = address or "?"
    if address is None:
        rows = await scan(timeout=timeout)
        match = next((r for r in rows if name_filter.lower() in (r[2] or "").lower()), None)
        if not match:
            print(f"\nNo device whose name contains {name_filter!r}. "
                  f"Re-run without --name and pick the address, or use --address "
                  f"to inspect an unnamed candidate directly.")
            return
        _, address, name, _ = match
    print(f"\nConnecting to {name!r} @ {address} …")
    async with BleakClient(address) as client:
        print("Connected. Services / characteristics:")
        writable = notifiable = None
        for svc in client.services:
            print(f"  service {svc.uuid}")
            for ch in svc.characteristics:
                props = ",".join(ch.properties)
                print(f"      char {ch.uuid}  [{props}]")
                if {"write", "write-without-response"} & set(ch.properties):
                    writable = writable or ch.uuid
                if "notify" in ch.properties:
                    notifiable = notifiable or ch.uuid
        print("\n── Use these in the BLE bridge ──")
        print(f"  address        = {address}")
        print(f"  write char     = {writable}")
        print(f"  notify char    = {notifiable}")
        if HM10_SERVICE.lower() in {s.uuid.lower() for s in client.services}:
            print("  ✓ HM-10-style UART service (FFE0) found — this looks like the HC-05 module.")
        else:
            print("  ✗ No FFE0 service — probably not the HC-05 module.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default=None,
                    help="substring of the device name to connect + inspect")
    ap.add_argument("--address", default=None,
                    help="connect + inspect a specific device address directly "
                         "(useful when the HC-05 shows up as '(no name)')")
    ap.add_argument("--timeout", type=float, default=12.0,
                    help="scan duration in seconds (raise this if the module's "
                         "name isn't showing up — it doesn't advertise its name "
                         "in every packet)")
    args = ap.parse_args()
    if args.address:
        asyncio.run(inspect(address=args.address, timeout=args.timeout))
    elif args.name:
        asyncio.run(inspect(name_filter=args.name, timeout=args.timeout))
    else:
        asyncio.run(scan(timeout=args.timeout))


if __name__ == "__main__":
    main()
