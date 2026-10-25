"""
Passive BLE sensor recorder — no motor commands are sent.

Logs every SENS packet to hardware/logs/rec_<timestamp>.csv and prints a
per-sensor summary (floor, peak, contrast, noise, peak timing, ultrasonic dropouts).

Usage:
    python3 -m hardware.record --seconds 25
"""

import argparse
import asyncio
import csv
import time
from pathlib import Path

import numpy as np
from bleak import BleakClient, BleakScanner

from hardware.protocol import SensReader

ADDRESS = "8C5D08A4-1949-58CD-BDAD-11B903C3673C"   # this car's module; CoreBluetooth UUIDs are per-Mac
CHAR    = "0000ffe1-0000-1000-8000-00805f9b34fb"
LOG_DIR = Path(__file__).parent / "logs"


async def record(seconds, address):
    dev = await BleakScanner.find_device_by_address(address, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    reader, rows, t0 = SensReader(), [], None

    def on_notify(_, data):
        nonlocal t0
        before = reader.get()
        reader.feed(data)
        g = reader.get()
        if g is not before and g is not None:
            t0 = t0 or g["t"]
            rows.append((g["t"] - t0, *g["ir_raw"], g["dist_mm"]))

    async with BleakClient(dev) as client:
        await client.start_notify(CHAR, on_notify)
        print(f"recording {seconds}s …", flush=True)
        await asyncio.sleep(seconds)
        await client.stop_notify(CHAR)
    return rows


def summarize(rows):
    a = np.array(rows, dtype=float)
    t, ir, dist = a[:, 0], a[:, 1:4], a[:, 4]
    print(f"\n{len(a)} packets over {t[-1]:.1f}s (~{len(a) / t[-1]:.0f} Hz)")
    still = t < 1.0
    for i, name in enumerate("LCR"):
        x = ir[:, i]
        floor, peak = np.percentile(x, 10), x.max()
        noise = x[still].std() if still.any() else float("nan")
        above = x > floor + 0.5 * (peak - floor)
        rises = np.flatnonzero(np.diff(above.astype(int)) == 1) + 1
        print(f"  IR {name}: floor {floor:5.1f}  peak {peak:5.0f}  contrast {peak - floor:5.1f}  "
              f"noise {noise:4.1f}  crossings at t={np.round(t[rises], 1).tolist()}")
    valid = (dist > 0) & (dist < 4000)
    print(f"  ultrasonic: {dist[valid].min():.0f}–{dist[valid].max():.0f} mm, "
          f"dropouts {np.count_nonzero(~valid)}/{len(dist)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=25)
    ap.add_argument("--address", default=ADDRESS)
    args = ap.parse_args()

    rows = asyncio.run(record(args.seconds, args.address))
    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"rec_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ir_raw_l", "ir_raw_c", "ir_raw_r", "dist_mm"])
        w.writerows(rows)
    print(f"saved {path}")
    summarize(rows)


if __name__ == "__main__":
    main()
