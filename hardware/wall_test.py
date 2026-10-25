"""
Drive straight at a wall over BLE and measure real speed from the ultrasonic.

Per speed step: fitted ground speed, reaction delay (command → distance starts
falling), and coast after stop. Aborts below --stop-mm or if the ultrasonic goes
silent. Log saved to hardware/logs/wall_<timestamp>.csv.

Usage:
    python3 -m hardware.wall_test
"""

import argparse
import asyncio
import csv
import time

import numpy as np
from bleak import BleakClient, BleakScanner

from hardware.motor_sweep import cmd_packet
from hardware.protocol import SensReader, pwm_packet
from hardware.record import ADDRESS, CHAR, LOG_DIR

STEPS = [0.10, 0.15, 0.20]
DRIVE_S, PAUSE_S, RATE_HZ = 2.0, 1.0, 10
STALE_S = 0.3


async def main(stop_mm, steps, drive_s, pwm_mode=False, trim=0.0):
    def packet(v):
        return pwm_packet(v, v * (1 + trim)) if pwm_mode else cmd_packet(v, v)

    dev = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    reader, log = SensReader(), []
    last_valid = {"t": time.monotonic(), "d": None}
    cmd = {"v": 0.0}

    def on_notify(_, data):
        before = reader.get()
        reader.feed(data)
        g = reader.get()
        if g is not before and g is not None:
            d = g["dist_mm"]
            if 0 < d < 4000:
                last_valid.update(t=g["t"], d=d)
            log.append((g["t"], cmd["v"], d))

    async def drive(v, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            t = time.monotonic()
            if v and (time.monotonic() - last_valid["t"] > STALE_S):
                raise RuntimeError("ultrasonic silent — aborting")
            if v and last_valid["d"] is not None and last_valid["d"] < stop_mm:
                raise RuntimeError(f"within {stop_mm} mm of wall — stopping")
            cmd["v"] = v
            await client.write_gatt_char(CHAR, packet(v), response=True)
            await asyncio.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t)))

    async with BleakClient(dev) as client:
        await client.start_notify(CHAR, on_notify)
        try:
            await drive(0, 1.0)
            print(f"start distance {last_valid['d']} mm", flush=True)
            for v in steps:
                print(f"driving {v:g} {'pwm' if pwm_mode else 'm/s'}", flush=True)
                await drive(v, drive_s)
                await drive(0, PAUSE_S)
        except RuntimeError as e:
            print(e, flush=True)
        finally:
            cmd["v"] = 0.0
            for _ in range(5):
                await client.write_gatt_char(CHAR, cmd_packet(0, 0), response=True)
            await asyncio.sleep(0.5)
            await client.stop_notify(CHAR)

    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"wall_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    t0 = log[0][0]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "cmd_v", "dist_mm"])
        w.writerows((t - t0, v, d) for t, v, d in log)
    print(f"saved {path}")
    analyze(np.array([(t - t0, v, d) for t, v, d in log], dtype=float), pwm_mode)


def analyze(a, pwm_mode=False):
    t, v, d = a.T
    ok = (d > 0) & (d < 4000)
    print(f"\ndropouts {np.count_nonzero(~ok)}/{len(d)}")
    edges = np.flatnonzero(np.diff(v) != 0) + 1
    segs = np.split(np.arange(len(v)), edges)
    for seg in segs:
        cv = v[seg[0]]
        if cv == 0:
            continue
        m = seg[ok[seg]]
        if len(m) < 10:
            continue
        ts, ds = t[m], d[m]
        late = ts > ts[0] + 0.5
        if late.sum() < 6:
            continue
        slope, icept = np.polyfit(ts[late], ds[late], 1)
        speed = -slope / 1000
        pre = np.flatnonzero((t < t[seg[0]]) & (t > t[seg[0]] - 0.5) & ok)
        d_start = np.median(d[pre]) if len(pre) else ds[:3].mean()
        delay = (d_start - icept) / slope - t[seg[0]]
        post = np.flatnonzero((t > t[seg[-1]] + 0.5) & (t < t[seg[-1]] + 1.0) & ok)
        d_cut = slope * t[seg[-1]] + icept
        coast = d_cut - np.median(d[post]) if len(post) else float("nan")
        cmd = f"pwm {cv:g}" if pwm_mode else f"cmd {cv:.2f} m/s"
        ratio = "" if pwm_mode else f" ({speed / cv:.0%})"
        print(f"  {cmd} → real {speed:.3f} m/s{ratio}  "
              f"reaction ≈{delay * 1000:.0f} ms  coast ≈{coast:.0f} mm")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stop-mm", type=float, default=400)
    ap.add_argument("--steps", type=float, nargs="+", default=STEPS)
    ap.add_argument("--drive-s", type=float, default=DRIVE_S)
    ap.add_argument("--pwm", action="store_true", help="steps are raw PWM values (firmware mode 1)")
    ap.add_argument("--trim", type=float, default=0.0, help="right PWM = left × (1 + trim)")
    args = ap.parse_args()
    asyncio.run(main(args.stop_mm, args.steps, args.drive_s, args.pwm, args.trim))
