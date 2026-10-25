"""
Turn-rate test over BLE (raw PWM): measures real yaw rate from IR tape crossings.

Start centred on a long straight tape. During each step the centre sensor
crosses the tape twice per revolution, so yaw rate = 2π / (t[n+2] − t[n]).
Compared against no-slip differential-drive kinematics to get turn efficiency.

Usage:
    python3 -m hardware.turn_test
"""

import argparse
import asyncio
import csv
import time

import numpy as np
from bleak import BleakClient, BleakScanner

from hardware.ir_calib import IRCalib
from hardware.protocol import M_PER_S_PER_PWM, WHEELBASE, SensReader, pwm_packet
from hardware.record import ADDRESS, CHAR, LOG_DIR

STEPS = [(30, -30, 10), (45, -45, 10), (60, -60, 8), (80, -80, 8), (45, 0, 10), (60, 0, 8), (80, 0, 8)]
PAUSE_S, RATE_HZ = 1.0, 10


async def main():
    dev = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    reader, log, cmd = SensReader(IRCalib.load()), [], {"pl": 0, "pr": 0}

    def on_notify(_, data):
        before = reader.get()
        reader.feed(data)
        g = reader.get()
        if g is not before and g is not None:
            log.append((g["t"], cmd["pl"], cmd["pr"], *g["ir_raw"], g["ir"][1]))

    async def hold(pl, pr, seconds):
        cmd.update(pl=pl, pr=pr)
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            t = time.monotonic()
            await client.write_gatt_char(CHAR, pwm_packet(pl, pr), response=True)
            await asyncio.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t)))

    async with BleakClient(dev) as client:
        await client.start_notify(CHAR, on_notify)
        try:
            await hold(0, 0, 1.0)
            for pl, pr, secs in STEPS:
                print(f"pwm L{pl:+4d} R{pr:+4d}", flush=True)
                await hold(pl, pr, secs)
                await hold(0, 0, PAUSE_S)
        finally:
            cmd.update(pl=0, pr=0)
            for _ in range(5):
                await client.write_gatt_char(CHAR, pwm_packet(0, 0), response=True)
            await client.stop_notify(CHAR)

    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"turn_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    t0 = log[0][0]
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "pwm_l", "pwm_r", "ir_raw_l", "ir_raw_c", "ir_raw_r", "ir_c"])
        w.writerows((t - t0, *rest) for t, *rest in log)
    print(f"saved {path}")
    analyze(np.array([(t - t0, *rest) for t, *rest in log], dtype=float))


def sweep_times(t, raw, thresh=80, window=0.35):
    """Times where ≥2 sensors see the tape within `window` s — a real crossing, not an edge blip."""
    hits = sorted((t[i], k) for k in range(3) for i in np.flatnonzero(raw[:, k] > thresh))
    out = []
    for i, (ti, _) in enumerate(hits):
        if out and ti - out[-1] < 1.0:
            continue
        if len({k for tj, k in hits[i:] if tj - ti <= window}) >= 2:
            out.append(ti)
    return out


def analyze(a):
    t, pl, pr, raw = a[:, 0], a[:, 1], a[:, 2], a[:, 3:6]
    print(f"\n{'PWM L/R':>10} {'no-slip':>9} {'real':>9} {'efficiency':>10}  sweeps")
    for spl, spr, _ in STEPS:
        m = (pl == spl) & (pr == spr)
        if not m.any():
            continue
        sw = [x for x in sweep_times(t[m], raw[m]) if x > t[m][0] + 0.5]   # skip the start-on-tape one
        ideal = abs(spl - spr) * M_PER_S_PER_PWM / WHEELBASE
        if spl == -spr and len(sw) >= 2:
            real = np.pi / np.median(np.diff(sw))     # pivot: sweeps are half a turn apart
            print(f"{spl:+4.0f}/{spr:+4.0f} {ideal:7.2f}r/s {real:7.2f}r/s {real / ideal:9.0%}  {len(sw)}")
        else:
            why = "too slow (<1 half-turn)" if spl == -spr else "unknown geometry"
            print(f"{spl:+4.0f}/{spr:+4.0f} {ideal:7.2f}r/s {'—':>9} {'—':>10}  {len(sw)} ({why})")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    asyncio.run(main())
