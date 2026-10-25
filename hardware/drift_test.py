"""
Open-loop straight-drive drift test over BLE (raw PWM, firmware mode 1).

Start centred on a straight tape. Drives with right PWM = left × (1 + trim) until
the IR loses the tape, a timeout, or an obstacle. Reports which sensor saw the
tape last (drift direction) and the distance travelled before losing it.

Usage:
    python3 -m hardware.drift_test --pwm 40 --trim 0.0
"""

import argparse
import asyncio
import csv
import time

import numpy as np
from bleak import BleakClient, BleakScanner

from hardware.protocol import M_PER_S_PER_PWM, SensReader, pwm_packet
from hardware.record import ADDRESS, CHAR, LOG_DIR

IR_THRESH = 80
LOST_FRAMES = 3


async def main(pwm, trim, max_s, stop_mm):
    dev = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    reader, log = SensReader(), []

    def on_notify(_, data):
        before = reader.get()
        reader.feed(data)
        g = reader.get()
        if g is not before and g is not None:
            log.append((g["t"], *g["ir_raw"], g["dist_mm"]))

    async with BleakClient(dev) as client:
        await client.start_notify(CHAR, on_notify)
        await asyncio.sleep(0.5)
        if not log or max(log[-1][1:4]) < IR_THRESH:
            await client.stop_notify(CHAR)
            raise SystemExit(f"not on the tape at start (IR {log[-1][1:4] if log else None})")
        t_start, reason = time.monotonic(), "timeout"
        try:
            while time.monotonic() - t_start < max_s:
                recent = [r for r in log[-LOST_FRAMES:] if r[0] > t_start]
                if len(recent) == LOST_FRAMES and all(max(r[1:4]) < IR_THRESH for r in recent):
                    reason = "lost tape"
                    break
                d = log[-1][4]
                if 0 < d < stop_mm:
                    reason = "obstacle"
                    break
                await client.write_gatt_char(CHAR, pwm_packet(pwm, pwm * (1 + trim)), response=True)
                await asyncio.sleep(0.05)
        finally:
            for _ in range(5):
                await client.write_gatt_char(CHAR, pwm_packet(0, 0), response=True)
        elapsed = time.monotonic() - t_start
        await asyncio.sleep(0.3)
        await client.stop_notify(CHAR)

    LOG_DIR.mkdir(exist_ok=True)
    path = LOG_DIR / f"drift_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["t_s", "ir_raw_l", "ir_raw_c", "ir_raw_r", "dist_mm"])
        w.writerows((t - t_start, *rest) for t, *rest in log)

    path_seq = []
    for r in log:
        if r[0] > t_start and max(r[1:4]) >= IR_THRESH:
            s_ = "LCR"[int(np.argmax(r[1:4]))]
            if not path_seq or path_seq[-1] != s_:
                path_seq.append(s_)
    exit_side = path_seq[-1] if path_seq else None
    drift = {"R": "drifted LEFT (tape exited under its right side)",
             "L": "drifted RIGHT (tape exited under its left side)"}.get(exit_side, "no clear drift")
    print(f"pwm {pwm}, trim {trim:+.3f}: {reason} after {elapsed:.2f}s "
          f"≈ {elapsed * pwm * M_PER_S_PER_PWM * 100:.0f} cm; tape under {'→'.join(path_seq)} → {drift}")
    print(f"saved {path}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pwm", type=float, default=40)
    ap.add_argument("--trim", type=float, default=0.0, help="right PWM = left × (1 + trim)")
    ap.add_argument("--max-s", type=float, default=4.0)
    ap.add_argument("--stop-mm", type=float, default=400)
    args = ap.parse_args()
    asyncio.run(main(args.pwm, args.trim, args.max_s, args.stop_mm))
