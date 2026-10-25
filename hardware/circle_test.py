"""
Turning-while-driving test over BLE (raw PWM): hold one fixed left/right PWM pair at a time.

The car drives in a circle; measure its diameter on the floor. `--sim` prints what the
MuJoCo model predicts for the same commands, for comparison.

Usage:
    python3 -m hardware.circle_test --pair 45 0 --seconds 6
    python3 -m hardware.circle_test --sim
"""

import argparse
import asyncio
import time

from bleak import BleakClient, BleakScanner

from hardware.protocol import pwm_packet
from hardware.record import ADDRESS, CHAR

PAIRS = [(45, 0), (45, 30), (45, 35), (60, 40)]


async def drive(pl, pr, seconds):
    dev = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    async with BleakClient(dev) as client:
        try:
            end = time.monotonic() + seconds
            while time.monotonic() < end:
                await client.write_gatt_char(CHAR, pwm_packet(pl, pr), response=True)
                await asyncio.sleep(0.1)
        finally:
            for _ in range(5):
                await client.write_gatt_char(CHAR, pwm_packet(0, 0), response=True)


def simulate():
    import numpy as np

    from mjsim.car_env import ElegooCarEnv, V8

    env = ElegooCarEnv(track="L", randomize=False, config=V8)
    for pl, pr in PAIRS:
        env.reset(seed=0)
        xs = []
        for k in range(160):
            env.step(np.array([pl, pr]) / env.cfg.pwm_scale)
            if k > 40:
                xs.append(env.data.qpos[:2].copy())
        xs = np.array(xs)
        A = np.c_[2 * xs, np.ones(len(xs))]
        cx, cy, c = np.linalg.lstsq(A, (xs ** 2).sum(1), rcond=None)[0]
        r = np.sqrt(c + cx ** 2 + cy ** 2)
        yaw_rate = env.data.qvel[5]
        print(f"PWM L{pl:3d} R{pr:3d}: sim circle diameter {2 * r * 100:5.0f} cm, yaw {yaw_rate:+.2f} rad/s")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pair", type=int, nargs=2, metavar=("PWM_L", "PWM_R"))
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--sim", action="store_true")
    args = ap.parse_args()
    if args.sim:
        simulate()
    elif args.pair:
        asyncio.run(drive(*args.pair, args.seconds))
    else:
        ap.error("give --pair PWM_L PWM_R or --sim")
