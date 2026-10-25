"""
Wheels-in-the-air motor sweep over BLE: numbered pulses of increasing speed,
left side first, then right. Count pulses to find where each side starts turning.

Usage:
    python3 -m hardware.motor_sweep
"""

import argparse
import asyncio
import time

import numpy as np
from bleak import BleakClient, BleakScanner

from hardware.protocol import CMD_HEADER, V_MAX
from hardware.record import ADDRESS, CHAR

SPEEDS = [0.02, 0.03, 0.04, 0.06, 0.08, 0.10, 0.125, 0.15, 0.20, 0.25]
ON_S, OFF_S, GAP_S, RATE_HZ = 1.5, 0.5, 3.0, 10


def cmd_packet(vl, vr):
    """Mode-0 (velocity) CMD packet, wheel speeds in m/s."""
    b = [to_byte(v) for v in (vl, vr)]
    return bytes([CMD_HEADER, b[0], b[1], 0, 0, b[0] ^ b[1]])


def to_byte(v):
    # round, not truncate: truncation turns 0.02 m/s into 0.0176, below the firmware's 0.02 stop band
    return int(np.clip(round((v / V_MAX + 1.0) / 2.0 * 255), 0, 255))


def pwm(v):
    """PWM the firmware's velToPWM applies to a mode-0 command."""
    v = (to_byte(v) / 255 * 2 - 1) * V_MAX
    return int(abs(v) / V_MAX * (230 - 55)) + 55 if abs(v) >= 0.02 else 0


async def hold(client, vl, vr, seconds, acks):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        t = time.monotonic()
        await client.write_gatt_char(CHAR, cmd_packet(vl, vr), response=True)
        acks.append(time.monotonic() - t)
        await asyncio.sleep(max(0.0, 1 / RATE_HZ - (time.monotonic() - t)))


async def main():
    dev = await BleakScanner.find_device_by_address(ADDRESS, timeout=20)
    if dev is None:
        raise SystemExit("car not found — is the battery switch on?")
    acks = []
    async with BleakClient(dev) as client:
        try:
            for side in ("left", "right"):
                for n, v in enumerate(SPEEDS, 1):
                    print(f"{side} pulse {n:2d}: v={v:.3f} m/s  pwm≈{pwm(v)}", flush=True)
                    await hold(client, v if side == "left" else 0, v if side == "right" else 0, ON_S, acks)
                    await hold(client, 0, 0, OFF_S, acks)
                await hold(client, 0, 0, GAP_S, acks)
        finally:
            for _ in range(5):
                await client.write_gatt_char(CHAR, cmd_packet(0, 0), response=True)
    a = np.array(acks) * 1000
    print(f"\nBLE write ack: median {np.median(a):.0f} ms, p95 {np.percentile(a, 95):.0f} ms, max {a.max():.0f} ms")


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    asyncio.run(main())
