"""Calibration for the analog ITR20001 line sensors.

IRCalib is a fixed calibration saved to ir_calib.json; AutoIRCalib adapts while driving.

Usage:
    python3 -m hardware.ir_calib              # live monitor (raw + thresholded)
    python3 -m hardware.ir_calib --calibrate  # guided calibration, writes ir_calib.json
"""

import json
import time
from pathlib import Path

_CALIB_PATH = Path(__file__).with_name("ir_calib.json")


class IRCalib:
    """Per-sensor threshold and polarity, plus optional floor/line levels for analog activations."""

    def __init__(self, thresholds=(128, 128, 128), line_when_above=False,
                 floor=None, line=None):
        self.thresholds      = list(thresholds)
        self.line_when_above = bool(line_when_above)
        self.floor = list(floor) if floor is not None else None
        self.line  = list(line)  if line  is not None else None

    def apply(self, raw):
        """raw = (l, c, r) analog bytes → (0/1, 0/1, 0/1), 1 = on line."""
        out = []
        for v, thr in zip(raw, self.thresholds):
            above = v > thr
            on    = above if self.line_when_above else not above
            out.append(1 if on else 0)
        return tuple(out)

    def activations(self, raw):
        """raw → per-sensor activation in 0..1 (0 = floor, 1 = line); None without floor/line levels."""
        if self.floor is None or self.line is None:
            return None
        out = []
        for v, f, l in zip(raw, self.floor, self.line):
            span = l - f
            if abs(span) < 1e-6:
                out.append(0.0)
                continue
            a = (v - f) / span
            out.append(min(1.0, max(0.0, a)))
        return tuple(out)

    def read(self, raw):
        """raw → (ir_pattern, activations); same interface as AutoIRCalib.read."""
        return self.apply(raw), self.activations(raw)

    def save(self, path=_CALIB_PATH):
        Path(path).write_text(json.dumps({
            "thresholds":      self.thresholds,
            "line_when_above": self.line_when_above,
            "floor":           self.floor,
            "line":            self.line,
        }, indent=2))

    @classmethod
    def load(cls, path=_CALIB_PATH):
        """Saved calibration, or None if there is no file."""
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text())
        return cls(d["thresholds"], d["line_when_above"],
                   d.get("floor"), d.get("line"))


class AutoIRCalib:
    """Flags the line as a deviation from each sensor's own floor baseline, in either polarity.

    The baseline is median-seeded over WARMUP samples, then adapts only on floor-looking samples.
    """

    WARMUP = 40
    BASELINE_ALPHA = 0.01   # tau ~5 s, so a brief line crossing barely moves it
    NOISE_ALPHA    = 0.05
    K              = 4.0    # full activation at K noise-floors of deviation
    MIN_THRESHOLD  = 6.0    # sensor read noise

    def __init__(self, n=3):
        self._baseline  = [None] * n
        self._noise     = [self.MIN_THRESHOLD * 2] * n
        self._warmup_buf = [[] for _ in range(n)]
        self._count     = 0

    def read(self, raw):
        """raw → (ir_pattern, activations). Stateful: call exactly once per sample."""
        acts = []
        for i, v in enumerate(raw):
            v = float(v)
            if self._count < self.WARMUP:
                self._warmup_buf[i].append(v)
                self._baseline[i] = sorted(self._warmup_buf[i])[len(self._warmup_buf[i]) // 2]
                acts.append(0.0)
                continue

            dev = v - self._baseline[i]
            thr = max(self._noise[i] * self.K, self.MIN_THRESHOLD)
            act = min(1.0, abs(dev) / thr) if thr > 0 else 0.0
            acts.append(act)

            if act < 0.3:
                # Adapting on on-line samples would let a sustained detection inflate
                # its own threshold (or drag the baseline) and fade out within seconds.
                self._baseline[i] += self.BASELINE_ALPHA * dev
                self._noise[i] += self.NOISE_ALPHA * (abs(dev) - self._noise[i])

        self._count += 1
        act_t = tuple(acts)
        ir_t  = tuple(1 if a > 0.5 else 0 for a in acts)
        return ir_t, act_t


def _avg_raw(link, seconds=1.5):
    acc = [0, 0, 0]
    n   = 0
    t0  = time.monotonic()
    while time.monotonic() - t0 < seconds:
        s = link.latest_sensors()
        if s is not None and "ir_raw" in s:
            for i in range(3):
                acc[i] += s["ir_raw"][i]
            n += 1
        time.sleep(0.03)
    if n == 0:
        return None
    return [a / n for a in acc]


def _calibrate(link):
    print("\n── IR calibration ──────────────────────────────────────────")
    input("1) Place the car so NO sensor is over the line (all on floor),\n"
          "   then press Enter…")
    bg = _avg_raw(link)
    if bg is None:
        print("No sensor data — is the car powered and linked?")
        return
    print(f"   floor level     (L C R) = {bg[0]:.0f} {bg[1]:.0f} {bg[2]:.0f}")

    input("2) Slide the car so the CENTER sensor sits ON the black line,\n"
          "   then press Enter…")
    on = _avg_raw(link)
    if on is None:
        print("No sensor data — aborting.")
        return
    print(f"   on-line level   (L C R) = {on[0]:.0f} {on[1]:.0f} {on[2]:.0f}")

    delta = on[1] - bg[1]
    if abs(delta) < 8:
        print(f"\n⚠️  Center sensor barely changed ({delta:+.0f}). The sensor may "
              "not be over the line, the line may be too thin, or the height is "
              "wrong. Re-run and make sure the center LED passes over the tape.")
        return

    line_when_above = delta > 0
    # Only the centre sensor was placed on the line; L/R reuse its line-floor delta.
    thresholds = [int(round(b + delta / 2.0)) for b in bg]
    thresholds = [max(1, min(254, t)) for t in thresholds]
    floor      = [round(b, 1) for b in bg]
    line       = [round(bg[0] + delta, 1), round(on[1], 1), round(bg[2] + delta, 1)]

    calib = IRCalib(thresholds, line_when_above, floor=floor, line=line)
    calib.save()
    pol = "brighter" if line_when_above else "darker"
    print(f"\n✓ Saved {_CALIB_PATH.name}")
    print(f"   line reads {pol} than floor (delta={delta:+.0f})")
    print(f"   thresholds (L C R) = {thresholds}")
    print(f"   floor={floor}  line={line}  (enables analog line-position tracking)")
    print("   Verify below — center should read 1 on the line, 0 off it.\n")
    _monitor(link, calib)


def _monitor(link, calib=None):
    if calib is None:
        calib = IRCalib.load()
    if calib is None:
        print("(no ir_calib.json yet — showing raw + fallback >127 threshold)")
    print("Live IR — move the car over the line. Ctrl+C to stop.\n")
    try:
        while True:
            s = link.latest_sensors()
            if s is not None and "ir_raw" in s:
                raw = s["ir_raw"]
                ir  = calib.apply(raw) if calib else s["ir"]
                act = calib.activations(raw) if calib else None
                bar = "".join("█" if v else "·" for v in ir)
                a_s = (f"  act={act[0]:.2f} {act[1]:.2f} {act[2]:.2f}"
                       if act else "")
                print(f"\r  raw(L C R)={raw[0]:3d} {raw[1]:3d} {raw[2]:3d}   "
                      f"line={ir[0]} {ir[1]} {ir[2]}  [{bar}]{a_s}   "
                      f"dist={s['dist_mm']:>5}mm   ", end="", flush=True)
            time.sleep(0.12)
    except KeyboardInterrupt:
        print("\nstopped")


def main():
    import argparse
    ap = argparse.ArgumentParser(description="Calibrate / monitor the analog IR line sensors")
    ap.add_argument("--calibrate", action="store_true",
                    help="run guided calibration and write ir_calib.json")
    ap.add_argument("--serial", default=None, metavar="PORT",
                    help="use a serial link (USB) instead of BLE")
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--address", default=None, help="BLE device address (auto-discovers if omitted)")
    args = ap.parse_args()

    if args.serial:
        from hardware.pc_bridge_serial import SerialLink
        link = SerialLink(args.serial, args.baud)
    else:
        from hardware.pc_bridge_ble import BleLink, discover
        addr = args.address or discover()
        if addr is None:
            print("No BLE device named 'HC-05' found. Run: python3 -m hardware.ble_scan")
            return
        print(f"Connecting to {addr} …")
        link = BleLink(addr)

    time.sleep(0.5)
    try:
        if args.calibrate:
            _calibrate(link)
        else:
            _monitor(link)
    finally:
        link.stop()


if __name__ == "__main__":
    main()
