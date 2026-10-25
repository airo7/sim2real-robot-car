"""Per-step CSV log of sensors and commands for hardware runs (hardware/logs/run_<timestamp>.csv)."""

import csv
import time
from pathlib import Path

_LOG_DIR = Path(__file__).with_name("logs")

_COLUMNS = [
    "step", "t_s", "sens_age_ms",
    "ir_raw_l", "ir_raw_c", "ir_raw_r",
    "ir_act_l", "ir_act_c", "ir_act_r",
    "ir_l", "ir_c", "ir_r",
    "dist_mm", "vl", "vr", "cte", "comp_ms",
]


class RunLogger:
    def __init__(self, path=None):
        _LOG_DIR.mkdir(exist_ok=True)
        if path is None:
            path = _LOG_DIR / f"run_{time.strftime('%Y%m%d_%H%M%S')}.csv"
        self.path = Path(path)
        self._f = open(self.path, "w", newline="")
        self._w = csv.writer(self._f)
        self._w.writerow(_COLUMNS)
        self._t0 = time.monotonic()

    def log(self, step, sensors, action, cte=None, comp_ms=0.0):
        now = time.monotonic()
        raw = sensors.get("ir_raw") if sensors else None
        act = sensors.get("ir_act") if sensors else None
        ir  = sensors.get("ir")     if sensors else None
        raw = raw if raw is not None else (None, None, None)
        act = act if act is not None else (None, None, None)
        ir  = ir  if ir  is not None else (None, None, None)
        dist   = sensors.get("dist_mm") if sensors else None
        age_ms = (now - sensors["t"]) * 1000 if sensors and "t" in sensors else None
        self._w.writerow([
            step, round(now - self._t0, 3),
            round(age_ms, 1) if age_ms is not None else "",
            *raw, *act, *ir,
            dist if dist is not None else "",
            round(float(action[0]), 3), round(float(action[1]), 3),
            round(cte, 4) if cte is not None else "",
            round(comp_ms, 1),
        ])

    def close(self):
        self._f.close()
        print(f"\nLog written: {self.path}")
