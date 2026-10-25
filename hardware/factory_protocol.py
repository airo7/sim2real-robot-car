"""Elegoo V4 factory firmware JSON protocol (USB serial, or TCP :100 via the stock ESP32).

Reference: https://github.com/elegooofficial/ELEGOO-Smart-Robot-Car-Kit-V4.0-New
"""

import re

import numpy as np

from hardware.protocol import V_MAX, IR_CTE

TCP_PORT = 100
BAUD     = 9600

# Analog IR at or below this = on the black line (Elegoo MotionControl logic)
TRACKING_THRESHOLD = 250

# {H_ok} or {H_123}
RESP_RE = re.compile(r"\{([^}_]+)_([^}]*)\}")

# "N" command numbers, from the firmware's analyzeSerialData
CMD_MOTOR_LR   = 4    # D1=left PWM, D2=right PWM (forward only)
CMD_CAR        = 3    # D1=direction (3=fwd), D2=speed PWM — official app path
CMD_ULTRASONIC = 21   # D1=2 → distance cm
CMD_TRACKING   = 22   # D1=0/1/2 → L/M/R analog
CMD_IDLE       = 110  # clear → DoNothing, so the firmware doesn't fight external control

CAR_FORWARD    = 3


def analog_on_line(value: int, threshold: int = TRACKING_THRESHOLD) -> int:
    """1 = on line (dark), 0 = off line, as in IR_CTE keys."""
    return 1 if value <= threshold else 0


def ir_tuple_from_analog(left: int, mid: int, right: int) -> tuple[int, int, int]:
    return (
        analog_on_line(left),
        analog_on_line(mid),
        analog_on_line(right),
    )


def cte_from_ir(ir: tuple[int, int, int]) -> float:
    cte = IR_CTE.get(ir, float("nan"))
    return cte if not np.isnan(cte) else 0.0


def vel_to_pwm(v: float) -> int:
    """m/s wheel speed → PWM byte; clamps reverse to 0 because N=4 is forward-only."""
    v = max(0.0, float(v))
    if v < 0.02:
        return 0
    lo, hi = 55, 230
    return int(np.clip(v / V_MAX * (hi - lo) + lo, lo, hi))