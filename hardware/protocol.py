"""Binary CMD/SENS protocol for the custom firmware (keep in sync with arduino/main/main.ino)."""

import threading
import time

import numpy as np

CMD_HEADER  = 0xAA
SENS_HEADER = 0xBB
SENS_LEN    = 8

V_MAX     = 0.5    # m/s, controller/env command limit (≈ PWM 158, not the motor top speed)
WHEELBASE = 0.145
DT        = 0.05

CMD_MODE_PWM    = 1
PWM_MAX         = 230
# Floor speed per PWM unit, both sides: wall_test --pwm, PWM 30–100 → 0.087–0.299 m/s.
M_PER_S_PER_PWM = 0.00316


def pwm_packet(pwm_l, pwm_r):
    """CMD packet commanding raw signed PWM per side (byte = 128 + pwm/2)."""
    b = [int(np.clip(round(128 + p / 2), 0, 255)) for p in (pwm_l, pwm_r)]
    return bytes([CMD_HEADER, b[0], b[1], CMD_MODE_PWM, 0, b[0] ^ b[1] ^ CMD_MODE_PWM])


def encode_cmd(vl, vr):
    """CMD packet for wheel speeds in m/s, using the measured PWM→speed scale."""
    return pwm_packet(*(np.clip(v / M_PER_S_PER_PWM, -PWM_MAX, PWM_MAX) for v in (vl, vr)))

# IR pattern (L, C, R), 1 = on line → cross-track error in m; positive = car left of the line.
IR_CTE = {
    (0, 0, 0): float("nan"),
    (0, 0, 1): +0.025,
    (0, 1, 0):  0.000,
    (0, 1, 1): +0.013,
    (1, 0, 0): -0.025,
    (1, 0, 1):  0.000,
    (1, 1, 0): -0.013,
    (1, 1, 1):  0.000,
}


class SensReader:
    """Thread-safe SENS packet reassembler; feed() from the I/O thread, get() from the control loop.

    Without an IRCalib the analog IR bytes are thresholded at >127.
    """

    def __init__(self, calib=None):
        self._buf    = bytearray()
        self._calib  = calib
        self._latest = None
        self._lock   = threading.Lock()

    def feed(self, data):
        self._buf.extend(data)
        while len(self._buf) >= SENS_LEN:
            if self._buf[0] != SENS_HEADER:
                self._buf.pop(0)
                continue
            pkt = bytes(self._buf[:SENS_LEN])
            chk = pkt[1] ^ pkt[2] ^ pkt[3] ^ pkt[4] ^ pkt[5]
            if chk != pkt[7]:
                self._buf.pop(0)
                continue
            # Firmware sends A0, A1, A2, but on this car A0 is the RIGHT sensor.
            raw  = (pkt[3], pkt[2], pkt[1])
            dist = (pkt[4] << 8) | pkt[5]
            if self._calib is not None:
                ir, act = self._calib.read(raw)
            else:
                ir  = tuple(1 if v > 127 else 0 for v in raw)
                act = None
            with self._lock:
                self._latest = {"ir": ir, "ir_raw": raw, "ir_act": act,
                                "dist_mm": dist, "t": time.monotonic()}
            del self._buf[:SENS_LEN]

    def get(self):
        """Latest reading dict (ir, ir_raw, ir_act, dist_mm, t), or None before the first packet."""
        with self._lock:
            return self._latest
