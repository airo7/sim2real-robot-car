"""Reactive IR line follower and ultrasonic obstacle avoider for the real car.

ir = (L, C, R) with 1 = sensor sees the line; positive error = line to the RIGHT → steer right.
"""

import numpy as np

_PATTERN_ERR = {
    (0, 1, 0):  0.0,
    (1, 1, 0): -0.5,
    (1, 0, 0): -1.0,
    (0, 1, 1): +0.5,
    (0, 0, 1): +1.0,
    (1, 0, 1):  0.0,
    (1, 1, 1):  0.0,
}

_ACT_LOST = 0.12
_ACT_GATE = 0.40     # floor grain / darker planks read activations up to ~0.3
_SIDE_MIN = 0.10
_EDGE_GAIN = 0.5
_D_ALPHA  = 0.5

# Real IR flickers between found and lost as the line crosses sensor gaps; without
# debouncing the car whiplashes between PD tracking and the search turn every step.
_DEBOUNCE_LOST  = 3
_DEBOUNCE_FOUND = 2

# Search escalation, in 20 Hz steps: arc toward the last side, then pivot, then stop.
_LOST_ARC_STEPS   = 20
_LOST_PIVOT_STEPS = 70


class UltrasonicGuard:
    """Ultrasonic distance (mm) → speed scale in [0, 1], with stop/resume hysteresis."""

    def __init__(self, stop_mm=120, slow_mm=350):
        self.stop_mm = stop_mm
        self.slow_mm = slow_mm
        self._buf = []
        self._blocked = False

    def scale(self, dist_mm):
        self._buf.append(dist_mm)
        if len(self._buf) > 3:
            self._buf.pop(0)
        d = sorted(self._buf)[len(self._buf) // 2]   # median of 3: one missed echo reads 9999 ("clear")

        if self._blocked:
            if d < self.stop_mm * 1.3:
                return 0.0
            self._blocked = False

        if d <= self.stop_mm:
            self._blocked = True
            return 0.0
        if d >= self.slow_mm:
            return 1.0
        return (d - self.stop_mm) / float(self.slow_mm - self.stop_mm)


class IRLineFollower:
    def __init__(self, base_v=0.10, kp=0.15, kd=0.08, v_max=0.5,
                 stop_mm=120, slow_mm=350, lost_gain=1.3, invert=False, slowdown=0.5):
        self.base_v    = base_v
        self.slowdown  = slowdown
        self.kp        = kp
        self.kd        = kd
        self.v_max     = v_max
        self.stop_mm   = stop_mm
        self.slow_mm   = slow_mm
        self.lost_gain = lost_gain
        self.sign      = -1.0 if invert else 1.0

        self._prev_err   = 0.0
        self._d_filt     = 0.0
        self._last_dir   = 1.0       # +1 line last seen right, -1 left
        self._lost_steps = 0
        self._guard      = UltrasonicGuard(stop_mm, slow_mm)
        self._confirmed_lost = False
        self._debounce_streak = 0

    def _error(self, ir, ir_act):
        """(error in [-1, 1], lost); weighted centroid of ir_act if given, else the binary pattern."""
        if ir_act is not None:
            l, c, r = (max(0.0, (a - _ACT_GATE) / (1.0 - _ACT_GATE)) for a in ir_act)
            total = l + c + r
            if total < _ACT_LOST:
                return 0.0, True
            side = r - l
            if abs(side) >= _SIDE_MIN:
                e = side / total
                self._last_dir = 1.0 if side > 0 else -1.0
            else:
                # A weak centre-only hit (line at C's edge) has a zero centroid;
                # steer toward the last side seen, harder the weaker C is.
                e = self._last_dir * _EDGE_GAIN * (1.0 - min(c, 1.0))
            return e, False

        ir = tuple(int(x) for x in ir)
        if ir == (0, 0, 0):
            return 0.0, True
        e = _PATTERN_ERR.get(ir, 0.0)

        if abs(e) > 0.05:
            self._last_dir = 1.0 if e > 0 else -1.0
        return e, False

    def _debounce(self, raw_lost):
        if raw_lost == self._confirmed_lost:
            self._debounce_streak = 0
            return self._confirmed_lost
        self._debounce_streak += 1
        need = _DEBOUNCE_LOST if raw_lost else _DEBOUNCE_FOUND
        if self._debounce_streak >= need:
            self._confirmed_lost = raw_lost
            self._debounce_streak = 0
        return self._confirmed_lost

    def compute(self, ir, dist_mm, ir_act=None):
        """(v_left, v_right) in m/s."""
        raw_e, raw_lost = self._error(ir, ir_act)
        lost = self._debounce(raw_lost)

        if lost:
            self._lost_steps += 1
            self._d_filt = 0.0
            vl, vr = self._search()
        else:
            self._lost_steps = 0
            # Hold the last error through a not-yet-confirmed dropout.
            e = raw_e if not raw_lost else self._prev_err
            de = e - self._prev_err
            self._d_filt += _D_ALPHA * (de - self._d_filt)
            steer = self.sign * (self.kp * e + self.kd * self._d_filt)
            base = self.base_v * (1.0 - self.slowdown * min(abs(e), 1.0))
            vl = base + steer
            vr = base - steer
            self._prev_err = e

        scale = self._guard.scale(dist_mm)
        vl *= scale
        vr *= scale

        vl = float(np.clip(vl, -self.v_max, self.v_max))
        vr = float(np.clip(vr, -self.v_max, self.v_max))
        return vl, vr

    def _search(self):
        steer = self.sign * self.kp * self._last_dir * self.lost_gain
        if self._lost_steps <= _LOST_ARC_STEPS:
            base = self.base_v * 0.5
            return base + steer, base - steer
        if self._lost_steps <= _LOST_PIVOT_STEPS:
            return +steer, -steer
        return 0.0, 0.0


class ObstacleAvoider:
    """Ultrasonic-only driving: cruise, slow near obstacles, and pivot away (alternating sides) when blocked."""

    def __init__(self, base_v=0.18, v_max=0.5, stop_mm=200, slow_mm=500,
                 turn_v=0.28, turn_steps=30):
        self.base_v    = base_v
        self.v_max     = v_max
        self.turn_v    = turn_v
        self.turn_steps = turn_steps
        self._guard    = UltrasonicGuard(stop_mm, slow_mm)
        self._turning  = 0
        self._turn_dir = 1.0

    def compute(self, dist_mm):
        """(v_left, v_right) in m/s."""
        scale = self._guard.scale(dist_mm)

        if scale <= 0.0:
            if self._turning <= 0:
                self._turning  = self.turn_steps
                self._turn_dir = -self._turn_dir
            self._turning -= 1
            vl = +self._turn_dir * self.turn_v
            vr = -self._turn_dir * self.turn_v
        else:
            self._turning = 0
            vl = vr = self.base_v * scale

        vl = float(np.clip(vl, -self.v_max, self.v_max))
        vr = float(np.clip(vr, -self.v_max, self.v_max))
        return vl, vr
