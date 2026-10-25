"""Gymnasium env for the custom firmware over USB serial (or any link with the SerialLink API).

Usage:
    env = SerialHardwareEnv(port="/dev/cu.usbmodem14101")
    obs, _ = env.reset()
    while True:
        action = controller(obs)
        obs, reward, done, _, info = env.step(action)

Find your port:
    python -c "import serial.tools.list_ports; [print(p) for p in serial.tools.list_ports.comports()]"
"""

import time
import threading

import numpy as np
import serial
import gymnasium as gym
from gymnasium import spaces

from hardware.protocol import V_MAX, WHEELBASE, DT, IR_CTE, SensReader, encode_cmd
from hardware.ir_calib import IRCalib, AutoIRCalib

BAUD = 9600     # USE_BLUETOOTH 1 build, over USB too; the USE_BLUETOOTH 0 build runs at 115200


class SerialLink:
    def __init__(self, port: str, baud: int = BAUD, calib=None):
        self._ser = serial.Serial(port, baud, timeout=0.1)
        time.sleep(2.0)   # opening the port resets the Arduino
        self._ser.reset_input_buffer()

        if calib is None:
            calib = IRCalib.load()
            if calib is None:
                print("IR calib: none saved — self-calibrating automatically as the car "
                      "drives (AutoIRCalib). Run ir_calib.py --calibrate instead for a "
                      "fixed, repeatable calibration.")
                calib = AutoIRCalib()
        self._reader  = SensReader(calib)
        self._running = True
        self._thread  = threading.Thread(target=self._recv_loop, daemon=True)
        self._thread.start()

    def send_cmd(self, vl: float, vr: float):
        self._ser.write(encode_cmd(vl, vr))

    def latest_sensors(self):
        return self._reader.get()

    def _recv_loop(self):
        while self._running:
            try:
                data = self._ser.read(self._ser.in_waiting or 1)
            except serial.SerialException:
                break
            self._reader.feed(data)

    def stop(self):
        self._running = False
        self.send_cmd(0.0, 0.0)
        self._ser.close()


class SerialHardwareEnv(gym.Env):
    """obs = [cte, sin e_psi, cos e_psi, v / V_MAX]; action = (vl, vr) in m/s."""

    metadata = {"render_modes": []}

    def __init__(self, port: str = None, max_steps: int = 2000, baud: int = BAUD, link=None):
        super().__init__()
        if link is None:
            print(f"Opening serial port {port} @ {baud} …")
            link = SerialLink(port, baud)
        self.link      = link
        self.max_steps = max_steps

        self.observation_space = spaces.Box(
            low=np.array([-2.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([ 2.0,  1.0,  1.0,  1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-V_MAX, -V_MAX], dtype=np.float32),
            high=np.array([ V_MAX,  V_MAX], dtype=np.float32),
        )

        self._prev_cte   = 0.0
        self._step_count = 0
        self.state       = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.link.send_cmd(0.0, 0.0)
        self._prev_cte   = 0.0
        self._step_count = 0
        self.state       = np.zeros(3)

        for _ in range(20):
            s = self.link.latest_sensors()
            if s is not None:
                break
            time.sleep(0.05)

        return self._make_obs(self.link.latest_sensors(), np.zeros(2)), {}

    def step(self, action: np.ndarray):
        t0 = time.monotonic()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.link.send_cmd(float(action[0]), float(action[1]))
        self._step_count += 1

        v     = (action[0] + action[1]) / 2.0
        omega = (action[1] - action[0]) / WHEELBASE
        th    = self.state[2]
        self.state = np.array([
            self.state[0] + v * np.cos(th) * DT,
            self.state[1] + v * np.sin(th) * DT,
            (th + omega * DT + np.pi) % (2 * np.pi) - np.pi,
        ])

        elapsed = time.monotonic() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

        sensors = self.link.latest_sensors()
        obs     = self._make_obs(sensors, action)
        cte     = float(obs[0])
        reward  = -abs(cte) - 0.1 * abs(float(obs[1]))

        terminated = abs(cte) > 0.8 or self._step_count >= self.max_steps
        info = {
            "cte":     cte,
            "ir":      sensors["ir"]      if sensors else (0, 0, 0),
            "dist_mm": sensors["dist_mm"] if sensors else 9999,
        }
        return obs, float(reward), terminated, False, info

    def close(self):
        self.link.stop()

    def _make_obs(self, sensors, action) -> np.ndarray:
        if sensors is None:
            return np.zeros(4, dtype=np.float32)

        act = sensors.get("ir_act")
        if act is not None and sum(act) > 0.12:
            l, c, r = act
            cte = (r - l) / (l + c + r) * 0.025
        else:
            cte = IR_CTE.get(sensors["ir"], float("nan"))
        if np.isnan(cte):
            cte = self._prev_cte

        v = (action[0] + action[1]) / 2.0
        if abs(v) > 0.05:
            e_psi = np.arctan2(cte - self._prev_cte, v * DT)
        else:
            e_psi = 0.0
        e_psi = np.clip(e_psi, -np.pi, np.pi)

        self._prev_cte = cte

        return np.array([
            np.clip(cte, -2.0, 2.0),
            np.sin(e_psi),
            np.cos(e_psi),
            np.clip(v / V_MAX, -1.0, 1.0),
        ], dtype=np.float32)
