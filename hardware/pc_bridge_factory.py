"""Gymnasium env for the Elegoo V4 factory firmware (JSON protocol) over USB serial or WiFi TCP.

Usage:
    env = FactoryHardwareEnv(serial_port="/dev/cu.usbmodem00001")   # USB @ 9600
    env = FactoryHardwareEnv(esp32_ip="192.168.4.1")                 # stock ESP32, TCP :100
"""

import json
import socket
import time
import threading

import numpy as np
import serial
import gymnasium as gym
from gymnasium import spaces

from hardware.protocol import WHEELBASE, DT
from hardware.factory_protocol import (
    TCP_PORT, BAUD, V_MAX,
    CMD_MOTOR_LR, CMD_CAR, CMD_ULTRASONIC, CMD_TRACKING, CMD_IDLE,
    CAR_FORWARD,
    RESP_RE, ir_tuple_from_analog, cte_from_ir, vel_to_pwm,
)


class _FactoryJSONLink:
    def _next_id(self) -> str:
        self._seq = (self._seq % 9999) + 1
        return str(self._seq)

    def _send_raw(self, payload: dict) -> str | None:
        payload = dict(payload)
        if "H" not in payload:
            payload["H"] = self._next_id()
        msg = json.dumps(payload, separators=(",", ":"))
        expected_h = str(payload["H"])
        with self._lock:
            self._write(msg)
            frame = self._read_frame(expected_h)
        if getattr(self, "_debug", False):
            tag = frame or "TIMEOUT"
            print(f"  TX {msg}")
            print(f"  RX {tag}")
        return frame

    def init_session(self) -> str | None:
        return self._send_raw({"N": CMD_IDLE})

    def set_motors(self, vl: float, vr: float):
        pl = vel_to_pwm(vl)
        pr = vel_to_pwm(vr)
        ack = self._send_raw({
            "N": CMD_MOTOR_LR,
            "D1": pl,
            "D2": pr,
        })
        if ack is None:
            self._try_car_forward(pl, pr)

    def _try_car_forward(self, pl: int, pr: int):
        """Fallback when N=4 gets no ack — official Elegoo car-forward command."""
        if pl == 0 and pr == 0:
            return
        speed = max(pl, pr)
        self._send_raw({
            "N": CMD_CAR,
            "H": "car_fwd",
            "D1": CAR_FORWARD,
            "D2": speed,
        })

    def read_tracking(self) -> tuple[int, int, int]:
        vals = []
        for sid in (0, 1, 2):
            frame = self._send_raw({"N": CMD_TRACKING, "D1": sid})
            if frame is None:
                vals.append(512)
                continue
            m = RESP_RE.match(frame)
            try:
                vals.append(int(m.group(2)))
            except (TypeError, ValueError, AttributeError):
                vals.append(512)
        return ir_tuple_from_analog(vals[0], vals[1], vals[2])

    def read_distance_mm(self) -> int:
        frame = self._send_raw({"N": CMD_ULTRASONIC, "D1": 2})
        if frame is None:
            return 9999
        m = RESP_RE.match(frame)
        try:
            return int(float(m.group(2)) * 10)
        except (TypeError, ValueError, AttributeError):
            return 9999

    def stop(self):
        try:
            self._send_raw({"N": CMD_MOTOR_LR, "D1": 0, "D2": 0})
            self._send_raw({"N": CMD_IDLE})
        except OSError:
            pass
        self._close()


class FactoryUSBLink(_FactoryJSONLink):
    """Direct USB to Arduino @ 9600 baud (factory firmware)."""

    def __init__(self, port: str, baud: int = BAUD, timeout: float = 0.5):
        self._seq  = 0
        self._lock = threading.Lock()
        self._buf  = ""
        print(f"Opening factory USB serial {port} @ {baud} …")
        self._ser = serial.Serial(port, baud, timeout=timeout)
        time.sleep(2.0)
        self._ser.reset_input_buffer()
        print("USB serial ready.")

    def _write(self, msg: str):
        self._ser.write((msg + "\n").encode())

    def _read_frame(self, expected_h: str | None = None) -> str | None:
        deadline = time.monotonic() + 0.8
        while time.monotonic() < deadline:
            chunk = self._ser.read(self._ser.in_waiting or 1)
            if chunk:
                self._buf += chunk.decode(errors="ignore")
            while "}" in self._buf:
                end = self._buf.index("}") + 1
                frame = self._buf[:end]
                self._buf = self._buf[end:]
                if frame.strip() in ("{Heartbeat}", "{Factory}"):
                    continue
                m = RESP_RE.search(frame)
                if m:
                    if expected_h is None or m.group(1) == expected_h:
                        return m.group(0)
            if not chunk:
                time.sleep(0.01)
        return None

    def _close(self):
        self._ser.close()


class FactoryTCPLink(_FactoryJSONLink):
    """TCP :100 via ESP32 WiFi bridge."""

    _HEARTBEAT = "{Heartbeat}"
    _IGNORE = frozenset({"{Heartbeat}", "{Factory}"})

    def __init__(self, ip: str, port: int = TCP_PORT, timeout: float = 3.0, debug: bool = False):
        self._seq      = 0
        self._lock     = threading.RLock()
        self._ip       = ip
        self._port     = port
        self._timeout  = timeout
        self._debug    = debug
        self._rx_text  = ""
        self._responses: list[str] = []
        self._resp_cv  = threading.Condition()
        self._stop_rx  = threading.Event()
        self._alive    = threading.Event()
        self._rx_thread: threading.Thread | None = None
        self._connect()

    def _connect(self):
        self._stop_rx.clear()
        self._responses.clear()
        self._rx_text = ""
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(self._timeout)
        print(f"Connecting TCP {self._ip}:{self._port} …")
        try:
            sock.connect((self._ip, self._port))
        except (TimeoutError, OSError) as e:
            sock.close()
            raise ConnectionError(
                f"Cannot connect to {self._ip}:{self._port}. Camera may work but TCP "
                f"bridge is offline — use USB instead:\n"
                f"  python3 -m hardware.test_factory --serial /dev/cu.usbmodem00001\n"
                f"Original error: {e}"
            ) from e
        sock.settimeout(0.25)
        self._sock = sock
        self._alive.set()
        self._rx_thread = threading.Thread(target=self._rx_loop, daemon=True)
        self._rx_thread.start()
        time.sleep(0.3)
        print("TCP connected.")

    def _send_raw(self, payload: dict) -> str | None:
        payload = dict(payload)
        if "H" not in payload:
            payload["H"] = self._next_id()
        msg = json.dumps(payload, separators=(",", ":"))
        expected_h = str(payload["H"])
        with self._lock:
            for attempt in range(2):
                try:
                    self._write(msg)
                    frame = self._read_frame(expected_h)
                    if self._debug:
                        print(f"  TX {msg}")
                        print(f"  RX {frame or 'TIMEOUT'}")
                    return frame
                except (BrokenPipeError, ConnectionResetError, OSError):
                    if attempt == 0:
                        self._reconnect()
                    else:
                        raise
        return None

    def _echo_heartbeat(self):
        with self._lock:
            if not self._alive.is_set():
                return
            try:
                self._sock.sendall(self._HEARTBEAT.encode())
            except OSError:
                self._alive.clear()

    def _rx_loop(self):
        while not self._stop_rx.is_set():
            try:
                chunk = self._sock.recv(4096)
            except socket.timeout:
                continue
            except OSError:
                break
            if not chunk:
                break
            self._rx_text += chunk.decode(errors="ignore")
            while "}" in self._rx_text:
                end = self._rx_text.index("}") + 1
                frame = self._rx_text[:end].strip()
                self._rx_text = self._rx_text[end:]
                if frame == self._HEARTBEAT:
                    self._echo_heartbeat()
                    continue
                if frame in self._IGNORE:
                    continue
                m = RESP_RE.search(frame)
                if m:
                    with self._resp_cv:
                        self._responses.append(m.group(0))
                        self._resp_cv.notify_all()
        self._alive.clear()
        with self._resp_cv:
            self._resp_cv.notify_all()

    def _write(self, msg: str):
        if not self._alive.is_set():
            raise BrokenPipeError("TCP link down")
        self._sock.sendall(msg.encode())

    def _read_frame(self, expected_h: str | None = None) -> str | None:
        deadline = time.monotonic() + 2.5
        with self._resp_cv:
            while time.monotonic() < deadline:
                for i, resp in enumerate(self._responses):
                    m = RESP_RE.match(resp)
                    if not m:
                        continue
                    if expected_h is None or m.group(1) == expected_h:
                        return self._responses.pop(i)
                if not self._alive.is_set():
                    break
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    break
                self._resp_cv.wait(timeout=min(remaining, 0.05))
        return None

    def _reconnect(self):
        self._stop_rx.set()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass
        self._connect()
        super().init_session()

    def _close(self):
        self._stop_rx.set()
        self._alive.clear()
        with self._resp_cv:
            self._resp_cv.notify_all()
        if self._rx_thread and self._rx_thread.is_alive():
            self._rx_thread.join(timeout=1.0)
        try:
            self._sock.close()
        except OSError:
            pass


class FactoryHardwareEnv(gym.Env):
    """Gymnasium env over Elegoo factory JSON protocol."""

    metadata = {"render_modes": []}

    def __init__(
        self,
        esp32_ip: str | None = None,
        serial_port: str | None = None,
        max_steps: int = 2000,
    ):
        super().__init__()
        if serial_port:
            self.link = FactoryUSBLink(serial_port)
        elif esp32_ip:
            self.link = FactoryTCPLink(esp32_ip)
        else:
            raise ValueError("Provide serial_port or esp32_ip")

        self.max_steps = max_steps
        self.observation_space = spaces.Box(
            low=np.array([-2.0, -1.0, -1.0, -1.0], dtype=np.float32),
            high=np.array([2.0, 1.0, 1.0, 1.0], dtype=np.float32),
        )
        self.action_space = spaces.Box(
            low=np.array([-V_MAX, -V_MAX], dtype=np.float32),
            high=np.array([V_MAX, V_MAX], dtype=np.float32),
        )
        self._prev_cte   = 0.0
        self._step_count = 0
        self.state       = np.zeros(3)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.link.init_session()
        self.link.set_motors(0.0, 0.0)
        self._prev_cte   = 0.0
        self._step_count = 0
        self.state       = np.zeros(3)
        time.sleep(0.1)
        return self._make_obs(np.zeros(2), (0, 1, 0)), {}

    def step(self, action: np.ndarray):
        t0 = time.monotonic()
        action = np.clip(action, self.action_space.low, self.action_space.high)

        self.link.set_motors(float(action[0]), float(action[1]))
        self._step_count += 1

        v     = (action[0] + action[1]) / 2.0
        omega = (action[1] - action[0]) / WHEELBASE
        th    = self.state[2]
        self.state = np.array([
            self.state[0] + v * np.cos(th) * DT,
            self.state[1] + v * np.sin(th) * DT,
            (th + omega * DT + np.pi) % (2 * np.pi) - np.pi,
        ])

        ir      = self.link.read_tracking()
        dist_mm = self.link.read_distance_mm()
        obs     = self._make_obs(action, ir)
        cte     = float(obs[0])
        reward  = -abs(cte) - 0.1 * abs(float(obs[1]))

        elapsed = time.monotonic() - t0
        if elapsed < DT:
            time.sleep(DT - elapsed)

        terminated = abs(cte) > 0.8 or self._step_count >= self.max_steps
        info = {"cte": cte, "ir": ir, "dist_mm": dist_mm}
        return obs, float(reward), terminated, False, info

    def close(self):
        self.link.stop()

    def _make_obs(self, action: np.ndarray, ir: tuple) -> np.ndarray:
        cte = cte_from_ir(ir)
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