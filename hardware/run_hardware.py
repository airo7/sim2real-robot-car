"""
Drive the real Elegoo V4.

    python3 -m hardware.run_hardware                                   # PD line follower over BLE
    python3 -m hardware.run_hardware --controller car_rl               # sim-trained policy (v8)
    python3 -m hardware.run_hardware --controller avoid                # ultrasonic-only wander
    python3 -m hardware.run_hardware --link custom --serial /dev/cu.usbmodem00001
    python3 -m hardware.run_hardware --link factory --ip 192.168.4.1   # stock Elegoo firmware over WiFi

Every run is logged to hardware/logs/run_<timestamp>.csv unless --no-log.
"""

import argparse
import signal
import sys
import time

import numpy as np

DEFAULT_MODEL = "rl/policy/v8_ppo/car_ppo.zip"
STALE_S = 0.5

_env_ref = None
_logger_ref = None


def _signal_handler(sig, frame):
    print("\nShutting down — stopping motors.")
    if _env_ref is not None:
        if hasattr(_env_ref.link, "set_motors"):
            _env_ref.link.set_motors(0.0, 0.0)
        elif hasattr(_env_ref.link, "send_cmd"):
            _env_ref.link.send_cmd(0.0, 0.0)
        _env_ref.close()
    if _logger_ref is not None:
        _logger_ref.close()
    sys.exit(0)


signal.signal(signal.SIGINT, _signal_handler)


def _fresh(env):
    s = env.link.latest_sensors()
    if s is None or time.monotonic() - s.get("t", 0.0) > STALE_S:
        return None
    return s


def make_line_controller(base_v, kp, kd, stop_mm, slow_mm, invert, lost_gain, slowdown):
    """Reactive PD on the IR centroid with ultrasonic slow-down/stop."""
    from hardware.line_follower import IRLineFollower

    follower = IRLineFollower(base_v=base_v, kp=kp, kd=kd, stop_mm=stop_mm, slow_mm=slow_mm,
                              lost_gain=lost_gain, invert=invert, slowdown=slowdown)
    print(f"Line follower: base_v={base_v} kp={kp} kd={kd} lost_gain={lost_gain} slowdown={slowdown} "
          f"stop={stop_mm}mm slow={slow_mm}mm invert={invert}")

    def _policy(obs, env):
        s = _fresh(env)
        if s is None:
            return np.zeros(2)
        return np.array(follower.compute(s["ir"], s["dist_mm"], ir_act=s.get("ir_act")))

    return _policy


def make_avoid_controller(base_v, stop_mm, slow_mm):
    """Ultrasonic-only cruise and turn-away; isolates the radio and motor path from IR sensing."""
    from hardware.line_follower import ObstacleAvoider

    avoider = ObstacleAvoider(base_v=base_v, stop_mm=stop_mm, slow_mm=slow_mm)
    print(f"Obstacle avoider: base_v={base_v} stop={stop_mm}mm slow={slow_mm}mm")

    def _policy(obs, env):
        s = _fresh(env)
        return np.zeros(2) if s is None else np.array(avoider.compute(s["dist_mm"]))

    return _policy


def make_car_rl_controller(model_path, stop_mm, slow_mm, max_u=1.0):
    """Policy trained in mjsim.car_env; the observation layout comes from its env_config.json."""
    from hardware.line_follower import UltrasonicGuard
    from hardware.protocol import M_PER_S_PER_PWM
    from mjsim.car_env import N_PRIV, US_MAX, command_pwm
    from rl.train_car import load_config, load_policy, make_actor

    cfg = load_config(model_path)
    actor = make_actor(*load_policy(model_path))
    guard = UltrasonicGuard(stop_mm, slow_mm)
    state = {"frames": None, "u": np.zeros(2)}
    print(f"Car RL policy: {model_path}  config={cfg}  max_u={max_u} stop={stop_mm}mm slow={slow_mm}mm")

    def _policy(obs, env):
        s = _fresh(env)
        if s is None or s.get("ir_act") is None:
            state["u"] = np.zeros(2)
            return np.zeros(2)
        # must match ElegooCarEnv._frame: IR L,C,R [, ultrasonic], (u_prev + 1) / 2
        parts = [s["ir_act"]]
        if cfg.ultrasonic:
            parts.append([min(s["dist_mm"] / 1000.0, US_MAX) / US_MAX])
        frame = np.concatenate(parts + [(state["u"] + 1) / 2])
        if state["frames"] is None:
            state["frames"] = [frame] * cfg.frames
        state["frames"] = state["frames"][1:] + [frame]
        x = np.concatenate(state["frames"])
        if cfg.privileged:
            x = np.r_[x, np.zeros(N_PRIV)]          # the actor masks this tail; only the critic used it
        u = np.clip(actor(x.astype(np.float32)), -max_u, max_u) * guard.scale(s["dist_mm"])
        state["u"] = u
        return command_pwm(u, cfg) * M_PER_S_PER_PWM

    return _policy


def _print_step(step, info, action, comp_ms):
    print(f"\r  step={step:4d}  IR={info.get('ir', '?')}  dist={info.get('dist_mm', 9999):4d}mm  "
          f"v=[{action[0]:+.2f},{action[1]:+.2f}]  comp={comp_ms:.1f}ms", end="", flush=True)


def _control_loop(env, policy, max_steps, dashboard=None, logger=None):
    global _env_ref
    _env_ref = env
    obs, _ = env.reset()
    for step in range(max_steps):
        t0 = time.perf_counter()
        action = policy(obs, env)
        comp_ms = (time.perf_counter() - t0) * 1000
        obs, _, terminated, truncated, info = env.step(action)
        _print_step(step, info, action, comp_ms)
        if logger is not None:
            logger.log(step, env.link.latest_sensors(), action, cte=info["cte"], comp_ms=comp_ms)
        if dashboard is not None:
            dashboard.push(step, info["cte"], info.get("ir", (0, 0, 0)), info.get("dist_mm", 9999),
                           float(action[0]), float(action[1]), *map(float, env.state[:3]))
        if terminated or truncated:
            obs, _ = env.reset()
    print("\nMax steps reached. Stopping.")
    env.close()
    if logger is not None:
        logger.close()


def _print_calib_summary():
    from hardware.ir_calib import IRCalib
    calib = IRCalib.load()
    if calib is None:
        print("IR calib: none saved — self-calibrating while driving (AutoIRCalib). "
              "For a fixed calibration run: python3 -m hardware.ir_calib --calibrate")
    else:
        print(f"IR calib: thresholds={calib.thresholds} floor={calib.floor} "
              f"line={calib.line} line_when_above={calib.line_when_above}")


def make_env(link, serial_port, baud, ip, address, max_steps):
    if link == "ble":
        from hardware.pc_bridge_ble import make_ble_env
        return make_ble_env(address=address, max_steps=max_steps)
    if link == "custom":
        from hardware.pc_bridge_serial import SerialHardwareEnv
        print(f"Custom binary serial — {serial_port} @ {baud}")
        return SerialHardwareEnv(port=serial_port, max_steps=max_steps, baud=baud)
    from hardware.pc_bridge_factory import FactoryHardwareEnv
    if serial_port:
        print(f"Factory JSON over USB — {serial_port} @ 9600")
        return FactoryHardwareEnv(serial_port=serial_port, max_steps=max_steps)
    print(f"Factory JSON over WiFi — {ip}:100")
    return FactoryHardwareEnv(esp32_ip=ip, max_steps=max_steps)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["line", "car_rl", "avoid"], default="line")
    ap.add_argument("--link", choices=["ble", "custom", "factory"], default="ble",
                    help="ble = BLE UART module; custom = binary firmware over USB; factory = stock Elegoo firmware")
    ap.add_argument("--address", metavar="UUID", help="BLE address (otherwise scans for 'HC-05')")
    ap.add_argument("--serial", metavar="PORT", help="serial port for --link custom/factory")
    ap.add_argument("--baud", type=int, default=9600, help="custom firmware baud (USE_BLUETOOTH build: 9600)")
    ap.add_argument("--ip", default="192.168.4.1", help="factory ESP32 address")
    ap.add_argument("--steps", type=int, default=2000)
    ap.add_argument("--track", choices=["rect", "oval", "figure8"], default="rect", help="map shown by --live")
    ap.add_argument("--live", action="store_true", help="live matplotlib dashboard")
    ap.add_argument("--base-v", type=float, default=0.10, help="line: cruise speed (m/s)")
    ap.add_argument("--kp", type=float, default=0.15, help="line: steering gain (m/s per unit error)")
    ap.add_argument("--kd", type=float, default=0.08, help="line: steering damping")
    ap.add_argument("--lost-gain", type=float, default=1.3, help="line: search turn = kp × this")
    ap.add_argument("--slowdown", type=float, default=0.5, help="line: forward speed cut at full error")
    ap.add_argument("--invert", action="store_true", help="line: flip steering")
    ap.add_argument("--stop-mm", type=int, default=120, help="ultrasonic hard stop")
    ap.add_argument("--slow-mm", type=int, default=350, help="ultrasonic slow-down start")
    ap.add_argument("--model", default=DEFAULT_MODEL, help="car_rl: policy zip")
    ap.add_argument("--max-u", type=float, default=1.0, help="car_rl: clip the policy action")
    ap.add_argument("--no-log", action="store_true", help="disable CSV logging")
    args = ap.parse_args()

    if args.link == "custom" and not args.serial:
        ap.error("--link custom needs --serial")
    if args.controller == "line":
        _print_calib_summary()

    env = make_env(args.link, args.serial, args.baud, args.ip, args.address, args.steps)
    if args.controller == "line":
        policy = make_line_controller(args.base_v, args.kp, args.kd, args.stop_mm, args.slow_mm,
                                      args.invert, args.lost_gain, args.slowdown)
    elif args.controller == "avoid":
        policy = make_avoid_controller(args.base_v, args.stop_mm, args.slow_mm)
    else:
        policy = make_car_rl_controller(args.model, args.stop_mm, args.slow_mm, args.max_u)

    global _logger_ref
    logger = None
    if not args.no_log:
        from hardware.logger import RunLogger
        logger = _logger_ref = RunLogger()
        print(f"Logging to {logger.path}")
    print("Starting control loop — Ctrl+C to stop.\n")

    if args.live:
        from hardware.dashboard import LiveDashboard
        from mjsim.track import Track
        dash = LiveDashboard(title=f"Elegoo V4 — {args.controller}", track=Track.from_name(args.track))
        dash.start(lambda: _control_loop(env, policy, args.steps, dashboard=dash, logger=logger))
    else:
        _control_loop(env, policy, args.steps, logger=logger)


if __name__ == "__main__":
    main()
