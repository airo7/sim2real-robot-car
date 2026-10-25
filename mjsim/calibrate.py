"""
Fit the MuJoCo drivetrain (kv, ω_nl, μ, brake) to the real car's straight-line speeds
(wall_test --pwm), pivot yaw rates (turn_test) and driving circles (circle_test), measured
on laminate floor.

Usage:
    python3 -m mjsim.calibrate            # fit and print parameters
    python3 -m mjsim.calibrate --check    # evaluate the saved parameters only
"""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import minimize

XML = Path(__file__).parent / "elegoo_v4.xml"
PARAMS = Path(__file__).parent / "drivetrain.json"
WHEEL_R, TRACK_W = 0.033, 0.145     # TRACK_W matches the turn_test efficiency metric

STRAIGHT = {30: 0.087, 40: 0.118, 50: 0.144, 60: 0.191, 80: 0.259, 100: 0.299}   # m/s
PIVOT = {45: 0.51, 60: 0.88, 80: 1.19}                                            # rad/s
PIVOT_STALL = (30, 0.30)   # ±30 turned less than half a turn in 10 s
CIRCLE = {(45, 0): 0.52}   # rad/s: half a turn in 6 s, right side stopped
CIRCLE_MAX = {(45, 30): 0.015}   # ≤ 5° in 6 s

WHEEL_GEOMS = ("lf_tyre", "lr_tyre", "rf_tyre", "rr_tyre")


def load_model():
    model = mujoco.MjModel.from_xml_path(str(XML))
    return model, mujoco.MjData(model)


def apply(model, kv, omega_nl, mu, brake=1.0):
    for i in range(model.nu):
        model.actuator_gainprm[i, 0] = kv
    for name in WHEEL_GEOMS + ("floor",):
        model.geom_friction[model.geom(name).id, 0] = mu
    return kv, omega_nl, brake


def set_damping(model, kv, brake, pwm_l, pwm_r):
    """Back-EMF damping per side; a side commanded to PWM 0 gets kv · brake.

    The real car turned ~180° in 6 s at L45/R0, which full braking (brake = 1) underestimates ~8×.
    """
    for i, pwm in enumerate((pwm_l, pwm_l, pwm_r, pwm_r)):
        model.actuator_biasprm[i, 2] = -kv * (brake if pwm == 0 else 1.0)


def run(model, data, motor, pwm_l, pwm_r, seconds=3.0, settle=1.0):
    kv, omega_nl, brake = motor
    set_damping(model, kv, brake, pwm_l, pwm_r)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    # velocity actuator as a DC motor: torque = kv·(ω_nl·duty − ω), duty = PWM / 255
    ul, ur = omega_nl * pwm_l / 255, omega_nl * pwm_r / 255
    data.ctrl[:] = [ul, ul, ur, ur]
    n, n0 = int(seconds / model.opt.timestep), int(settle / model.opt.timestep)
    v_sum = w_sum = 0.0
    for k in range(n):
        mujoco.mj_step(model, data)
        if k >= n0:
            qw, qz = data.qpos[3], data.qpos[6]
            yaw = 2 * np.arctan2(qz, qw)
            v_sum += np.cos(yaw) * data.qvel[0] + np.sin(yaw) * data.qvel[1]
            w_sum += data.qvel[5]
    m = n - n0
    return v_sum / m, w_sum / m


def predict(model, data, params):
    motor = apply(model, *params)
    straight = {p: run(model, data, motor, p, p)[0] for p in STRAIGHT}
    pivot = {p: -run(model, data, motor, p, -p)[1] for p in list(PIVOT) + [PIVOT_STALL[0]]}
    circle = {k: -run(model, data, motor, *k, seconds=6.0)[1] for k in list(CIRCLE) + list(CIRCLE_MAX)}
    return straight, pivot, circle


def cost(log_params, model, data):
    params = np.exp(log_params)
    params[3] = min(params[3], 1.0)
    s, p, c = predict(model, data, params)
    err = [(s[k] - v) / v for k, v in STRAIGHT.items()]
    err += [(p[k] - w) / w for k, w in PIVOT.items()]
    err += [2 * (c[k] - w) / w for k, w in CIRCLE.items()]
    err += [max(0.0, abs(c[k]) - w) / 0.05 for k, w in CIRCLE_MAX.items()]
    over = max(0.0, p[PIVOT_STALL[0]] - PIVOT_STALL[1]) / PIVOT_STALL[1]
    return float(np.sum(np.square(err)) + over ** 2)


def report(model, data, params):
    s, p, c = predict(model, data, params)
    print(f"kv={params[0]:.5f} N·m·s/rad  ω_nl={params[1]:.2f} rad/s  μ={params[2]:.3f}  brake={params[3]:.3f}  "
          f"(stall torque/wheel {params[0] * params[1]:.3f} N·m, mass {model.body_subtreemass[1]:.3f} kg)")
    print("  straight PWM: " + "  ".join(f"{k}: {s[k]:.3f}/{v:.3f}" for k, v in STRAIGHT.items()) + "  (sim/real m/s)")
    print("  pivot ±PWM:   " + "  ".join(f"{k}: {p[k]:.2f}/{w:.2f}" for k, w in PIVOT.items())
          + f"  {PIVOT_STALL[0]}: {p[PIVOT_STALL[0]]:.2f}/<{PIVOT_STALL[1]}  (sim/real rad/s)")
    no_slip = {k: 2 * k / 255 * params[1] * WHEEL_R / TRACK_W for k in PIVOT}
    print("  pivot efficiency (sim): " + "  ".join(f"{k}: {p[k] / no_slip[k]:.0%}" for k in PIVOT))
    print("  circle L/R:   " + "  ".join(f"{k}: {c[k]:.2f}/{w:.2f}" for k, w in CIRCLE.items())
          + "  " + "  ".join(f"{k}: {c[k]:.3f}/≤{w}" for k, w in CIRCLE_MAX.items()) + "  (sim/real rad/s)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    model, data = load_model()

    if args.check:
        report(model, data, json.loads(PARAMS.read_text())["params"])
        return

    x0 = np.log(json.loads(PARAMS.read_text())["params"][:3] + [0.3])
    res = minimize(cost, x0, args=(model, data), method="Nelder-Mead",
                   options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 300})
    params = np.exp(res.x)
    params[3] = min(params[3], 1.0)
    params = params.tolist()
    print(f"cost {res.fun:.4f} after {res.nfev} evals")
    report(model, data, params)
    PARAMS.write_text(json.dumps({"params": params, "names": ["kv", "omega_nl", "mu", "brake"]}, indent=2))
    print(f"saved {PARAMS}")


if __name__ == "__main__":
    main()
