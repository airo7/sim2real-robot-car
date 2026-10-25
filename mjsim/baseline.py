"""
Run the real car's IR line follower (hardware/line_follower.py) inside ElegooCarEnv:
a check of the sim against real runs and the baseline an RL policy must beat.

Usage:
    python3 -m mjsim.baseline                 # L track, real-run settings
    python3 -m mjsim.baseline --track random --randomize --episodes 50
"""

import argparse

import numpy as np

from hardware.line_follower import IRLineFollower
from hardware.protocol import M_PER_S_PER_PWM
from mjsim.car_env import TEST_TRACKS, US_MAX, ElegooCarEnv

SETTINGS = {
    "run3 (kp .25)": dict(kp=0.25, kd=0.08, lost_gain=0.8, slowdown=0.0),
    "run4 defaults": dict(kp=0.15, kd=0.08, lost_gain=1.3, slowdown=0.5),
    "kp .12":        dict(kp=0.12, kd=0.04, lost_gain=0.8, slowdown=0.0),
}


def pd_policy(env, **tune):
    """The real-car PD rule on the newest frame's IR (and ultrasonic, if the env provides it)."""
    ctl = IRLineFollower(base_v=0.10, **tune)
    i, us = env.ir_index, env.cfg.ultrasonic

    def act(obs):
        ir_act = tuple(float(a) for a in obs[i:i + 3])
        dist_mm = obs[i + 3] * US_MAX * 1000 if us else 9999
        vl, vr = ctl.compute(tuple(int(a > 0.5) for a in ir_act), dist_mm, ir_act)
        return np.array([vl, vr]) / M_PER_S_PER_PWM / env.cfg.pwm_scale

    return act


def evaluate(make_policy, env, episodes, seed=0):
    out = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        policy = make_policy()
        lost_frames = steps = 0
        i = env.ir_index
        while True:
            obs, _, term, trunc, info = env.step(policy(obs))
            steps += 1
            lost_frames += max(obs[i:i + 3]) < 0.4
            if term or trunc:
                break
        out.append((info["finished"], info["lost"], info["progress_m"], lost_frames / steps,
                    steps * 0.05, info.get("collided", False)))
    a = np.array(out, float)
    return {"finished": a[:, 0].mean(), "lost": a[:, 1].mean(), "progress_m": a[:, 2].mean(),
            "lost_frames": a[:, 3].mean(), "time_s": a[:, 4].mean(), "collided": a[:, 5].mean()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", default="L", choices=["L", "random", *TEST_TRACKS])
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()

    env = ElegooCarEnv(track=args.track, randomize=args.randomize, max_steps=1200)
    print(f"track={args.track} randomize={args.randomize} episodes={args.episodes}")
    for name, tune in SETTINGS.items():
        r = evaluate(lambda: pd_policy(env, **tune), env, args.episodes)
        print(f"  {name:14s} finished {r['finished']:4.0%}  lost-the-track {r['lost']:4.0%}  "
              f"progress {r['progress_m']:4.2f} m  line-lost frames {r['lost_frames']:4.0%}  time {r['time_s']:4.1f}s")


if __name__ == "__main__":
    main()
