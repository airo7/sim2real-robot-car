"""
Watch a controller drive the realistic sim car in a live window (real-time, 20 Hz).

Usage:
    python3 -m mjsim.watch --controller pd --track L
    python3 -m mjsim.watch --controller rl --track random --randomize
"""

import argparse
import functools
import time

import matplotlib.pyplot as plt

from mjsim.baseline import pd_policy
from mjsim.car_env import DT, TEST_TRACKS, ElegooCarEnv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["pd", "rl"], default="rl")
    ap.add_argument("--model", default="rl/policy/v8_ppo/car_ppo.zip")
    ap.add_argument("--track", choices=["L", "random", *TEST_TRACKS], default="L")
    ap.add_argument("--randomize", action="store_true")
    ap.add_argument("--speed", type=float, default=1.0, help="playback speed multiplier")
    args = ap.parse_args()

    from rl.train_car import PD_TUNE, load_config, load_policy, make_actor
    env = ElegooCarEnv(track=args.track, randomize=args.randomize, max_steps=1200,
                       config=load_config(args.model))
    if args.controller == "rl":
        make = functools.partial(make_actor, *load_policy(args.model))
    else:
        make = functools.partial(pd_policy, env, **PD_TUNE)
    plt.ion()
    fig, ax = plt.subplots(figsize=(9, 6.75))
    ax.axis("off")
    fig.tight_layout(pad=0)
    im = text = None
    ep = 0
    while plt.fignum_exists(fig.number):
        obs, _ = env.reset(seed=ep)
        policy = make()
        while plt.fignum_exists(fig.number):
            t0 = time.monotonic()
            obs, _, term, trunc, info = env.step(policy(obs))
            frame = env.render()
            label = (f"{args.controller.upper()}  track={args.track}  ep {ep}   "
                     f"progress {info['progress_m']:.2f} m   offset {info['cte'] * 1000:+.0f} mm   "
                     f"IR L{obs[env.ir_index]:.2f} C{obs[env.ir_index + 1]:.2f} R{obs[env.ir_index + 2]:.2f}")
            if im is None:
                im = ax.imshow(frame)
                text = ax.text(8, 20, label, color="white", fontsize=9, family="monospace",
                               bbox=dict(facecolor="black", alpha=0.6, pad=3))
            else:
                im.set_data(frame)
                text.set_text(label)
            plt.pause(max(0.001, DT / args.speed - (time.monotonic() - t0)))
            if term or trunc:
                outcome = "finished" if info["finished"] else "LOST the line" if info["lost"] else "time up"
                print(f"episode {ep}: {outcome}, progress {info['progress_m']:.2f} m")
                break
        ep += 1
    env.close()


if __name__ == "__main__":
    main()
