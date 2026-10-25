"""
Watch a training run live: drive the newest checkpoint (reloaded after every episode) next to
the learning curves, alternating the L track and random tracks at the current difficulty.

Usage:
    python3 -m mjsim.watch_training --run rl/policy/v9_ppo
"""

import argparse
import glob
import re
import time
from pathlib import Path

import matplotlib.pyplot as plt
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

from mjsim.car_env import DT, ElegooCarEnv
from rl.train_car import load_config, load_policy, make_actor


def _steps(path):
    return int(re.search(r"_(\d+)_steps", path).group(1))


def newest_checkpoint(run):
    ckpts = glob.glob(str(Path(run) / "car_ckpt" / "car_ppo_*_steps.zip"))
    if not ckpts:
        return None, 0
    best = max(ckpts, key=_steps)
    return best, _steps(best)


def curves(run):
    files = sorted(glob.glob(str(Path(run) / "tb" / "*")))
    if not files:
        return {}
    ea = EventAccumulator(files[-1], size_guidance={"scalars": 0})
    ea.Reload()
    out = {}
    for tag in ("rollout/ep_rew_mean", "eval/mean_reward", "curriculum/level", "rollout/ep_len_mean"):
        if tag in ea.Tags()["scalars"]:
            v = ea.Scalars(tag)
            out[tag] = ([e.step / 1e6 for e in v], [e.value for e in v])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="rl/policy/v9_ppo")
    ap.add_argument("--speed", type=float, default=2.0, help="playback speed multiplier")
    args = ap.parse_args()

    plt.ion()
    fig = plt.figure(figsize=(14, 6.5))
    ax_sim = fig.add_axes([0.0, 0.0, 0.52, 1.0])
    ax_rew = fig.add_axes([0.58, 0.55, 0.39, 0.38])
    ax_lvl = fig.add_axes([0.58, 0.09, 0.39, 0.33])
    ax_sim.axis("off")
    im = text = None
    loaded, loaded_steps, ep, results = None, 0, 0, []

    while plt.fignum_exists(fig.number):
        path, steps = newest_checkpoint(args.run)
        if path is None:
            ax_sim.set_title("waiting for the first checkpoint…")
            plt.pause(5.0)
            continue
        if path != loaded:
            model, rec = load_policy(path)
            cfg = load_config(path)
            loaded, loaded_steps = path, steps

        track = "L" if ep % 2 == 0 else "random"
        env = ElegooCarEnv(track=track, randomize=True, max_steps=1200, config=cfg)
        c = curves(args.run)
        if "curriculum/level" in c:
            env.set_level(c["curriculum/level"][1][-1])
        obs, _ = env.reset(seed=ep)
        act = make_actor(model, rec)

        ax_rew.cla()
        ax_lvl.cla()
        if "rollout/ep_rew_mean" in c:
            ax_rew.plot(*c["rollout/ep_rew_mean"], lw=1.2, label="training (stochastic)")
        if "eval/mean_reward" in c:
            ax_rew.plot(*c["eval/mean_reward"], "o-", ms=3, lw=1, label="evaluation (deterministic)")
        ax_rew.axvline(loaded_steps / 1e6, color="0.5", ls="--", lw=0.8)
        ax_rew.set_title("episode reward", fontsize=10)
        ax_rew.legend(fontsize=8)
        ax_rew.grid(alpha=0.3)
        if "curriculum/level" in c:
            ax_lvl.plot(*c["curriculum/level"], color="tab:green", lw=1.2)
        ax_lvl.set_ylim(0, 1.05)
        ax_lvl.set_xlabel("training steps (millions)")
        ax_lvl.set_title("curriculum difficulty", fontsize=10)
        ax_lvl.grid(alpha=0.3)
        if results:
            ok = sum(r == "finished" or r == "time up" for r in results[-10:])
            ax_lvl.text(0.02, 0.85, f"watcher, last {min(10, len(results))} episodes: {ok} without losing the line",
                        transform=ax_lvl.transAxes, fontsize=8)

        while plt.fignum_exists(fig.number):
            t0 = time.monotonic()
            obs, _, term, trunc, info = env.step(act(obs))
            frame = env.render()
            label = (f"checkpoint {loaded_steps / 1e6:.1f}M steps   track={track}   ep {ep}\n"
                     f"progress {info['progress_m']:.2f} m   offset {info['cte'] * 1000:+.0f} mm   "
                     f"IR L{obs[env.ir_index]:.2f} C{obs[env.ir_index + 1]:.2f} R{obs[env.ir_index + 2]:.2f}")
            if im is None:
                im = ax_sim.imshow(frame)
                text = ax_sim.text(8, 30, label, color="white", fontsize=9, family="monospace",
                                   bbox=dict(facecolor="black", alpha=0.6, pad=3))
            else:
                im.set_data(frame)
                text.set_text(label)
            plt.pause(max(0.001, DT / args.speed - (time.monotonic() - t0)))
            if term or trunc:
                outcome = ("finished" if info["finished"] else "crashed" if info["collided"]
                           else "LOST" if info["lost"] else "time up")
                results.append(outcome)
                print(f"[{loaded_steps / 1e6:.1f}M] ep {ep} {track}: {outcome}, {info['progress_m']:.2f} m", flush=True)
                break
        env.close()
        ep += 1


if __name__ == "__main__":
    main()
