"""
Compare a real run (log + tracked phone video) with the simulator.

Closed loop: the same policy on the same track in simulation, weave and corner-overshoot statistics
side by side. Open loop: the logged wheel commands replayed into the simulated car, and a grid search
over left/right speed asymmetry, friction scale and delay for the best heading match.

Usage:
    python3 -m hardware.video_track VIDEO --log hardware/logs/run_....csv      # writes VIDEO.track.csv
    python3 -m analysis.compare_video --log hardware/logs/run_....csv --track VIDEO.track.csv \
        --model rl/policy/v9_ppo/car_ppo.zip
"""

import argparse
import csv
import dataclasses
import itertools
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from hardware.protocol import M_PER_S_PER_PWM
from mjsim.car_env import V9, ElegooCarEnv
from rl.train_car import load_config, load_policy, make_actor

OUT = Path(__file__).parent / "out"


def _smooth(x, n=9):
    return np.convolve(np.nan_to_num(x), np.ones(n) / n, mode="same")


def weave(t, offset, heading, lo, hi):
    s = (t > lo) & (t < hi)
    o = _smooth(offset)[s]
    crossings = np.sum(np.diff(np.sign(o - o.mean())) != 0)
    return np.nanstd(offset[s]), np.nanstd(heading[s]), 2 * (hi - lo) / max(crossings, 1)


def load_track(path):
    rows = list(csv.DictReader(open(path)))
    def col(k):
        return np.array([float(r[k]) if r[k] else np.nan for r in rows])
    return col("t_log"), col("offset_mm"), col("angle_deg"), col("corner_fwd_mm")


def closed_loop(model_path, episodes=10):
    cfg = dataclasses.replace(load_config(model_path), scenarios=False)
    model, recurrent = load_policy(model_path)
    out = []
    for randomize, n in ((False, 1), (True, episodes)):
        env = ElegooCarEnv(track="L", randomize=randomize, max_steps=600, config=cfg)
        for ep in range(n):
            obs, _ = env.reset(seed=11 + ep)
            act = make_actor(model, recurrent)
            times, offsets, headings, xs = [], [], [], []
            for k in range(440):
                obs, _, term, _, info = env.step(act(obs))
                _, _, yaw = env._pose()
                sx, sy = env._sensor_xy()[1]
                _, _, _, h = env.track.project(sx, sy, s_hint=env._s_prev)
                times.append((k + 1) * 0.05)
                offsets.append(info["cte"] * 1000)
                xs.append(sx)
                headings.append(np.degrees((yaw - h + np.pi) % (2 * np.pi) - np.pi))
                if term:
                    break
            times, offsets, headings, xs = map(np.array, (times, offsets, headings, xs))
            i_c = int(np.argmax(xs >= 1.5)) if (xs >= 1.5).any() else len(xs) - 1
            overshoot = (xs[i_c:i_c + 80].max() - 1.5) * 1000 if (xs >= 1.5).any() else np.nan
            out.append(("nominal" if not randomize else "random", times - times[i_c], offsets, headings, overshoot,
                        weave(times, offsets, headings, 0.5, times[i_c] - 0.5)))
    return out


def replay(log_path, t_track, heading, t_end=8.0):
    """Heading of the simulated car driven open loop by the logged commands, for each parameter set."""
    rows = list(csv.DictReader(open(log_path)))
    tl = np.array([float(r["t_s"]) for r in rows])
    pwm = np.array([[float(r["vl"]), float(r["vr"])] for r in rows]) / M_PER_S_PER_PWM
    m = (t_track >= 0) & (t_track <= t_end) & ~np.isnan(heading)
    real = _smooth(heading[m] - heading[m][0], 5)
    cfg = dataclasses.replace(V9, scenarios=False, min_pwm=0.0, shield=False)   # commands already compensated
    env = ElegooCarEnv(track="L", randomize=False, max_steps=10**5, config=cfg)
    env.reset(seed=0)
    mu0 = env.v.mu
    n = int(np.searchsorted(tl, t_end))
    results = []
    for asym, mu_s, delay in itertools.product((1.0, 1.1, 1.2, 1.3), (0.5, 0.7, 1.0), (1, 2, 3, 4)):
        env.reset(seed=0, options={"variant": {"delay": delay, "mu": mu0 * mu_s}})
        y0, ys = env._pose()[2], []
        for k in range(n):
            env.step(pwm[k] * np.array([asym, 1.0]) / cfg.pwm_scale)
            ys.append(-np.degrees(env._pose()[2] - y0))
        sim = np.interp(t_track[m], tl[:n], np.array(ys))
        results.append((float(np.sqrt(np.mean((sim - real) ** 2))), asym, mu_s, delay, sim))
    return t_track[m], real, sorted(results, key=lambda r: r[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True)
    ap.add_argument("--track", required=True, help="CSV written by hardware.video_track")
    ap.add_argument("--model", required=True)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    t, off, head, corner = load_track(args.track)
    seen = ~np.isnan(corner) & (t > 0)
    t_corner = t[seen][np.argmax(corner[seen] <= 0)]
    first_pass = seen & (t < t_corner + 4.0)   # after ~4 s the car has rotated; 'corner behind the sensors' is no longer overshoot
    real = weave(t, off, head, 0.5, t_corner - 0.5)
    print(f"{'':16s} {'offset std':>10s} {'heading std':>11s} {'period':>7s} {'overshoot':>10s}")
    print(f"{'real (video)':16s} {real[0]:8.1f}mm {real[1]:10.1f}° {real[2]:6.2f}s {-np.nanmin(corner[first_pass]):8.0f}mm")
    sims = closed_loop(args.model)
    for kind in ("nominal", "random"):
        r = [s for s in sims if s[0] == kind]
        st = np.nanmean([s[5] for s in r], axis=0)
        print(f"{'sim ' + kind:16s} {st[0]:8.1f}mm {st[1]:10.1f}° {st[2]:6.2f}s {np.nanmean([s[4] for s in r]):8.0f}mm")

    fig, ax = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    for kind, times, offsets, headings, *_ in sims:
        style = dict(color="tab:orange", lw=1.6) if kind == "nominal" else dict(color="0.75", lw=0.7)
        ax[0].plot(times, offsets, **style)
        ax[1].plot(times, headings, **style)
    ax[0].plot(t - t_corner, off, color="tab:blue", lw=1.4)
    ax[1].plot(t - t_corner, head, color="tab:blue", lw=1.4)
    ax[0].set_ylabel("offset at sensors (mm)")
    ax[0].set_ylim(-80, 80)
    ax[1].set_ylabel("heading vs tape (deg)")
    ax[1].set_xlabel("time relative to reaching the corner (s)")
    for a in ax:
        a.axvline(0, color="k", lw=0.8)
        a.grid(alpha=0.3)
    ax[0].set_xlim(-8, 10)
    fig.tight_layout()
    fig.savefig(OUT / "sim_vs_real.png", dpi=110)

    tr, hr, fits = replay(args.log, t, head)
    base = next(f for f in fits if f[1] == 1.0 and f[2] == 1.0 and f[3] == 3)
    print(f"\nreplay heading RMSE: current model {base[0]:.1f}°, best {fits[0][0]:.1f}° with left/right speed "
          f"×{fits[0][1]}, friction ×{fits[0][2]}, delay {fits[0][3] * 50} ms")
    fig, ax = plt.subplots(figsize=(12, 3.5))
    ax.plot(tr, hr, color="tab:blue", lw=1.5, label="real (video)")
    ax.plot(tr, base[4], lw=1, label="sim, current model")
    ax.plot(tr, fits[0][4], lw=1, label="sim, best fit")
    ax.set_xlabel("run time (s)")
    ax.set_ylabel("heading change (deg)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "replay.png", dpi=110)
    print(f"saved {OUT / 'sim_vs_real.png'} and {OUT / 'replay.png'}")


if __name__ == "__main__":
    main()
