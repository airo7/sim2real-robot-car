"""
Evaluate policy versions (and the PD baseline) on identical courses in the current simulator,
plus completion-vs-training-steps from a run's checkpoints. Runs in parallel.

Usage:
    python3 -m analysis.evaluate_policies --policies v8=rl/policy/v8_ppo/car_ppo.zip v9=rl/policy/v9_ppo/car_ppo.zip
    python3 -m analysis.evaluate_policies --checkpoints rl/policy/v8_ppo
"""

import argparse
import dataclasses
import functools
import glob
import json
import os
import re
from multiprocessing import Pool
from pathlib import Path

TRACKS = {"L": 30, "random": 20, "zigzag": 10, "s_curve": 10, "spiral": 10, "figure8": 10, "dashed": 10, "rect": 10}
OUT = Path(__file__).parent / "out"


def _course(args):
    name, path, track, episodes = args
    from mjsim.baseline import evaluate, pd_policy
    from mjsim.car_env import V8, ElegooCarEnv
    from rl.train_car import PD_TUNE, load_config, load_policy, make_actor

    if path is None:
        env = ElegooCarEnv(track=track, randomize=True, max_steps=2400, config=dataclasses.replace(V8, scenarios=False))
        make = functools.partial(pd_policy, env, **PD_TUNE)
    else:
        # older policies are evaluated under the current physics, including the measured dead zone
        cfg = dataclasses.replace(load_config(path), dead_zone=True, scenarios=False)
        env = ElegooCarEnv(track=track, randomize=True, max_steps=2400, config=cfg)
        model, recurrent = load_policy(path)
        make = functools.partial(make_actor, model, recurrent)
    return name, track, {k: float(v) for k, v in evaluate(make, env, episodes, seed=123_000).items()}


def _checkpoint(args):
    path, steps = args
    from mjsim.baseline import evaluate
    from mjsim.car_env import ElegooCarEnv
    from rl.train_car import load_config, load_policy, make_actor

    cfg = dataclasses.replace(load_config(path), scenarios=False)
    model, recurrent = load_policy(path)
    out = {}
    for track in ("L", "random"):
        env = ElegooCarEnv(track=track, randomize=True, max_steps=1800, config=cfg)
        r = evaluate(lambda: make_actor(model, recurrent), env, 10, seed=321_000)
        out[track] = {k: float(v) for k, v in r.items()}
    return steps, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policies", nargs="*", default=[], metavar="NAME=PATH")
    ap.add_argument("--no-pd", action="store_true", help="skip the PD baseline")
    ap.add_argument("--checkpoints", metavar="RUN_DIR", help="evaluate every checkpoint in RUN_DIR/car_ckpt")
    ap.add_argument("--workers", type=int, default=os.cpu_count() - 1)
    args = ap.parse_args()
    OUT.mkdir(exist_ok=True)

    with Pool(args.workers) as pool:
        if args.policies or not args.no_pd:
            entries = [tuple(p.split("=", 1)) for p in args.policies] + ([] if args.no_pd else [("PD", None)])
            jobs = [(n, p, t, e) for n, p in entries for t, e in TRACKS.items()]
            res = pool.map(_course, jobs, chunksize=1)
            (OUT / "policies.json").write_text(json.dumps(res, indent=1))
            print(f"{'':8s}" + "".join(f"{t:>9s}" for t in TRACKS) + "   (fraction of episodes that lost the line)")
            for n, _ in entries:
                row = {t: r for name, t, r in res if name == n}
                print(f"{n:8s}" + "".join(f"{row[t]['lost']:>9.0%}" for t in TRACKS))
        if args.checkpoints:
            ckpts = glob.glob(str(Path(args.checkpoints) / "car_ckpt" / "*_steps.zip"))
            jobs = sorted((p, int(re.search(r"_(\d+)_steps", p).group(1))) for p in ckpts)
            res = sorted(pool.map(_checkpoint, jobs, chunksize=1))
            (OUT / f"checkpoints_{Path(args.checkpoints).name}.json").write_text(json.dumps(res, indent=1))
            for steps, r in res:
                print(f"{steps / 1e6:5.1f}M  L finished {r['L']['finished']:4.0%}   random lost {r['random']['lost']:4.0%}")


if __name__ == "__main__":
    main()
