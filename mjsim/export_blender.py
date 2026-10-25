"""
Record a sim run for Blender: static geometry once, then every body's pose per frame,
plus a MuJoCo-rendered preview MP4 (needs ffmpeg).

Usage:
    python3 -m mjsim.export_blender --controller rl --track figure8 --seconds 30
    blender --background --python mjsim/blender_scene.py -- runs/figure8_rl.json runs/figure8_blender.mp4
"""

import argparse
import json
import subprocess
from pathlib import Path

import mujoco
import numpy as np

from mjsim.baseline import pd_policy
from mjsim.car_env import DT, TEST_TRACKS, ElegooCarEnv
from rl.train_car import PD_TUNE, load_config, load_policy, make_actor

GEOM_TYPES = {mujoco.mjtGeom.mjGEOM_PLANE: "plane", mujoco.mjtGeom.mjGEOM_BOX: "box",
              mujoco.mjtGeom.mjGEOM_CYLINDER: "cylinder", mujoco.mjtGeom.mjGEOM_CAPSULE: "capsule",
              mujoco.mjtGeom.mjGEOM_SPHERE: "sphere"}


def static_scene(env):
    m = env.model
    geoms = []
    for g in range(m.ngeom):
        kind = GEOM_TYPES.get(m.geom_type[g])
        if kind is None or m.geom_rgba[g, 3] == 0:
            continue
        geoms.append({"name": m.geom(g).name, "body": int(m.geom_bodyid[g]), "type": kind,
                      "size": m.geom_size[g].tolist(), "pos": m.geom_pos[g].tolist(),
                      "quat": m.geom_quat[g].tolist(), "rgba": m.geom_rgba[g].tolist()})
    t = env.track
    tape = [np.c_[t.a[i], t.b[i]].T.tolist() for i in range(len(t.a)) if t.visible[i]]
    tape += [[p.tolist(), q.tolist()] for p, q in zip(t.extra_a, t.extra_b)]
    return {"geoms": geoms, "bodies": [m.body(b).name for b in range(m.nbody)],
            "tape_segments": tape, "tape_width": 2 * env.v.tape_half, "fps": int(round(1 / DT))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--controller", choices=["pd", "rl"], default="rl")
    ap.add_argument("--model", default="rl/policy/v8_ppo/car_ppo.zip")
    ap.add_argument("--track", choices=["L", "random", *TEST_TRACKS], default="figure8")
    ap.add_argument("--seconds", type=float, default=30)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--no-preview", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.model)
    env = ElegooCarEnv(track=args.track, randomize=True, max_steps=10**6, config=cfg)
    obs, _ = env.reset(seed=args.seed)
    if args.controller == "rl":
        policy = make_actor(*load_policy(args.model))
    else:
        policy = pd_policy(env, **PD_TUNE)

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    stem = f"{args.track}_{args.controller}"
    scene, frames, preview = static_scene(env), [], None
    if not args.no_preview:
        preview = subprocess.Popen(
            ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
             "-s", "640x480", "-r", str(scene["fps"]), "-i", "-", "-pix_fmt", "yuv420p",
             str(out / f"{stem}_preview.mp4")], stdin=subprocess.PIPE)
    for _ in range(int(args.seconds / DT)):
        obs, _, term, trunc, info = env.step(policy(obs))
        frames.append({"xpos": env.data.xpos.round(5).tolist(), "xquat": env.data.xquat.round(5).tolist(),
                       "ir": obs[env.ir_index:env.ir_index + 3].round(3).tolist()})
        if preview:
            preview.stdin.write(env.render().tobytes())
        if term:
            break
    if preview:
        preview.stdin.close()
        preview.wait()
    scene["frames"] = frames
    scene["outcome"] = {k: (bool(v) if isinstance(v, (bool, np.bool_)) else float(v)) for k, v in info.items()}
    (out / f"{stem}.json").write_text(json.dumps(scene))
    print(f"{len(frames)} frames, outcome {scene['outcome']}")
    print(f"saved {out / (stem + '.json')}" + ("" if args.no_preview else f" and {out / (stem + '_preview.mp4')}"))


if __name__ == "__main__":
    main()
