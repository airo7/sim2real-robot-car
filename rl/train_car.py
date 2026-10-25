"""
Train and evaluate PPO line-following policies in the sim-to-real env (mjsim.car_env).
Each run writes env_config.json next to its policies so evaluation and the real-car
controller rebuild the exact observation layout.

Usage:
    python3 -m rl.train_car --config v4 --steps 10000000 --out rl/policy/v4
    python3 -m rl.train_car --config v5 --algo lstm --steps 20000000 --out rl/policy/v6_lstm
    python3 -m rl.train_car --suite rl/policy/v8_ppo/car_ppo.zip
"""

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import get_linear_fn
from stable_baselines3.common.vec_env import SubprocVecEnv, VecNormalize

from hardware.line_follower import UltrasonicGuard
from mjsim.baseline import evaluate, pd_policy
from mjsim.car_env import TEST_TRACKS, US_MAX, V4, V5, V7, V8, V9, ElegooCarEnv, EnvConfig
from rl.privileged import PrivilegedCriticPolicy, RecurrentPrivilegedCriticPolicy

CONFIGS = {"v3": EnvConfig(), "v4": V4, "v5": V5, "v7": V7, "v8": V8, "v9": V9}
PD_TUNE = dict(kp=0.15, kd=0.08, lost_gain=1.3, slowdown=0.5)


def save_config(directory, cfg):
    Path(directory).mkdir(parents=True, exist_ok=True)
    (Path(directory) / "env_config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))


def _sidecar(model_path, name):
    for d in (Path(model_path).parent, Path(model_path).parent.parent):
        if (d / name).exists():
            return json.loads((d / name).read_text())
    return None


def load_config(model_path):
    """env_config.json next to the policy or one level up; v1–v3 policies predate it."""
    return EnvConfig(**(_sidecar(model_path, "env_config.json") or {}))


def load_policy(model_path):
    """→ (model, recurrent). algo.json records which algorithm trained it (v1–v5: plain PPO)."""
    algo = (_sidecar(model_path, "algo.json") or {}).get("algo", "ppo")
    if algo == "lstm":
        from sb3_contrib import RecurrentPPO
        return RecurrentPPO.load(model_path, device="cpu"), True
    return PPO.load(model_path, device="cpu"), False


def make_actor(model, recurrent):
    """Fresh per-episode act(obs) → action; carries the LSTM state for recurrent policies."""
    if not recurrent:
        return lambda obs: model.predict(obs, deterministic=True)[0]
    state = {"h": None, "start": True}

    def act(obs):
        a, state["h"] = model.predict(obs, state=state["h"], episode_start=np.array([state["start"]]),
                                      deterministic=True)
        state["start"] = False
        return a

    return act


class Curriculum(BaseCallback):
    """Scenario difficulty from `start` to 1.0 over the first `ramp` fraction of training."""

    def __init__(self, total, start=0.2, ramp=0.6):
        super().__init__()
        self.total, self.start, self.ramp = total, start, ramp

    def _on_step(self):
        frac = min(1.0, self.num_timesteps / (self.ramp * self.total))
        level = self.start + (1.0 - self.start) * frac
        self.training_env.env_method("set_level", level)
        self.logger.record("curriculum/level", level)
        return True


# for stability: v5's eval reward swung 5–49 between evals without improving
TUNED = dict(n_steps=2048, batch_size=4096, n_epochs=5, gamma=0.995, gae_lambda=0.95,
             learning_rate=get_linear_fn(3e-4, 3e-5, 1.0), clip_range=0.15, ent_coef=0.001,
             target_kl=0.03)


def build_model(algo, env, cfg, out):
    policy_dim = env.get_attr("policy_dim")[0]
    common = dict(device="cpu", verbose=0, tensorboard_log=str(out / "tb"))
    if algo == "ppo":
        policy = PrivilegedCriticPolicy if cfg.privileged else "MlpPolicy"
        extra = dict(policy_dim=policy_dim) if cfg.privileged else {}
        return PPO(policy, env, n_steps=1024, batch_size=2048, n_epochs=10, gamma=0.99,
                   gae_lambda=0.95, learning_rate=3e-4, clip_range=0.2, ent_coef=0.0,
                   policy_kwargs=dict(net_arch=[128, 128], **extra), **common)
    if algo == "ppo_tuned":
        return PPO(PrivilegedCriticPolicy, env, **TUNED, **common,
                   policy_kwargs=dict(net_arch=[256, 256], log_std_init=-0.5, policy_dim=policy_dim))
    from sb3_contrib import RecurrentPPO
    lstm = dict(TUNED, n_steps=512, batch_size=1536)        # n_steps = sequence chunk per env
    return RecurrentPPO(RecurrentPrivilegedCriticPolicy, env, **lstm, **common,
                        policy_kwargs=dict(net_arch=[128], lstm_hidden_size=128, log_std_init=-0.5,
                                           policy_dim=policy_dim))


def train(steps, n_envs, config_name, out, algo, ckpt_every=1_000_000):
    cfg = CONFIGS[config_name]
    if algo != "ppo" and not cfg.privileged:
        raise SystemExit(f"--algo {algo} needs a privileged config (v4/v5)")
    out = Path(out)
    for d in (out, out / "car_best"):
        save_config(d, cfg)
        (d / "algo.json").write_text(json.dumps({"algo": algo}))
    start_level = 0.2 if cfg.scenarios else 1.0

    def make(level):
        return lambda: ElegooCarEnv(track="random", randomize=True, config=cfg, level=level)

    env = make_vec_env(make(start_level), n_envs=n_envs, vec_env_cls=SubprocVecEnv, seed=0)
    eval_env = make_vec_env(make(1.0), n_envs=4, vec_env_cls=SubprocVecEnv, seed=10_000)
    if algo != "ppo":
        # rewards only: raw observations mean the real-car controller needs no saved stats
        env = VecNormalize(env, norm_obs=False, norm_reward=True, gamma=TUNED["gamma"])
        eval_env = VecNormalize(eval_env, norm_obs=False, norm_reward=False, training=False)
    model = build_model(algo, env, cfg, out)
    callbacks = [
        CheckpointCallback(ckpt_every // n_envs, str(out / "car_ckpt"), name_prefix="car_ppo"),
        EvalCallback(eval_env, n_eval_episodes=8, eval_freq=250_000 // n_envs,
                     best_model_save_path=str(out / "car_best"), verbose=1),
    ]
    if cfg.scenarios:
        callbacks.append(Curriculum(steps))
    model.learn(total_timesteps=steps, callback=callbacks, progress_bar=False)
    model.save(out / "car_ppo")
    print(f"saved {out / 'car_ppo.zip'}")


def _guarded(model, recurrent, env):
    """RL with the same ultrasonic stop the real-car controller applies (make_car_rl_controller)."""
    guard = UltrasonicGuard(stop_mm=120, slow_mm=350)
    actor = make_actor(model, recurrent)
    i = env.ir_index + 3

    def act(obs):
        u = actor(obs)
        return u * guard.scale(obs[i] * US_MAX * 1000) if env.cfg.ultrasonic else u

    return act


def _controllers(model, recurrent, cfg):
    pd = ("PD", lambda env: pd_policy(env, **PD_TUNE))
    rl = ("RL", lambda env: make_actor(model, recurrent))
    if cfg.shield:                      # the env already applies the guard, as on the car
        return (pd, rl)
    return (pd, rl, ("RL+", lambda env: _guarded(model, recurrent, env)))


def _row(track, name, r, closed):
    done = "loop" if closed else f"{r['finished']:.0%}"
    return (f"{track:9s} {name:3s} {done:>8s} {r['lost']:6.0%} {r['collided']:6.0%} "
            f"{r['progress_m']:7.2f} m {r['progress_m'] / r['time_s']:6.3f} m/s")


def suite(path, episodes, tracks=None):
    """Regression suite: the real L track, random tracks, and held-out courses."""
    cfg = load_config(path)
    model, recurrent = load_policy(path)
    tracks = tracks or ["L", "random", *TEST_TRACKS]
    print(f"policy {path}  config {dataclasses.asdict(cfg)}")
    print(f"{'course':9s} {'':3s} {'finished':>8s} {'lost':>6s} {'crash':>6s} {'distance':>9s} {'speed':>10s}")
    results = {}
    for track in tracks:
        env = ElegooCarEnv(track=track, randomize=True, max_steps=2400, config=cfg)
        for name, make in _controllers(model, recurrent, cfg):
            r = evaluate(lambda: make(env), env, episodes, seed=90_000)
            results[f"{track}/{name}"] = {k: float(v) for k, v in r.items()}
            print(_row(track, name, r, env.track.closed))
    out = Path(path).with_suffix(".suite.json")
    out.write_text(json.dumps(results, indent=2))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=3_000_000)
    ap.add_argument("--envs", type=int, default=12)
    ap.add_argument("--config", choices=list(CONFIGS), default="v3")
    ap.add_argument("--algo", choices=["ppo", "ppo_tuned", "lstm"], default="ppo")
    ap.add_argument("--out", default=str(Path(__file__).parent / "policy"))
    ap.add_argument("--suite", type=str, help="run the regression suite on a saved policy")
    ap.add_argument("--tracks", nargs="+", help="limit --suite to these tracks")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--ckpt-every", type=int, default=1_000_000, help="checkpoint interval (steps)")
    args = ap.parse_args()
    if args.suite:
        suite(args.suite, args.episodes, args.tracks)
    else:
        train(args.steps, args.envs, args.config, args.out, args.algo, args.ckpt_every)
