from __future__ import annotations

# Keep numerical libraries from taking over every CPU core on Windows.
import os
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from art_ran_commag.env import COMMAGPolicyEnv
from art_ran_commag.twin import COMMAGResponseModel

TRAIN_SCENARIOS = ["nominal", "urllc_burst", "embb_surge", "channel_shift", "mixed"]
EVAL_SCENARIOS = TRAIN_SCENARIOS + ["mobility_shift"]


def evaluate_episode(model, vecnorm, val_df, twin, actions, cal, scenario, seed, horizon):
    env = COMMAGPolicyEnv(val_df, twin, actions, cal, scenario=scenario, horizon=horizon, seed=seed)
    obs, _ = env.reset(seed=seed)
    rows = []
    done = False
    while not done:
        mask = env.action_masks()
        valid = np.flatnonzero(mask)

        # Exact oracle over supported actions. Slow by design, but bounded by
        # single-threaded ExtraTrees inference so it cannot monopolize the PC.
        oracle = [env.counterfactual(int(a)) for a in valid]
        rewards = np.asarray([x[0] for x in oracle], dtype=float)
        best = float(rewards.max())
        worst = float(rewards.min())
        feasible = any(x[1]["all_sla_met"] for x in oracle)

        obs_n = vecnorm.normalize_obs(obs[None, :])[0]
        act, _ = model.predict(obs_n, deterministic=True, action_masks=mask)
        obs, r, term, trunc, info = env.step(int(act))
        oracle_score = (r - worst) / (best - worst + 1e-8) if best > worst + 1e-8 else 1.0
        rows.append({
            "reward": r,
            "regret": best - r,
            "oracle_score": np.clip(oracle_score, 0, 1),
            "feasible": float(feasible),
            "feasible_success": float(feasible and info["all_sla_met"]),
            "urllc_violation": float(info["sla_violations"][2]),
            "all_sla": float(info["all_sla_met"]),
            "churn": float(info["action_churn"]),
        })
        done = term or trunc

    d = pd.DataFrame(rows)
    feas = d.feasible.sum()
    return {
        "return": d.reward.sum(),
        "regret": d.regret.mean(),
        "oracle_score": d.oracle_score.mean(),
        "fcsr": d.feasible_success.sum() / feas if feas > 0 else np.nan,
        "feasible_rate": d.feasible.mean(),
        "urllc_violation": d.urllc_violation.mean(),
        "all_sla": d.all_sla.mean(),
        "churn": d.churn.mean(),
    }


class HeldOutCallback(BaseCallback):
    def __init__(self, vecnorm, val_df, twin, actions, cal, every, n_eps, horizon, out_csv):
        super().__init__()
        self.vecnorm = vecnorm
        self.val_df = val_df
        self.twin = twin
        self.actions = actions
        self.cal = cal
        self.every = every
        self.n_eps = n_eps
        self.horizon = horizon
        self.out_csv = Path(out_csv)
        self.last = 0
        self.rows = []

    def _on_step(self):
        if self.num_timesteps - self.last < self.every:
            return True
        self.last = self.num_timesteps
        print(f"\n[eval] held-out evaluation starting at {self.num_timesteps} timesteps...")
        for s in EVAL_SCENARIOS:
            rr = [
                evaluate_episode(
                    self.model, self.vecnorm, self.val_df, self.twin,
                    self.actions, self.cal, s, 1000 + i, self.horizon
                )
                for i in range(self.n_eps)
            ]
            row = {"timesteps": self.num_timesteps, "scenario": s}
            for k in rr[0]:
                x = np.asarray([z[k] for z in rr], dtype=float)
                x = x[np.isfinite(x)]
                row[k + "_mean"] = x.mean() if len(x) else np.nan
                row[k + "_se"] = x.std(ddof=1) / np.sqrt(len(x)) if len(x) > 1 else 0.0
            self.rows.append(row)
            print(f"[eval] finished {s}")

        d = pd.DataFrame(self.rows)
        self.out_csv.parent.mkdir(parents=True, exist_ok=True)
        d.to_csv(self.out_csv, index=False)
        cur = d[d.timesteps == self.num_timesteps]
        print("\n=== HELD-OUT @", self.num_timesteps, "===")
        print(cur[[
            "scenario", "return_mean", "oracle_score_mean", "regret_mean",
            "fcsr_mean", "feasible_rate_mean", "urllc_violation_mean"
        ]].round(4).to_string(index=False))
        return True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prepared", default="prepared_v062")
    p.add_argument("--run-dir", default="runs/v062_safe")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--device", default="cpu")
    p.add_argument("--threads", type=int, default=1,
                   help="CPU threads for Torch and ExtraTrees prediction")
    p.add_argument("--horizon", type=int, default=80,
                   help="training episode horizon")
    p.add_argument("--n-steps", type=int, default=128,
                   help="PPO rollout length per environment")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--skip-eval", action="store_true",
                   help="skip expensive exact-oracle held-out evaluation")
    p.add_argument("--eval-every", type=int, default=2000)
    p.add_argument("--eval-episodes", type=int, default=2)
    p.add_argument("--eval-horizon", type=int, default=60)
    a = p.parse_args()

    a.threads = max(1, int(a.threads))
    torch.set_num_threads(a.threads)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass

    prep = Path(a.prepared)
    run = Path(a.run_dir)
    run.mkdir(parents=True, exist_ok=True)

    print("[setup] loading COMMAG parquet...")
    df = pd.read_parquet(prep / "commag_wide.parquet")
    train = df[df.split == "train"].reset_index(drop=True)
    val = df[df.split == "val"].reset_index(drop=True)
    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)
    cal = json.loads((prep / "calibration.json").read_text())
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")

    # This is the key Windows safety fix. The model was trained with n_jobs=-1,
    # which also makes ExtraTrees.predict use all logical cores.
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = a.threads

    print(f"[setup] torch threads={torch.get_num_threads()}, twin n_jobs={getattr(twin.model, 'n_jobs', None)}")
    print(f"[setup] train rows={len(train):,}, val rows={len(val):,}, actions={len(actions)}")

    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)

    def make(s, z):
        def f():
            return COMMAGPolicyEnv(train, twin, actions, cal, scenario=s, horizon=a.horizon, seed=z)
        return f

    v = DummyVecEnv([make(s, a.seed + i * 101) for i, s in enumerate(TRAIN_SCENARIOS)])
    v = VecNormalize(v, norm_obs=True, norm_reward=True, clip_obs=10, clip_reward=10, gamma=.99)

    cb = None
    if not a.skip_eval:
        cb = HeldOutCallback(
            v, val, twin, actions, cal,
            a.eval_every, a.eval_episodes, a.eval_horizon,
            run / "heldout_learning_curves.csv",
        )

    m = MaskablePPO(
        "MlpPolicy", v,
        learning_rate=1.5e-4,
        n_steps=a.n_steps,
        batch_size=a.batch_size,
        gamma=.99,
        gae_lambda=.95,
        clip_range=.15,
        ent_coef=.003,
        vf_coef=.5,
        max_grad_norm=.5,
        target_kl=.02,
        policy_kwargs=dict(net_arch=dict(pi=[128, 128], vf=[128, 128])),
        seed=a.seed,
        verbose=1,
        device=a.device,
        tensorboard_log=str(run / "tb"),
    )

    print("[train] starting MaskablePPO...")
    m.learn(a.steps, callback=cb, progress_bar=True)
    m.save(run / f"maskable_ppo_{a.steps}")
    v.save(run / "vecnormalize.pkl")
    if cb is not None:
        pd.DataFrame(cb.rows).to_csv(run / "heldout_learning_curves.csv", index=False)
    print(f"[done] outputs saved to {run}")


if __name__ == "__main__":
    main()
