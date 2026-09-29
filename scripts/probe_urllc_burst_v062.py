from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sb3_contrib import MaskablePPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from art_ran_commag.env import COMMAGPolicyEnv
from art_ran_commag.twin import COMMAGResponseModel


def js(x):
    if isinstance(x, dict):
        return {str(k): js(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [js(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (np.floating,)):
        return float(x)
    if isinstance(x, (np.bool_,)):
        return bool(x)
    if x is None or isinstance(x, (str, int, float, bool)):
        return x
    return repr(x)


def make_env(df, twin, actions, cal, scenario, horizon, seed):
    return COMMAGPolicyEnv(
        df, twin, actions, cal,
        scenario=scenario, horizon=horizon, seed=seed
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepared", default="prepared_v062")
    ap.add_argument("--run-dir", default="runs/v062_seed4")
    ap.add_argument("--model", default="maskable_ppo_best.zip")
    ap.add_argument("--vecnorm", default="vecnormalize.pkl")
    ap.add_argument("--scenario", default="urllc_burst")
    ap.add_argument("--split", default="val")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--steps", type=int, default=3)
    args = ap.parse_args()

    prep = Path(args.prepared)
    run = Path(args.run_dir)

    df = pd.read_parquet(prep / "commag_wide.parquet")
    data = df[df["split"] == args.split].reset_index(drop=True)
    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)
    cal = json.loads((prep / "calibration.json").read_text())
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = 1

    env = make_env(data, twin, actions, cal, args.scenario, args.horizon, args.seed)
    obs, reset_info = env.reset(seed=args.seed)

    # Separate env used only to obtain VecNormalize statistics.
    vn_env = make_env(data, twin, actions, cal, args.scenario, args.horizon, args.seed)
    vn = VecNormalize.load(str(run / args.vecnorm), DummyVecEnv([lambda: vn_env]))
    vn.training = False
    vn.norm_reward = False

    model = MaskablePPO.load(run / args.model, device="cpu")

    report = {
        "scenario": args.scenario,
        "seed": args.seed,
        "calibration": cal,
        "action_catalog": actions.tolist(),
        "reset_info": js(reset_info),
        "steps": [],
    }

    for t in range(args.steps):
        obs = np.asarray(obs, dtype=float).reshape(-1)
        mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
        obs_n = vn.normalize_obs(obs[None, :])[0]
        chosen, _ = model.predict(obs_n, deterministic=True, action_masks=mask)
        chosen = int(np.asarray(chosen).reshape(-1)[0])

        step_rec = {
            "t": t,
            "policy_tail_7": obs[-7:].tolist() if len(obs) >= 7 else [],
            "mask": mask.astype(int).tolist(),
            "valid_actions": np.flatnonzero(mask).tolist(),
            "ppo_chosen_action": chosen,
            "counterfactuals": [],
        }

        print("\n" + "=" * 90)
        print(f"t={t}  policy_tail_7={np.round(obs[-7:], 4).tolist()}")
        print(f"mask={mask.astype(int).tolist()} valid={np.flatnonzero(mask).tolist()} chosen={chosen}")

        for a in range(len(actions)):
            reward_cf, info_cf = env.counterfactual(int(a))
            rec = {
                "action_idx": a,
                "allocation": actions[a].tolist(),
                "mask_valid": bool(mask[a]),
                "reward": float(reward_cf),
                "info": js(info_cf),
            }
            step_rec["counterfactuals"].append(rec)
            print(f"\nACTION {a} allocation={actions[a].tolist()} mask_valid={bool(mask[a])}")
            print(f"reward={float(reward_cf):.6f}")
            print(json.dumps(js(info_cf), indent=2, sort_keys=True))

        report["steps"].append(step_rec)

        obs, reward, term, trunc, info = env.step(chosen)
        if term or trunc:
            break

    out = run / f"probe_{args.scenario}_counterfactual_info.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print("\n" + "=" * 90)
    print(f"Saved raw probe to: {out}")


if __name__ == "__main__":
    main()
