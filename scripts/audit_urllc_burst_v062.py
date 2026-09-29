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


POLICY_NAMES = ["w_e", "w_m", "w_u", "m_e", "m_m", "m_u", "delta_max"]


def _arr3(x, default=np.nan):
    """Return first 3 values of x as floats, padding with NaN."""
    if x is None:
        return [default, default, default]
    a = np.asarray(x).reshape(-1)
    out = [default, default, default]
    for i in range(min(3, len(a))):
        try:
            out[i] = float(a[i])
        except Exception:
            pass
    return out


def _boolish(x):
    try:
        return bool(x)
    except Exception:
        return False


def make_env(df, twin, actions, cal, scenario, horizon, seed):
    return COMMAGPolicyEnv(
        df, twin, actions, cal,
        scenario=scenario, horizon=horizon, seed=seed
    )


def load_vecnorm(path, template_env):
    dummy = DummyVecEnv([lambda: template_env])
    vn = VecNormalize.load(str(path), dummy)
    vn.training = False
    vn.norm_reward = False
    return vn


def eval_cf(env, action_idx):
    """Evaluate one supported action without changing the environment."""
    try:
        reward, info = env.counterfactual(int(action_idx))
        return float(reward), dict(info), None
    except Exception as e:
        return np.nan, {}, repr(e)


def main():
    ap = argparse.ArgumentParser(
        description="Diagnose why URLLC-burst has zero QoS-feasible rate."
    )
    ap.add_argument("--prepared", default="prepared_v062")
    ap.add_argument("--run-dir", default="runs/v062_seed4")
    ap.add_argument("--model", default="maskable_ppo_best.zip",
                    help="model file inside --run-dir")
    ap.add_argument("--vecnorm", default="vecnormalize.pkl",
                    help="VecNormalize file inside --run-dir")
    ap.add_argument("--scenario", default="urllc_burst")
    ap.add_argument("--episodes", type=int, default=10)
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--split", default="val", choices=["train", "val", "test"])
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    prep = Path(args.prepared)
    run = Path(args.run_dir)
    out = Path(args.out_dir) if args.out_dir else run / f"audit_{args.scenario}"
    out.mkdir(parents=True, exist_ok=True)

    print("[1/5] Loading prepared COMMAG data...")
    df = pd.read_parquet(prep / "commag_wide.parquet")
    data = df[df["split"] == args.split].reset_index(drop=True)
    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)
    cal = json.loads((prep / "calibration.json").read_text())
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = 1

    print(f"split={args.split}, rows={len(data):,}, actions={len(actions)}")
    print("\nACTION CATALOG")
    print(pd.DataFrame(actions, columns=["eMBB", "mMTC", "URLLC"]).to_string(index=True))
    print("\nCALIBRATION")
    print(json.dumps(cal, indent=2)[:5000])

    template = make_env(
        data, twin, actions, cal, args.scenario, args.horizon, args.seed0
    )
    vecnorm = load_vecnorm(run / args.vecnorm, template)
    model = MaskablePPO.load(run / args.model, device="cpu")

    candidate_rows = []
    step_rows = []

    print(f"\n[2/5] Auditing {args.scenario} across {args.episodes} episodes...")

    for ep in range(args.episodes):
        seed = args.seed0 + ep
        env = make_env(
            data, twin, actions, cal, args.scenario, args.horizon, seed
        )
        obs, _ = env.reset(seed=seed)
        done = False
        t = 0

        while not done:
            obs = np.asarray(obs, dtype=float).reshape(-1)
            if len(obs) < 7:
                raise RuntimeError(
                    f"Observation dimension {len(obs)} is too small to contain 7-D rApp policy."
                )
            p = obs[-7:].copy()

            mask = np.asarray(env.action_masks(), dtype=bool).reshape(-1)
            valid = np.flatnonzero(mask)
            if len(mask) != len(actions):
                raise RuntimeError(
                    f"Mask length {len(mask)} != action catalog length {len(actions)}"
                )

            # Evaluate every action in the supported COMMAG catalog. This is the
            # key test: does QoS feasibility exist before the rApp mask is applied?
            all_cf = {}
            any_cf_error = False
            for a in range(len(actions)):
                reward_cf, info_cf, err = eval_cf(env, a)
                if err is not None:
                    any_cf_error = True

                viol = _arr3(info_cf.get("sla_violations"))
                all_sla = _boolish(
                    info_cf.get("all_sla_met", info_cf.get("all_qos_met", False))
                )

                row = {
                    "episode": ep,
                    "seed": seed,
                    "t": t,
                    "action_idx": a,
                    "a_embb": int(actions[a, 0]),
                    "a_mmtc": int(actions[a, 1]),
                    "a_urllc": int(actions[a, 2]),
                    "mask_valid": int(mask[a]),
                    "reward_cf": reward_cf,
                    "all_qos_cf": int(all_sla),
                    "viol_embb": viol[0],
                    "viol_mmtc": viol[1],
                    "viol_urllc": viol[2],
                    "cf_error": err or "",
                }

                # Preserve useful scalar/short-array diagnostics without relying
                # on a specific env version.
                for k, v in info_cf.items():
                    if k in {"sla_violations", "all_sla_met", "all_qos_met"}:
                        continue
                    if np.isscalar(v) and isinstance(v, (int, float, np.integer, np.floating, bool)):
                        row[f"info_{k}"] = float(v)
                    else:
                        try:
                            av = np.asarray(v).reshape(-1)
                            if 1 <= len(av) <= 3 and np.issubdtype(av.dtype, np.number):
                                for j, vv in enumerate(av):
                                    row[f"info_{k}_{j}"] = float(vv)
                        except Exception:
                            pass

                candidate_rows.append(row)
                all_cf[a] = row

            catalog_feasible = any(r["all_qos_cf"] == 1 for r in all_cf.values())
            masked_feasible = any(
                all_cf[a]["all_qos_cf"] == 1 for a in valid
            ) if len(valid) else False

            # Per-slice attainability across all catalog actions and masked actions.
            def any_nonviol(action_ids, key):
                vals = []
                for a in action_ids:
                    v = all_cf[a][key]
                    if np.isfinite(v):
                        vals.append(v <= 0.0)
                return bool(any(vals)) if vals else False

            catalog_slice = [
                any_nonviol(range(len(actions)), "viol_embb"),
                any_nonviol(range(len(actions)), "viol_mmtc"),
                any_nonviol(range(len(actions)), "viol_urllc"),
            ]
            masked_slice = [
                any_nonviol(valid, "viol_embb"),
                any_nonviol(valid, "viol_mmtc"),
                any_nonviol(valid, "viol_urllc"),
            ]

            # PPO action under the same normalization used for training.
            obs_n = vecnorm.normalize_obs(obs[None, :])[0]
            chosen, _ = model.predict(
                obs_n, deterministic=True, action_masks=mask
            )
            chosen = int(np.asarray(chosen).reshape(-1)[0])

            chosen_cf = all_cf.get(chosen, {})
            finite_rewards = np.asarray(
                [all_cf[a]["reward_cf"] for a in valid
                 if np.isfinite(all_cf[a]["reward_cf"])],
                dtype=float,
            )
            best_masked = float(np.max(finite_rewards)) if len(finite_rewards) else np.nan
            chosen_reward_cf = float(chosen_cf.get("reward_cf", np.nan))
            regret_cf = (
                best_masked - chosen_reward_cf
                if np.isfinite(best_masked) and np.isfinite(chosen_reward_cf)
                else np.nan
            )

            sr = {
                "episode": ep,
                "seed": seed,
                "t": t,
                "mask_size": int(len(valid)),
                "catalog_feasible": int(catalog_feasible),
                "masked_feasible": int(masked_feasible),
                "catalog_embb_attainable": int(catalog_slice[0]),
                "catalog_mmtc_attainable": int(catalog_slice[1]),
                "catalog_urllc_attainable": int(catalog_slice[2]),
                "masked_embb_attainable": int(masked_slice[0]),
                "masked_mmtc_attainable": int(masked_slice[1]),
                "masked_urllc_attainable": int(masked_slice[2]),
                "chosen_action": chosen,
                "chosen_mask_valid": int(mask[chosen]) if 0 <= chosen < len(mask) else 0,
                "chosen_all_qos_cf": int(chosen_cf.get("all_qos_cf", 0)),
                "chosen_reward_cf": chosen_reward_cf,
                "best_masked_reward_cf": best_masked,
                "regret_cf": regret_cf,
                "counterfactual_error": int(any_cf_error),
            }
            for name, val in zip(POLICY_NAMES, p):
                sr[name] = float(val)
            step_rows.append(sr)

            obs, reward, term, trunc, info = env.step(chosen)
            done = bool(term or trunc)
            t += 1

    cand = pd.DataFrame(candidate_rows)
    steps = pd.DataFrame(step_rows)
    cand.to_csv(out / "candidate_actions.csv", index=False)
    steps.to_csv(out / "step_audit.csv", index=False)

    print("\n[3/5] Core diagnosis")
    summary = {
        "steps": len(steps),
        "mask_size_mean": steps["mask_size"].mean(),
        "mask_size_min": steps["mask_size"].min(),
        "mask_size_max": steps["mask_size"].max(),
        "single_action_pct": 100.0 * (steps["mask_size"] == 1).mean(),
        "catalog_feasible_rate": steps["catalog_feasible"].mean(),
        "masked_feasible_rate": steps["masked_feasible"].mean(),
        "ppo_all_qos_rate_cf": steps["chosen_all_qos_cf"].mean(),
        "mean_cf_regret": steps["regret_cf"].mean(),
        "cf_error_rate": steps["counterfactual_error"].mean(),
    }
    for k, v in summary.items():
        print(f"{k:28s}: {v:.6f}" if isinstance(v, (float, np.floating)) else f"{k:28s}: {v}")

    print("\nPer-slice attainability")
    for scope in ("catalog", "masked"):
        for s in ("embb", "mmtc", "urllc"):
            col = f"{scope}_{s}_attainable"
            print(f"{col:28s}: {steps[col].mean():.6f}")

    print("\nPolicy vector statistics")
    print(
        steps[POLICY_NAMES]
        .agg(["mean", "std", "min", "max"])
        .round(4)
        .to_string()
    )

    print("\n[4/5] Action-level behavior")
    action_summary = (
        cand.groupby(
            ["action_idx", "a_embb", "a_mmtc", "a_urllc"],
            as_index=False
        )
        .agg(
            mask_rate=("mask_valid", "mean"),
            reward_mean=("reward_cf", "mean"),
            all_qos_rate=("all_qos_cf", "mean"),
            embb_violation=("viol_embb", "mean"),
            mmtc_violation=("viol_mmtc", "mean"),
            urllc_violation=("viol_urllc", "mean"),
        )
        .sort_values(["all_qos_rate", "reward_mean"], ascending=[False, False])
    )
    action_summary.to_csv(out / "action_summary.csv", index=False)
    print(action_summary.round(4).to_string(index=False))

    print("\n[5/5] Interpretation")
    cf = summary["catalog_feasible_rate"]
    mf = summary["masked_feasible_rate"]
    pf = summary["ppo_all_qos_rate_cf"]

    if cf == 0:
        print(
            "DIAGNOSIS A: No action in the full COMMAG action catalog meets all QoS "
            "targets in the audited states. The primary bottleneck is therefore "
            "scenario/action-catalog/QoS-target/response-model feasibility, not PPO."
        )
    elif mf == 0 and cf > 0:
        print(
            "DIAGNOSIS B: QoS-feasible actions exist in the COMMAG catalog, but the "
            "rApp-induced action mask removes them. Inspect policy minima and delta_max."
        )
    elif mf > 0 and pf == 0:
        print(
            "DIAGNOSIS C: QoS-feasible masked actions exist, but the PPO does not select "
            "them. This is a controller-learning/generalization issue."
        )
    else:
        print(
            "DIAGNOSIS D: Feasible actions are available and PPO sometimes selects them. "
            "Use the CSVs to quantify which constraints/actions dominate residual failures."
        )

    print(f"\nSaved:\n  {out / 'step_audit.csv'}\n  {out / 'candidate_actions.csv'}\n  {out / 'action_summary.csv'}")


if __name__ == "__main__":
    main()
