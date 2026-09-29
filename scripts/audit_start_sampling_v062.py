from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from art_ran_commag.env import COMMAGPolicyEnv
from art_ran_commag.twin import COMMAGResponseModel


def qsummary(x: pd.Series):
    x = pd.to_numeric(x, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
    if len(x) == 0:
        return [np.nan] * 5
    return np.quantile(x, [0.01, 0.25, 0.50, 0.75, 0.99]).tolist()


def main():
    ap = argparse.ArgumentParser(
        description="Audit whether COMMAGPolicyEnv reset sampling is biased within a scenario."
    )
    ap.add_argument("--prepared", default="prepared_v062")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenario", default="urllc_burst")
    ap.add_argument("--horizon", type=int, default=60)
    ap.add_argument("--resets", type=int, default=500)
    ap.add_argument("--seed0", type=int, default=1000)
    ap.add_argument("--out", default="runs/v062_seed4/urllc_burst_start_sampling.csv")
    args = ap.parse_args()

    prep = Path(args.prepared)
    df = pd.read_parquet(prep / "commag_wide.parquet")
    data = df[df["split"] == args.split].reset_index(drop=True)
    focus = data[data["scenario"] == args.scenario].copy()

    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)
    cal = json.loads((prep / "calibration.json").read_text())
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = 1

    rows = []
    for k in range(args.resets):
        seed = args.seed0 + k
        env = COMMAGPolicyEnv(
            data, twin, actions, cal,
            scenario=args.scenario,
            horizon=args.horizon,
            seed=seed,
        )
        obs, info = env.reset(seed=seed)
        r = env._row()

        rec = {
            "seed": seed,
            "scenario_row": str(r.get("scenario", "")),
            "trajectory": str(getattr(env, "key", info.get("trajectory", ""))),
        }
        for s in range(3):
            rec[f"buffer_s{s}"] = float(env.buffer[s])
            rec[f"req_s{s}"] = float(r[f"sum_requested_prbs_s{s}"])
            rec[f"tput_obs_s{s}"] = float(r[f"tx_brate downlink [Mbps]_s{s}"])
            rec[f"ratio_obs_s{s}"] = float(r[f"ratio_granted_req_s{s}"])
            rec[f"prb_obs_s{s}"] = int(r[f"slice_prb_s{s}"])
        rows.append(rec)

    starts = pd.DataFrame(rows)

    print("=== RESET LABEL CHECK ===")
    print(starts["scenario_row"].value_counts(dropna=False).to_string())

    metrics = []
    for s in range(3):
        metrics += [
            f"buffer_s{s}", f"req_s{s}", f"tput_obs_s{s}", f"ratio_obs_s{s}"
        ]

    # Construct population frame with matching names.
    pop = pd.DataFrame(index=focus.index)
    for s in range(3):
        pop[f"buffer_s{s}"] = focus[f"dl_buffer [bytes]_s{s}"]
        pop[f"req_s{s}"] = focus[f"sum_requested_prbs_s{s}"]
        pop[f"tput_obs_s{s}"] = focus[f"tx_brate downlink [Mbps]_s{s}"]
        pop[f"ratio_obs_s{s}"] = focus[f"ratio_granted_req_s{s}"]

    comp = []
    for c in metrics:
        pq = qsummary(pop[c])
        sq = qsummary(starts[c])
        comp.append({
            "feature": c,
            "population_q01": pq[0],
            "population_q25": pq[1],
            "population_median": pq[2],
            "population_q75": pq[3],
            "population_q99": pq[4],
            "sampled_q01": sq[0],
            "sampled_q25": sq[1],
            "sampled_median": sq[2],
            "sampled_q75": sq[3],
            "sampled_q99": sq[4],
            "median_ratio_sampled_to_population": (
                sq[2] / pq[2] if np.isfinite(pq[2]) and abs(pq[2]) > 1e-12 else np.nan
            ),
        })
    comp = pd.DataFrame(comp)

    print("\n=== POPULATION VS SAMPLED RESET STATES ===")
    print(comp.round(4).to_string(index=False))

    print("\n=== MOST COMMON SAMPLED TRAJECTORIES ===")
    print(starts["trajectory"].value_counts().head(20).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    starts.to_csv(out, index=False)
    comp.to_csv(out.with_name(out.stem + "_comparison.csv"), index=False)

    print(f"\nSaved:\n  {out}\n  {out.with_name(out.stem + '_comparison.csv')}")


if __name__ == "__main__":
    main()
