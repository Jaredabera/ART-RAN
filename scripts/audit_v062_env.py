from __future__ import annotations
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from art_ran_commag.env import COMMAGPolicyEnv
from art_ran_commag.twin import COMMAGResponseModel

SCENARIOS = ["nominal", "urllc_burst", "embb_surge", "channel_shift", "mixed", "mobility_shift"]

def main():
    prep = Path("prepared_v062")
    df = pd.read_parquet(prep / "commag_wide.parquet")
    val = df[df["split"] == "val"].reset_index(drop=True)
    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)
    cal = json.loads((prep / "calibration.json").read_text())
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = 1

    print("ACTION CATALOG")
    print(pd.DataFrame(actions, columns=["s0", "s1", "s2"]).to_string(index=False))
    print("\nSCENARIO COUNTS (VAL)")
    print(val["scenario"].value_counts().to_string())

    print("\nENVIRONMENT AUDIT")
    for scenario in SCENARIOS:
        starts = []
        mask_sizes = []
        reward_spreads = []
        for seed in range(1000, 1010):
            env = COMMAGPolicyEnv(
                val, twin, actions, cal,
                scenario=scenario, horizon=40, seed=seed
            )
            obs, _ = env.reset(seed=seed)
            starts.append(str(env._row()["scenario"]))

            for _ in range(10):
                mask = env.action_masks()
                valid = np.flatnonzero(mask)
                mask_sizes.append(len(valid))

                # Only a few cheap oracle checks to diagnose action degeneracy.
                rs = [env.counterfactual(int(a))[0] for a in valid]
                reward_spreads.append(float(max(rs) - min(rs)) if rs else np.nan)

                # Step using the first valid action; this is only an audit.
                a = int(valid[0])
                _, _, term, trunc, _ = env.step(a)
                if term or trunc:
                    break

        c = Counter(starts)
        ms = np.asarray(mask_sizes, dtype=float)
        sp = np.asarray(reward_spreads, dtype=float)
        print(
            f"{scenario:15s} "
            f"start_labels={dict(c)}  "
            f"valid_actions mean/min/max={ms.mean():.2f}/{int(ms.min())}/{int(ms.max())}  "
            f"single_action={100*np.mean(ms==1):.1f}%  "
            f"reward_spread_mean={np.nanmean(sp):.6f}"
        )

if __name__ == "__main__":
    main()
