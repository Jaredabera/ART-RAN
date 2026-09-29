from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, r2_score

from art_ran_commag.twin import (
    COMMAGResponseModel,
    make_xy,
    target_columns,
)


SCENARIOS = ["nominal", "urllc_burst", "embb_surge", "channel_shift", "mobility_shift"]


def exact_action(row):
    return np.asarray([row[f"slice_prb_s{s}"] for s in range(3)], dtype=int)


def row_y(row):
    t = np.asarray([row[f"tx_brate downlink [Mbps]_s{s}"] for s in range(3)], dtype=float)
    r = np.asarray([row[f"ratio_granted_req_s{s}"] for s in range(3)], dtype=float)
    return t, r


def scenario_metrics(twin, df):
    rows = []
    names = target_columns()
    for split in ["train", "val", "test"]:
        ds = df[df["split"] == split]
        if ds.empty:
            continue
        for sc in sorted(ds["scenario"].dropna().unique()):
            g = ds[ds["scenario"] == sc]
            if len(g) < 5:
                continue
            X, Y = make_xy(g)
            P = twin.model.predict(X)
            rec = {"split": split, "scenario": sc, "n": len(g)}
            for i, name in enumerate(names):
                short = (
                    name.replace("tx_brate downlink [Mbps]_", "tput_")
                        .replace("ratio_granted_req_", "ratio_")
                )
                rec[f"{short}_mae"] = mean_absolute_error(Y[:, i], P[:, i])
                try:
                    rec[f"{short}_r2"] = r2_score(Y[:, i], P[:, i])
                except Exception:
                    rec[f"{short}_r2"] = np.nan
                rec[f"{short}_obs_mean"] = float(np.mean(Y[:, i]))
                rec[f"{short}_pred_mean"] = float(np.mean(P[:, i]))
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser(
        description="Audit COMMAG twin identity-action fidelity and local URLLC-burst behavior."
    )
    ap.add_argument("--prepared", default="prepared_v062")
    ap.add_argument("--split", default="val")
    ap.add_argument("--scenario", default="urllc_burst")
    ap.add_argument("--show", type=int, default=8)
    ap.add_argument("--out", default="runs/v062_seed4/twin_identity_audit.csv")
    args = ap.parse_args()

    prep = Path(args.prepared)
    df = pd.read_parquet(prep / "commag_wide.parquet")
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")
    if hasattr(twin.model, "n_jobs"):
        twin.model.n_jobs = 1
    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(int)

    print("=== ACTION CATALOG ===")
    print(pd.DataFrame(actions, columns=["eMBB", "mMTC", "URLLC"]).to_string(index=True))

    # 1) Negative-request sanity check.
    print("\n=== NEGATIVE REQUEST FRACTION ===")
    neg_rows = []
    for split in sorted(df["split"].dropna().unique()):
        ds = df[df["split"] == split]
        for sc in sorted(ds["scenario"].dropna().unique()):
            g = ds[ds["scenario"] == sc]
            rec = {"split": split, "scenario": sc, "n": len(g)}
            for s in range(3):
                x = pd.to_numeric(g[f"sum_requested_prbs_s{s}"], errors="coerce")
                rec[f"neg_req_s{s}"] = float((x < 0).mean())
            neg_rows.append(rec)
    neg = pd.DataFrame(neg_rows)
    print(neg.round(5).to_string(index=False))

    # 2) Ordinary validation metrics, but broken down by scenario.
    print("\n=== TWIN METRICS BY SPLIT / SCENARIO (OBSERVED ACTION) ===")
    met = scenario_metrics(twin, df)
    focus_cols = [
        "split", "scenario", "n",
        "tput_s0_mae", "tput_s0_r2",
        "tput_s1_mae", "tput_s1_r2",
        "tput_s2_mae", "tput_s2_r2",
        "ratio_s0_mae", "ratio_s0_r2",
        "ratio_s1_mae", "ratio_s1_r2",
        "ratio_s2_mae", "ratio_s2_r2",
    ]
    print(met[focus_cols].round(5).to_string(index=False))

    # 3) Identity-action check in the exact focus regime.
    g = df[(df["split"] == args.split) & (df["scenario"] == args.scenario)].copy()
    if g.empty:
        raise RuntimeError(f"No rows for {args.split}/{args.scenario}")

    records = []
    for idx, row in g.iterrows():
        a_obs = exact_action(row)
        t_obs, r_obs = row_y(row)
        b = np.asarray([row[f"dl_buffer [bytes]_s{s}"] for s in range(3)], dtype=float)

        t_id, r_id = twin.predict_from_row(row, b, a_obs)

        exact_catalog = bool(np.any(np.all(actions == a_obs[None, :], axis=1)))
        nearest_idx = int(np.argmin(np.square(actions - a_obs[None, :]).sum(axis=1)))
        nearest = actions[nearest_idx]

        rec = {
            "row_index": int(idx),
            "exact_catalog_action": exact_catalog,
            "nearest_action_idx": nearest_idx,
            "obs_a0": int(a_obs[0]), "obs_a1": int(a_obs[1]), "obs_a2": int(a_obs[2]),
            "nearest_a0": int(nearest[0]), "nearest_a1": int(nearest[1]), "nearest_a2": int(nearest[2]),
        }
        for s in range(3):
            rec[f"buffer_s{s}"] = float(b[s])
            rec[f"req_s{s}"] = float(row[f"sum_requested_prbs_s{s}"])
            rec[f"cqi_s{s}"] = float(row[f"dl_cqi_s{s}"]) if pd.notna(row[f"dl_cqi_s{s}"]) else np.nan
            rec[f"mcs_s{s}"] = float(row[f"dl_mcs_s{s}"]) if pd.notna(row[f"dl_mcs_s{s}"]) else np.nan
            rec[f"t_obs_s{s}"] = float(t_obs[s])
            rec[f"t_pred_s{s}"] = float(t_id[s])
            rec[f"t_abs_err_s{s}"] = float(abs(t_id[s] - t_obs[s]))
            rec[f"rho_obs_s{s}"] = float(r_obs[s])
            rec[f"rho_pred_s{s}"] = float(r_id[s])
            rec[f"rho_abs_err_s{s}"] = float(abs(r_id[s] - r_obs[s]))
        records.append(rec)

    ident = pd.DataFrame(records)

    print(f"\n=== IDENTITY-ACTION FIDELITY: {args.split}/{args.scenario} ===")
    print(f"rows: {len(ident)}")
    print(f"observed action exactly in catalog: {ident['exact_catalog_action'].mean():.4f}")

    print("\n=== FOCUS ACTION SUPPORT ===")
    action_support = (
        ident.groupby(["obs_a0", "obs_a1", "obs_a2"], as_index=False)
             .agg(
                 n=("row_index", "size"),
                 tput_s0_mae=("t_abs_err_s0", "mean"),
                 tput_s1_mae=("t_abs_err_s1", "mean"),
                 tput_s2_mae=("t_abs_err_s2", "mean"),
                 ratio_s0_mae=("rho_abs_err_s0", "mean"),
                 ratio_s1_mae=("rho_abs_err_s1", "mean"),
                 ratio_s2_mae=("rho_abs_err_s2", "mean"),
             )
             .sort_values("n", ascending=False)
    )
    action_support["share"] = action_support["n"] / len(ident)
    print(action_support.round(5).to_string(index=False))
    for s in range(3):
        print(
            f"s{s}: throughput obs_mean={ident[f't_obs_s{s}'].mean():.6f}, "
            f"pred_mean={ident[f't_pred_s{s}'].mean():.6f}, "
            f"MAE={ident[f't_abs_err_s{s}'].mean():.6f} | "
            f"ratio obs_mean={ident[f'rho_obs_s{s}'].mean():.6f}, "
            f"pred_mean={ident[f'rho_pred_s{s}'].mean():.6f}, "
            f"MAE={ident[f'rho_abs_err_s{s}'].mean():.6f}"
        )

    # Worst local rows by total throughput error.
    ident["t_err_sum"] = sum(ident[f"t_abs_err_s{s}"] for s in range(3))
    ident["rho_err_sum"] = sum(ident[f"rho_abs_err_s{s}"] for s in range(3))
    show_cols = [
        "row_index", "exact_catalog_action",
        "obs_a0", "obs_a1", "obs_a2",
        "buffer_s0", "buffer_s1", "buffer_s2",
        "req_s0", "req_s1", "req_s2",
        "t_obs_s0", "t_pred_s0",
        "t_obs_s1", "t_pred_s1",
        "t_obs_s2", "t_pred_s2",
        "rho_obs_s0", "rho_pred_s0",
        "rho_obs_s1", "rho_pred_s1",
        "rho_obs_s2", "rho_pred_s2",
    ]
    print("\n=== WORST FOCUS ROWS BY THROUGHPUT ERROR ===")
    print(
        ident.sort_values("t_err_sum", ascending=False)
             .head(args.show)[show_cols]
             .round(5)
             .to_string(index=False)
    )

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    ident.to_csv(out, index=False)
    met.to_csv(out.with_name(out.stem + "_metrics_by_scenario.csv"), index=False)
    neg.to_csv(out.with_name(out.stem + "_negative_requests.csv"), index=False)
    action_support.to_csv(out.with_name(out.stem + "_action_support.csv"), index=False)

    print(f"\nSaved:\n  {out}")
    print(f"  {out.with_name(out.stem + '_metrics_by_scenario.csv')}")
    print(f"  {out.with_name(out.stem + '_negative_requests.csv')}")
    print(f"  {out.with_name(out.stem + '_action_support.csv')}")


if __name__ == "__main__":
    main()
