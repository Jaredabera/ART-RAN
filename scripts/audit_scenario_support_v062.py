from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


KEYWORDS = (
    "buffer", "req", "request", "grant", "ratio", "throughput", "brate",
    "cqi", "mcs", "prb", "queue", "traffic", "arrival"
)


def main():
    ap = argparse.ArgumentParser(
        description="Audit whether stress scenarios remain within COMMAG empirical support."
    )
    ap.add_argument("--prepared", default="prepared_v062")
    ap.add_argument("--focus-scenario", default="urllc_burst")
    ap.add_argument("--focus-split", default="val")
    ap.add_argument("--out", default="runs/scenario_support_audit.csv")
    args = ap.parse_args()

    prep = Path(args.prepared)
    df = pd.read_parquet(prep / "commag_wide.parquet")

    if "scenario" not in df.columns or "split" not in df.columns:
        raise RuntimeError("Expected 'scenario' and 'split' columns in commag_wide.parquet")

    print("\n=== SCENARIO COUNTS ===")
    counts = (
        df.groupby(["split", "scenario"], dropna=False)
          .size()
          .rename("rows")
          .reset_index()
    )
    print(counts.to_string(index=False))

    numeric = [
        c for c in df.columns
        if pd.api.types.is_numeric_dtype(df[c])
    ]
    chosen = [
        c for c in numeric
        if any(k in c.lower() for k in KEYWORDS)
    ]

    print("\n=== DETECTED RADIO / QUEUE FEATURES ===")
    for c in chosen:
        print(c)

    base = df[df["split"] == "train"]
    focus = df[
        (df["split"] == args.focus_split) &
        (df["scenario"] == args.focus_scenario)
    ]

    if len(focus) == 0:
        raise RuntimeError(
            f"No rows found for split={args.focus_split}, scenario={args.focus_scenario}"
        )

    rows = []
    for c in chosen:
        x = pd.to_numeric(base[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        y = pd.to_numeric(focus[c], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if len(x) < 10 or len(y) < 3:
            continue

        q01, q25, q50, q75, q99 = np.quantile(x, [0.01, 0.25, 0.50, 0.75, 0.99])
        fy01, fy50, fy99 = np.quantile(y, [0.01, 0.50, 0.99])

        iqr = max(q75 - q25, 1e-12)
        outside = np.mean((y < q01) | (y > q99))
        robust_shift = abs(fy50 - q50) / iqr

        rows.append({
            "feature": c,
            "train_q01": q01,
            "train_median": q50,
            "train_q99": q99,
            "focus_q01": fy01,
            "focus_median": fy50,
            "focus_q99": fy99,
            "outside_train_01_99_rate": outside,
            "median_shift_in_train_IQR": robust_shift,
        })

    out = pd.DataFrame(rows).sort_values(
        ["outside_train_01_99_rate", "median_shift_in_train_IQR"],
        ascending=False
    )

    print(
        f"\n=== {args.focus_split}/{args.focus_scenario} VS TRAIN SUPPORT ==="
    )
    print(out.head(40).round(4).to_string(index=False))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path, index=False)

    print(f"\nSaved: {out_path}")
    print("\nInterpretation:")
    print("  high outside_train_01_99_rate  -> empirical out-of-support stress")
    print("  high median_shift_in_train_IQR -> strong distribution shift")
    print("Inspect especially queue/buffer/request/throughput features for slices 1 and 2.")


if __name__ == "__main__":
    main()
