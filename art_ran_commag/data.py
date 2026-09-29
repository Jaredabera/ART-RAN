from __future__ import annotations

from pathlib import Path
import re

import numpy as np
import pandas as pd


TIME_BIN_MS = 250

RAW_COLS = [
    "Timestamp",
    "slice_id",
    "slice_prb",
    "scheduling_policy",
    "dl_buffer [bytes]",
    "tx_brate downlink [Mbps]",
    "sum_requested_prbs",
    "sum_granted_prbs",
    "dl_cqi",
    "dl_mcs",
    "tx_errors downlink (%)",
]


META_RE = re.compile(
    r"slice_traffic/"
    r"(?P<rf>[^/]+)/"
    r"(?P<tr>tr\d+)/"
    r"(?P<exp>exp\d+)/"
    r"(?P<bs>bs\d+)/"
    r"slices_[^/]+/"
)


# ============================================================================
# Metadata parsing
# ============================================================================

def _parse_meta(path: Path):
    p = str(path).replace("\\", "/")

    m = META_RE.search(p)

    if not m:
        raise ValueError(
            f"COMMAG path does not match expected structure: {path}"
        )

    return m.groupdict()


# ============================================================================
# Read and aggregate one COMMAG metrics file
# ============================================================================

def read_one_metrics(
    path: str | Path,
    time_bin_ms: int = TIME_BIN_MS,
) -> pd.DataFrame:
    """
    Read one per-UE COMMAG metrics file and aggregate it into
    slice-level 250-ms bins.

    The granted/requested PRB ratio is first computed at the individual
    measurement level and then averaged. This preserves service shortfalls
    that can disappear if requested and granted PRBs are aggregated before
    computing the ratio.

    Non-positive requested-PRB reports are treated as invalid measurements
    and removed before aggregation.
    """

    path = Path(path)
    meta = _parse_meta(path)

    # ------------------------------------------------------------------
    # Load required COMMAG columns
    # ------------------------------------------------------------------
    df = pd.read_csv(
        path,
        usecols=lambda c: c in RAW_COLS,
        low_memory=False,
    )

    missing = [
        c
        for c in RAW_COLS
        if c not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{path}: missing {missing}"
        )

    # ------------------------------------------------------------------
    # Convert all required fields to numeric
    # ------------------------------------------------------------------
    for c in RAW_COLS:
        df[c] = pd.to_numeric(
            df[c],
            errors="coerce",
        )

    df = df.dropna(
        subset=[
            "Timestamp",
            "slice_id",
        ]
    )

    # ------------------------------------------------------------------
    # Construct 250-ms time bins
    # ------------------------------------------------------------------
    df["time_bin"] = (
        df["Timestamp"].astype("int64")
        // time_bin_ms
    ) * time_bin_ms

    # ------------------------------------------------------------------
    # Sanitize COMMAG PRB counters before aggregation
    # ------------------------------------------------------------------

    # Treat non-positive requested-PRB reports as invalid observations
    # rather than physical zero-demand measurements.
    df = df.loc[
        df["sum_requested_prbs"] > 0
    ].copy()

    # Granted PRBs must remain physically non-negative.
    df["sum_granted_prbs"] = (
        df["sum_granted_prbs"]
        .clip(lower=0)
    )

    req = (
        df["sum_requested_prbs"]
        .astype(float)
    )

    granted = (
        df["sum_granted_prbs"]
        .astype(float)
    )

    # Per-report grant satisfaction ratio.
    df["ratio_granted_req"] = (
        granted / req
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # Fraction of requested PRBs that were not granted.
    df["grant_shortfall_frac"] = (
        (req - granted)
        .clip(lower=0.0)
        / req
    ).clip(
        lower=0.0,
        upper=1.0,
    )

    # ------------------------------------------------------------------
    # Aggregate one metrics file into slice/time bins
    # ------------------------------------------------------------------
    agg = (
        df.groupby(
            [
                "time_bin",
                "slice_id",
            ],
            as_index=False,
        )
        .agg(
            {
                "slice_prb": "median",
                "scheduling_policy": "median",
                "dl_buffer [bytes]": "sum",
                "tx_brate downlink [Mbps]": "sum",
                "sum_requested_prbs": "sum",
                "sum_granted_prbs": "sum",
                "ratio_granted_req": "mean",
                "grant_shortfall_frac": "mean",
                "dl_cqi": "mean",
                "dl_mcs": "mean",
                "tx_errors downlink (%)": "mean",
            }
        )
    )

    # Add RF / traffic / experiment / BS metadata.
    for k, v in meta.items():
        agg[k] = v

    return agg


# ============================================================================
# Build system-wide COMMAG dataframe
# ============================================================================

def build_commag_wide(
    data_root: str | Path,
    output_parquet: str | Path | None = None,
) -> pd.DataFrame:

    data_root = Path(data_root)

    paths = sorted(
        data_root.glob(
            "slice_traffic/"
            "*/tr*/exp*/bs*/"
            "slices_bs*/*_metrics.csv"
        )
    )

    if not paths:
        raise FileNotFoundError(
            f"No COMMAG *_metrics.csv below {data_root}"
        )

    pieces = []

    for i, p in enumerate(paths, 1):
        piece = read_one_metrics(p)

        if not piece.empty:
            pieces.append(piece)

        if i % 250 == 0:
            print(
                f"  parsed {i}/{len(paths)} files"
            )

    if not pieces:
        raise RuntimeError(
            "No valid COMMAG measurements remained "
            "after preprocessing."
        )

    long = pd.concat(
        pieces,
        ignore_index=True,
    )

    # ------------------------------------------------------------------
    # Merge aggregates belonging to the same BS/slice/time bin
    # ------------------------------------------------------------------
    keys = [
        "rf",
        "tr",
        "exp",
        "bs",
        "time_bin",
        "slice_id",
    ]

    long = (
        long.groupby(
            keys,
            as_index=False,
        )
        .agg(
            {
                "slice_prb": "median",
                "scheduling_policy": "median",
                "dl_buffer [bytes]": "sum",
                "tx_brate downlink [Mbps]": "sum",
                "sum_requested_prbs": "sum",
                "sum_granted_prbs": "sum",
                "ratio_granted_req": "mean",
                "grant_shortfall_frac": "mean",
                "dl_cqi": "mean",
                "dl_mcs": "mean",
                "tx_errors downlink (%)": "mean",
            }
        )
    )

    # ------------------------------------------------------------------
    # Convert long slice representation to one row per BS/time bin
    # ------------------------------------------------------------------
    metrics = [
        "slice_prb",
        "scheduling_policy",
        "dl_buffer [bytes]",
        "tx_brate downlink [Mbps]",
        "sum_requested_prbs",
        "sum_granted_prbs",
        "ratio_granted_req",
        "grant_shortfall_frac",
        "dl_cqi",
        "dl_mcs",
        "tx_errors downlink (%)",
    ]

    wide = (
        long.set_index(
            [
                "rf",
                "tr",
                "exp",
                "bs",
                "time_bin",
                "slice_id",
            ]
        )[metrics]
        .unstack("slice_id")
    )

    wide.columns = [
        f"{metric}_s{int(slice_id)}"
        for metric, slice_id in wide.columns
    ]

    wide = (
        wide.reset_index()
        .sort_values(
            [
                "rf",
                "tr",
                "exp",
                "bs",
                "time_bin",
            ]
        )
        .reset_index(drop=True)
    )

    # ------------------------------------------------------------------
    # Require all three slice allocations
    # ------------------------------------------------------------------
    required = [
        f"slice_prb_s{s}"
        for s in range(3)
    ]

    wide = (
        wide.dropna(
            subset=required
        )
        .copy()
    )

    for c in required:
        wide[c] = (
            np.rint(wide[c])
            .astype(int)
        )

    # ------------------------------------------------------------------
    # Final physical sanity checks
    # ------------------------------------------------------------------
    for s in range(3):
        req_col = f"sum_requested_prbs_s{s}"
        grant_col = f"sum_granted_prbs_s{s}"

        if (wide[req_col] <= 0).any():
            raise RuntimeError(
                f"Non-positive requested PRBs remain in {req_col}"
            )

        if (wide[grant_col] < 0).any():
            raise RuntimeError(
                f"Negative granted PRBs remain in {grant_col}"
            )

    # ------------------------------------------------------------------
    # Optional raw-wide parquet output
    # ------------------------------------------------------------------
    if output_parquet is not None:
        output_parquet = Path(
            output_parquet
        )

        output_parquet.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        wide.to_parquet(
            output_parquet,
            index=False,
        )

    return wide


# ============================================================================
# Train / validation / test split and scenario labels
# ============================================================================

def add_splits_and_labels(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, dict]:

    df = df.copy()

    exp_num = (
        df["exp"]
        .str.extract(r"(\d+)")[0]
        .astype(int)
    )

    # ------------------------------------------------------------------
    # Experiment-level split
    # exp1-exp4 -> train
    # exp5      -> validation
    # exp6+     -> test
    # ------------------------------------------------------------------
    df["split"] = np.select(
        [
            exp_num <= 4,
            exp_num == 5,
            exp_num == 6,
        ],
        [
            "train",
            "val",
            "test",
        ],
        default="test",
    )

    # ------------------------------------------------------------------
    # Scenario thresholds are calibrated only from clean static-close
    # training data.
    # ------------------------------------------------------------------
    train_close = df[
        (df["split"] == "train")
        & (df["rf"] == "rome_static_close")
    ]

    if train_close.empty:
        raise RuntimeError(
            "No static-close training rows available "
            "for scenario calibration."
        )

    q_u = float(
        train_close[
            "sum_requested_prbs_s2"
        ].quantile(0.85)
    )

    q_e = float(
        train_close[
            "sum_requested_prbs_s0"
        ].quantile(0.85)
    )

    # ------------------------------------------------------------------
    # Scenario labeling
    # ------------------------------------------------------------------
    labels = []

    for r in df.itertuples(
        index=False
    ):
        if r.rf in (
            "rome_static_medium",
            "rome_static_far",
        ):
            lab = "channel_shift"

        elif r.rf == "rome_slow_close":
            lab = "mobility_shift"

        elif (
            getattr(
                r,
                "sum_requested_prbs_s2",
            )
            >= q_u
        ):
            lab = "urllc_burst"

        elif (
            getattr(
                r,
                "sum_requested_prbs_s0",
            )
            >= q_e
        ):
            lab = "embb_surge"

        else:
            lab = "nominal"

        labels.append(lab)

    df["scenario"] = labels

    thresholds = {
        "urllc_req_q85": q_u,
        "embb_req_q85": q_e,
    }

    return df, thresholds


# ============================================================================
# Build trace-driven transitions
# ============================================================================

def build_transitions(
    df: pd.DataFrame,
    time_bin_ms: int = TIME_BIN_MS,
) -> pd.DataFrame:
    """
    Add next observed buffer and infer exogenous arrivals using

        B_{t+1} = max(B_t - S_t, 0) + A_t

    therefore

        A_t = max(
            B_{t+1}
            - max(B_t - S_t, 0),
            0
        ).

    The inferred arrival process remains trace-driven while a
    counterfactual action changes the service process through the
    empirical COMMAG response model.
    """

    df = (
        df.sort_values(
            [
                "rf",
                "tr",
                "exp",
                "bs",
                "time_bin",
            ]
        )
        .copy()
    )

    grp = [
        "rf",
        "tr",
        "exp",
        "bs",
    ]

    # ------------------------------------------------------------------
    # Time difference to next observation
    # ------------------------------------------------------------------
    next_time = (
        df.groupby(grp)["time_bin"]
        .shift(-1)
    )

    dt = (
        (next_time - df["time_bin"])
        / 1000.0
    ).clip(
        lower=0.05,
        upper=1.0,
    )

    df["dt_sec"] = dt

    # ------------------------------------------------------------------
    # Infer exogenous arrivals for each slice
    # ------------------------------------------------------------------
    for s in range(3):
        buffer_col = (
            f"dl_buffer [bytes]_s{s}"
        )

        throughput_col = (
            f"tx_brate downlink [Mbps]_s{s}"
        )

        next_buffer_col = (
            f"next_buffer_s{s}"
        )

        df[next_buffer_col] = (
            df.groupby(grp)[buffer_col]
            .shift(-1)
        )

        # Bytes served during this transition.
        service = (
            df[throughput_col]
            .clip(lower=0)
            * 1e6
            / 8.0
            * dt
        ).fillna(0.0)

        residual = (
            df[next_buffer_col]
            - np.maximum(
                df[buffer_col] - service,
                0.0,
            )
        ).clip(
            lower=0.0
        )

        df[
            f"arrival_bytes_s{s}"
        ] = residual

    # ------------------------------------------------------------------
    # Remove terminal rows without a valid next state
    # ------------------------------------------------------------------
    required = [
        f"next_buffer_s{s}"
        for s in range(3)
    ] + [
        "dt_sec"
    ]

    return (
        df.dropna(
            subset=required
        )
        .reset_index(drop=True)
    )


# ============================================================================
# Observed COMMAG action catalog
# ============================================================================

def observed_action_catalog(
    train_df: pd.DataFrame,
    min_count: int = 100,
) -> tuple[np.ndarray, pd.DataFrame]:

    action_cols = [
        f"slice_prb_s{s}"
        for s in range(3)
    ]

    counts = (
        train_df.groupby(
            action_cols
        )
        .size()
        .reset_index(
            name="count"
        )
        .sort_values(
            "count",
            ascending=False,
        )
    )

    kept = counts[
        counts["count"]
        >= min_count
    ]

    # Preserve a usable empirical action set if the count threshold
    # leaves too few actions.
    if len(kept) < 5:
        kept = counts.head(
            min(
                15,
                len(counts),
            )
        )

    actions = (
        kept[action_cols]
        .to_numpy(
            dtype=np.int32
        )
    )

    return actions, counts