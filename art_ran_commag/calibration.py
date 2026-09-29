from __future__ import annotations

import numpy as np
import pandas as pd


def _queue_health(buffer_value, scale):
    return np.exp(
        -max(float(buffer_value), 0.0)
        / max(float(scale), 1.0)
    )


def utilities(throughput, ratio, buffer_state, cal):
    """
    Bounded service utilities derived from COMMAG-supported signals.

    eMBB:
        throughput-dominant.

    mMTC:
        throughput + queue health.

    URLLC:
        queue-health dominant + per-UE grant/request ratio.

    The URLLC grant ratio is auxiliary because its aggregate trace
    distribution is highly saturated near one.
    """
    e = np.clip(
        float(throughput[0]) / cal["tput_ref_s0"],
        0.0,
        1.0,
    )

    m_t = np.clip(
        float(throughput[1]) / cal["tput_ref_s1"],
        0.0,
        1.0,
    )

    m_q = _queue_health(
        buffer_state[1],
        cal["buffer_scale_s1"],
    )

    u_q = _queue_health(
        buffer_state[2],
        cal["buffer_scale_s2"],
    )

    u_r = np.clip(
        float(ratio[2]),
        0.0,
        1.0,
    )

    m = 0.70 * m_t + 0.30 * m_q
    u = 0.75 * u_q + 0.25 * u_r

    return np.asarray(
        [e, m, u],
        dtype=np.float32,
    )


def violations(u, cal):
    th = np.asarray(
        [
            cal[f"qos_target_s{s}"]
            for s in range(3)
        ],
        dtype=float,
    )

    return (
        np.maximum(th - u, 0.0)
        / np.maximum(th, 1e-6)
    )


def _catalog_connectivity_radius(actions):
    """
    Return the smallest L1 action-change radius that keeps the
    empirical PRB-action catalog connected.

    This prevents a change constraint smaller than the topology of the
    supported action space from collapsing the controller to a
    single-action policy.
    """
    actions = np.asarray(actions, dtype=np.int32)

    if len(actions) <= 1:
        return 0

    dist = np.abs(
        actions[:, None, :] - actions[None, :, :]
    ).sum(axis=2)

    thresholds = np.sort(
        np.unique(dist[dist > 0])
    )

    for radius in thresholds:
        visited = {0}
        stack = [0]

        while stack:
            i = stack.pop()

            neighbours = np.flatnonzero(
                (dist[i] > 0)
                & (dist[i] <= radius)
            )

            for j in neighbours:
                j = int(j)

                if j not in visited:
                    visited.add(j)
                    stack.append(j)

        if len(visited) == len(actions):
            return int(radius)

    return int(dist.max())


# def calibrate(train_df: pd.DataFrame) -> dict:
#     c = {}

def calibrate(
    train_df: pd.DataFrame,
    action_catalog=None,) -> dict:
    c = {}

    # --------------------------------------------------------------
    # Normalization references
    # --------------------------------------------------------------
    for s in range(3):
        c[f"buffer_ref_s{s}"] = max(
            float(
                train_df[
                    f"dl_buffer [bytes]_s{s}"
                ].quantile(0.95)
            ),
            1.0,
        )

        c[f"buffer_scale_s{s}"] = max(
            float(
                train_df[
                    f"dl_buffer [bytes]_s{s}"
                ].quantile(0.90)
            ),
            1.0,
        )

        c[f"req_ref_s{s}"] = max(
            float(
                train_df[
                    f"sum_requested_prbs_s{s}"
                ].quantile(0.95)
            ),
            1.0,
        )

        c[f"cqi_ref_s{s}"] = max(
            float(
                train_df[
                    f"dl_cqi_s{s}"
                ].quantile(0.95)
            ),
            1.0,
        )

        c[f"tput_ref_s{s}"] = max(
            float(
                train_df[
                    f"tx_brate downlink [Mbps]_s{s}"
                ].quantile(0.95)
            ),
            1e-3,
        )

    # --------------------------------------------------------------
    # Empirical QoS targets
    # --------------------------------------------------------------
    # Targets are calibrated from clean static-close training traces.
    # The 20th percentile means that approximately 80% of clean
    # baseline samples satisfy each individual utility target before
    # joint service constraints are imposed.
    base = train_df[
        train_df.rf == "rome_static_close"
    ]

    if len(base) < 100:
        base = train_df

    utilities_clean = []

    for _, row in base.iterrows():
        throughput = np.asarray(
            [
                row[
                    f"tx_brate downlink [Mbps]_s{s}"
                ]
                for s in range(3)
            ],
            dtype=float,
        )

        ratio = np.asarray(
            [
                row[
                    f"ratio_granted_req_s{s}"
                ]
                for s in range(3)
            ],
            dtype=float,
        )

        buffer_state = np.asarray(
            [
                row[
                    f"dl_buffer [bytes]_s{s}"
                ]
                for s in range(3)
            ],
            dtype=float,
        )

        utilities_clean.append(
            utilities(
                throughput,
                ratio,
                buffer_state,
                c,
            )
        )

    utilities_clean = np.asarray(
        utilities_clean,
        dtype=float,
    )

    for s in range(3):
        c[f"qos_target_s{s}"] = float(
            np.quantile(
                utilities_clean[:, s],
                0.20,
            )
        )

    # --------------------------------------------------------------
    # Empirical PRB action catalog
    # --------------------------------------------------------------
    # prb_cols = [
    #     f"slice_prb_s{s}"
    #     for s in range(3)
    # ]

    # sums = train_df[
    #     prb_cols
    # ].sum(axis=1)

    # c["resource_budget"] = int(
    #     round(float(sums.median()))
    # )

    # action_catalog = (
    #     train_df[prb_cols]
    #     .dropna()
    #     .drop_duplicates()
    #     .to_numpy(dtype=np.int32)
    # )

    # action_sums = action_catalog.sum(axis=1)

    # if not np.all(
    #     action_sums == c["resource_budget"]
    # ):
    #     raise ValueError(
    #         "Training action catalog contains inconsistent "
    #         "PRB budgets: "
    #         f"{np.unique(action_sums).tolist()}"
    #     )
    prb_cols = [
        f"slice_prb_s{s}"
        for s in range(3)
    ]

    sums = train_df[prb_cols].sum(axis=1)

    c["resource_budget"] = int(
        round(float(sums.median()))
    )

    if action_catalog is None:
        action_catalog = (
            train_df[prb_cols]
            .dropna()
            .drop_duplicates()
            .to_numpy(dtype=np.int32)
        )
    else:
        action_catalog = np.asarray(
            action_catalog,
            dtype=np.int32,
        )

    action_sums = action_catalog.sum(axis=1)

    if not np.all(
        action_sums == c["resource_budget"]
    ):
        raise ValueError(
            "Supported action catalog contains inconsistent "
            f"PRB budgets: {np.unique(action_sums).tolist()}"
        )
    # --------------------------------------------------------------
    # Empirical action-change statistics
    # --------------------------------------------------------------
    g = train_df.sort_values(
        [
            "rf",
            "tr",
            "exp",
            "bs",
            "time_bin",
        ]
    ).copy()

    dcols = []

    for s in range(3):
        col = f"d{s}"
        dcols.append(col)

        g[col] = (
            g.groupby(
                ["rf", "tr", "exp", "bs"]
            )[f"slice_prb_s{s}"]
            .diff()
            .abs()
        )

    transition_change = (
        g[dcols]
        .sum(
            axis=1,
            min_count=1,
        )
        .dropna()
    )

    if len(transition_change):
        empirical_q90 = int(
            round(
                float(
                    transition_change.quantile(
                        0.90
                    )
                )
            )
        )
    else:
        empirical_q90 = 0

    # Minimum change radius required by the actual supported
    # discrete action space.
    catalog_radius = (
        _catalog_connectivity_radius(
            action_catalog
        )
    )

    c["max_change_empirical_q90"] = int(
        empirical_q90
    )

    c[
        "max_change_catalog_connectivity"
    ] = int(catalog_radius)

    c["max_change_default"] = int(
        max(
            2,
            empirical_q90,
            catalog_radius,
        )
    )

    c["time_bin_ms"] = 250

    return c