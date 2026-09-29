#!/usr/bin/env python3
"""
ART-RAN IEEE publication plotting utility.

This script generates publication-ready figures from REAL experiment CSV files.
It never fabricates values.

Outputs
-------
Figure 1: PPO validation learning curves, one panel per validation scenario.
          Mean across independent training seeds; 95% t-confidence interval.
Figure 2: Attack-impact comparison for joint QoS satisfaction and URLLC violation.
          Grouped grayscale/hatch bars with 95% t-confidence intervals.
Table I:  Macro-averaged attack summary as CSV and LaTeX.

IEEE-oriented graphics
----------------------
* PDF and EPS are vector outputs.
* PNG is exported at 600 dpi for black/white line-art compatibility.
* Two-column figure width = 7.16 in.
* All methods remain distinguishable without color by hatch, marker, and line style.

Expected learning-curve inputs
------------------------------
runs/v062_seed0/validation_learning_curves.csv
...
runs/v062_seed4/validation_learning_curves.csv

Each file is expected to contain at least:
    timesteps, scenario, return_mean

Expected attack-evaluation CSV
------------------------------
Required columns:
    seed
    scenario
    method
    episode
    all_qos_rate
    urllc_violation_rate
    embb_throughput_mbps
    action_change_rate
    attack_success

Rows must be episode-level and paired across methods using
(seed, scenario, episode).  The clean method key must be "clean".

Recommended method keys:
    clean
    obs_attack
    maloran                 # optional external reproduced baseline
    context_attack
    lifecycle_attack
    reasoning_attack

Important scientific note
-------------------------
Do not paste numbers reported in another paper into this CSV.  A prior-work row
should contain results from a reproduction of that attack under the SAME
COMMAG traces, actuator, resource budget, and evaluation protocol.

Definitions:

- ll_qos_rate: fraction of time steps in the episode where all slice QoS targets are satisfied.
- urllc_violation_rate: fraction of time steps with URLLC QoS violation.
- embb_throughput_mbps: mean eMBB throughput over the episode.
- action_change_rate: fraction of time steps where the attacked downstream PRB
- action differs from the paired clean action.
- attack_success: binary or fractional attack-success measure defined at the
- rApp layer. Define the criterion once and keep it identical across attacks.

The plotting script computes eMBB throughput loss
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import t as student_t


# ---------------------------------------------------------------------------
# Labels used only for presentation. Code/database scenario names remain intact.
# ---------------------------------------------------------------------------

SCENARIO_ORDER = [
    "nominal",
    "urllc_burst",
    "embb_surge",
    "channel_shift",
    "mobility_shift",
]

SCENARIO_LABEL = {
    "nominal": "Normal operation",
    "urllc_burst": "URLLC traffic burst",
    "embb_surge": "eMBB traffic surge",
    "channel_shift": "Channel shift",
    "mobility_shift": "Mobility shift",
    "mixed": "Mixed stress",
}

LEARNING_SCENARIOS = [
    "nominal",
    "urllc_burst",
    "embb_surge",
    "channel_shift",
]

METHOD_ORDER = [
    "clean",
    "obs_attack",
    "maloran",
    "context_attack",
    "lifecycle_attack",
    "reasoning_attack",
]

METHOD_LABEL = {
    "clean": "Clean",
    "obs_attack": "Obs.-space baseline",
    "maloran": "MalO-RAN baseline",
    "context_attack": "Context attack",
    "lifecycle_attack": "Lifecycle attack",
    "reasoning_attack": "Reasoning attack",
}

# Black/white distinguishability: hatches are primary; grayscale is secondary.
METHOD_STYLE = {
    "clean":            dict(facecolor="white", hatch="",      marker="o", linestyle="-"),
    "obs_attack":       dict(facecolor="0.88", hatch="////",   marker="s", linestyle="--"),
    "maloran":          dict(facecolor="0.78", hatch="\\\\\\\\", marker="^", linestyle="-."),
    "context_attack":   dict(facecolor="0.68", hatch="xx",     marker="D", linestyle=":"),
    "lifecycle_attack": dict(facecolor="0.58", hatch="..",     marker="v", linestyle=(0, (5, 2))),
    "reasoning_attack": dict(facecolor="0.45", hatch="++",     marker="P", linestyle=(0, (3, 1, 1, 1))),
}


# ---------------------------------------------------------------------------
# IEEE visual style
# ---------------------------------------------------------------------------

def set_ieee_style() -> None:
    """Set a conservative IEEE-friendly Matplotlib style."""
    mpl.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "axes.linewidth": 0.75,
        "lines.linewidth": 1.25,
        "lines.markersize": 4.0,
        "xtick.major.width": 0.7,
        "ytick.major.width": 0.7,
        "xtick.major.size": 3.0,
        "ytick.major.size": 3.0,
        "grid.linewidth": 0.45,
        "grid.alpha": 0.32,
        "grid.linestyle": "--",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "axes.unicode_minus": True,
    })


def save_figure(fig: plt.Figure, stem: Path) -> None:
    """Save the same figure as vector PDF/EPS and 600-dpi PNG."""
    stem.parent.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "eps"):
        fig.savefig(
            stem.with_suffix("." + ext),
            bbox_inches="tight",
            pad_inches=0.025,
        )
    fig.savefig(
        stem.with_suffix(".png"),
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.025,
    )


def t_ci95(values: np.ndarray) -> tuple[float, float]:
    """Mean and 95% t-CI half-width across independent seeds."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]
    if len(x) == 0:
        return np.nan, np.nan
    mean = float(x.mean())
    if len(x) == 1:
        return mean, np.nan
    se = float(x.std(ddof=1) / np.sqrt(len(x)))
    half = float(student_t.ppf(0.975, df=len(x) - 1) * se)
    return mean, half


# ---------------------------------------------------------------------------
# Figure 1: standard RL validation learning curves
# ---------------------------------------------------------------------------

def load_learning_runs(runs_root: Path, seeds: list[int]) -> pd.DataFrame:
    frames = []
    for seed in seeds:
        path = runs_root / f"v062_seed{seed}" / "validation_learning_curves.csv"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path}. Run all requested seeds before plotting."
            )
        d = pd.read_csv(path)
        needed = {"timesteps", "scenario", "return_mean"}
        missing = needed - set(d.columns)
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        d = d.copy()
        d["seed"] = seed
        frames.append(d)
    return pd.concat(frames, ignore_index=True)


def plot_learning_curves(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot one standard RL learning curve per validation scenario.

    Point = mean validation episodic return across independent PPO training seeds.
    Band  = 95% t-confidence interval across those seeds.
    No smoothing is applied.
    """
    set_ieee_style()

    fig, axes = plt.subplots(
        2, 2,
        figsize=(7.16, 4.75),
        sharex=True,
    )
    axes = axes.ravel()

    panel_markers = ["o", "s", "^", "D"]

    for ax, scenario, marker in zip(axes, LEARNING_SCENARIOS, panel_markers):
        sub = df[df["scenario"] == scenario].copy()
        if sub.empty:
            raise ValueError(f"No learning-curve rows for scenario={scenario!r}")

        # Each seed contributes one validation mean at each checkpoint.
        grouped = []
        for step, g in sub.groupby("timesteps", sort=True):
            mean, ci = t_ci95(g["return_mean"].to_numpy(float))
            grouped.append((int(step), mean, ci, g["seed"].nunique()))

        gdf = pd.DataFrame(grouped, columns=["timesteps", "mean", "ci95", "n_seed"])
        if gdf["n_seed"].min() < 2:
            raise ValueError(
                f"{scenario}: fewer than two independent seeds at some checkpoints."
            )

        x = gdf["timesteps"].to_numpy()
        y = gdf["mean"].to_numpy()
        ci = gdf["ci95"].to_numpy()

        ax.plot(
            x, y,
            color="black",
            marker=marker,
            markevery=1,
            linewidth=1.25,
            markerfacecolor="white",
            markeredgecolor="black",
            markeredgewidth=0.8,
        )
        ax.fill_between(
            x,
            y - ci,
            y + ci,
            facecolor="0.82",
            edgecolor="none",
            alpha=0.60,
        )

        ax.set_title(SCENARIO_LABEL[scenario], pad=2.5)
        ax.grid(True, axis="both")
        ax.set_axisbelow(True)
        ax.margins(x=0.025)

    axes[0].set_ylabel("Validation episodic return")
    axes[2].set_ylabel("Validation episodic return")
    axes[2].set_xlabel("Environment steps")
    axes[3].set_xlabel("Environment steps")

    # Panel labels in the conventional IEEE style.
    for label, ax in zip(["(a)", "(b)", "(c)", "(d)"], axes):
        ax.text(
            0.5, -0.27, label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.0,
        )

    fig.subplots_adjust(
        left=0.09, right=0.99, top=0.95, bottom=0.14,
        wspace=0.23, hspace=0.40,
    )

    save_figure(fig, outdir / "fig_ppo_learning_curves")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Attack-evaluation aggregation
# ---------------------------------------------------------------------------

ATTACK_REQUIRED = {
    "seed",
    "scenario",
    "method",
    "episode",
    "all_qos_rate",
    "urllc_violation_rate",
    "embb_throughput_mbps",
    "action_change_rate",
    "attack_success",
}


def load_attack_results(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    missing = ATTACK_REQUIRED - set(df.columns)
    if missing:
        raise ValueError(
            f"{path} is missing required columns: {sorted(missing)}"
        )
    if "clean" not in set(df["method"]):
        raise ValueError('Attack CSV must contain method="clean".')

    # Validate pairing: clean row must exist for every attack row's
    # (seed, scenario, episode).
    clean_keys = set(
        map(
            tuple,
            df.loc[df["method"] == "clean", ["seed", "scenario", "episode"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    attack_keys = set(
        map(
            tuple,
            df.loc[df["method"] != "clean", ["seed", "scenario", "episode"]]
            .drop_duplicates()
            .to_numpy(),
        )
    )
    missing_clean = attack_keys - clean_keys
    if missing_clean:
        sample = list(missing_clean)[:5]
        raise ValueError(
            "Attack results are not fully paired with clean episodes. "
            f"Example missing keys: {sample}"
        )
    return df


def available_methods(df: pd.DataFrame) -> list[str]:
    present = set(df["method"].unique())
    known = [m for m in METHOD_ORDER if m in present]
    unknown = sorted(present - set(METHOD_ORDER))
    if unknown:
        raise ValueError(
            "Unknown method keys in attack CSV: "
            f"{unknown}. Update METHOD_ORDER/METHOD_STYLE explicitly."
        )
    return known


def seed_scenario_metric(
    df: pd.DataFrame,
    metric: str,
) -> pd.DataFrame:
    """Episode -> seed/scenario/method means; seeds remain statistical units."""
    return (
        df.groupby(["seed", "scenario", "method"], as_index=False)[metric]
        .mean()
    )


# ---------------------------------------------------------------------------
# Figure 2: B/W attack comparison
# ---------------------------------------------------------------------------

def plot_attack_bars(df: pd.DataFrame, outdir: Path) -> None:
    """
    Two-panel attack-impact figure.

    (a) Joint QoS satisfaction rate (%), higher is better.
    (b) URLLC violation rate (%), lower is better.

    Bars show means across independent seeds after each seed was first averaged
    over episodes. Error bars are 95% t-confidence intervals across seeds.
    """
    set_ieee_style()

    methods = available_methods(df)
    scenarios = [s for s in SCENARIO_ORDER if s in set(df["scenario"])]
    if not scenarios:
        raise ValueError("No recognized evaluation scenarios in attack CSV.")

    fig, axes = plt.subplots(
        1, 2,
        figsize=(7.16, 3.15),
        sharey=False,
    )

    panels = [
        ("all_qos_rate", "Joint QoS satisfaction (%)", axes[0]),
        ("urllc_violation_rate", "URLLC violation rate (%)", axes[1]),
    ]

    x = np.arange(len(scenarios), dtype=float)
    n_m = len(methods)
    total_group_width = 0.82
    width = total_group_width / max(n_m, 1)

    for metric, ylabel, ax in panels:
        agg = seed_scenario_metric(df, metric)

        for j, method in enumerate(methods):
            style = METHOD_STYLE[method]
            offset = (j - (n_m - 1) / 2.0) * width

            means = []
            cis = []

            for scenario in scenarios:
                vals = agg.loc[
                    (agg["scenario"] == scenario) &
                    (agg["method"] == method),
                    metric,
                ].to_numpy(float)

                mean, ci = t_ci95(vals * 100.0)
                means.append(mean)
                cis.append(ci)

            ax.bar(
                x + offset,
                means,
                width=width * 0.93,
                label=METHOD_LABEL[method],
                facecolor=style["facecolor"],
                edgecolor="black",
                linewidth=0.65,
                hatch=style["hatch"],
                yerr=cis,
                error_kw=dict(
                    ecolor="black",
                    elinewidth=0.65,
                    capsize=1.8,
                    capthick=0.65,
                ),
                zorder=3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(
            [SCENARIO_LABEL[s] for s in scenarios],
            rotation=18,
            ha="right",
        )
        ax.set_ylabel(ylabel)
        ax.set_ylim(bottom=0)
        ax.grid(True, axis="y")
        ax.set_axisbelow(True)

    # Shared legend above both panels keeps the plotting area clean.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=min(3, len(methods)),
        frameon=False,
        bbox_to_anchor=(0.5, 1.01),
        columnspacing=1.1,
        handlelength=1.7,
    )

    for label, ax in zip(["(a)", "(b)"], axes):
        ax.text(
            0.5, -0.34, label,
            transform=ax.transAxes,
            ha="center", va="top",
            fontsize=8.0,
        )

    fig.subplots_adjust(
        left=0.08, right=0.995, top=0.84, bottom=0.25,
        wspace=0.25,
    )

    save_figure(fig, outdir / "fig_attack_qos_impact")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Table I: paired macro-averaged attack summary
# ---------------------------------------------------------------------------

def _paired_throughput_loss(df: pd.DataFrame) -> pd.DataFrame:
    """
    Episode-level eMBB throughput loss relative to the paired clean trajectory:
        100 * (clean - attacked) / clean

    Positive = throughput degradation.
    """
    keys = ["seed", "scenario", "episode"]

    clean = (
        df[df["method"] == "clean"][keys + ["embb_throughput_mbps"]]
        .rename(columns={"embb_throughput_mbps": "embb_clean"})
    )

    merged = df.merge(clean, on=keys, how="left", validate="many_to_one")
    denom = merged["embb_clean"].to_numpy(float)

    loss = np.full(len(merged), np.nan, dtype=float)
    ok = np.isfinite(denom) & (np.abs(denom) > 1e-12)
    loss[ok] = (
        100.0
        * (
            merged.loc[ok, "embb_clean"].to_numpy(float)
            - merged.loc[ok, "embb_throughput_mbps"].to_numpy(float)
        )
        / denom[ok]
    )
    merged["embb_loss_pct"] = loss

    # Clean is exactly the reference.
    merged.loc[merged["method"] == "clean", "embb_loss_pct"] = 0.0
    return merged


def macro_seed_values(df: pd.DataFrame, metric: str) -> pd.DataFrame:
    """
    Macro-average correctly:
      episode -> seed/scenario/method mean
      scenarios -> seed/method macro mean

    The final CI is therefore across independent training seeds, not episodes.
    """
    ssm = (
        df.groupby(["seed", "scenario", "method"], as_index=False)[metric]
        .mean()
    )
    return (
        ssm.groupby(["seed", "method"], as_index=False)[metric]
        .mean()
    )


def fmt_mean_ci(vals: np.ndarray, scale: float = 1.0, digits: int = 1) -> str:
    mean, ci = t_ci95(np.asarray(vals, dtype=float) * scale)
    if not np.isfinite(mean):
        return "--"
    if not np.isfinite(ci):
        return f"{mean:.{digits}f}"
    return f"{mean:.{digits}f} $\\pm$ {ci:.{digits}f}"


def make_summary_table(df: pd.DataFrame, outdir: Path) -> None:
    """
    Produce a macro-averaged table using independent seeds as the uncertainty unit.

    Columns:
      Attack success rate (%)
      Downstream action-change rate (%)
      Joint QoS satisfaction (%)
      URLLC violation rate (%)
      Paired eMBB throughput loss (%)
    """
    outdir.mkdir(parents=True, exist_ok=True)
    d = _paired_throughput_loss(df)
    methods = available_methods(d)

    metrics = {
        "attack_success": macro_seed_values(d, "attack_success"),
        "action_change_rate": macro_seed_values(d, "action_change_rate"),
        "all_qos_rate": macro_seed_values(d, "all_qos_rate"),
        "urllc_violation_rate": macro_seed_values(d, "urllc_violation_rate"),
        "embb_loss_pct": macro_seed_values(d, "embb_loss_pct"),
    }

    rows_plain = []
    rows_tex = []

    for method in methods:
        def vals(metric):
            return metrics[metric].loc[
                metrics[metric]["method"] == method, metric
            ].to_numpy(float)

        if method == "clean":
            asr_tex = "--"
        else:
            asr_tex = fmt_mean_ci(vals("attack_success"), 100.0, 1)

        action_tex = (
            "0.0"
            if method == "clean"
            else fmt_mean_ci(vals("action_change_rate"), 100.0, 1)
        )
        qos_tex = fmt_mean_ci(vals("all_qos_rate"), 100.0, 1)
        urllc_tex = fmt_mean_ci(vals("urllc_violation_rate"), 100.0, 1)
        loss_tex = (
            "0.0"
            if method == "clean"
            else fmt_mean_ci(vals("embb_loss_pct"), 1.0, 1)
        )

        rows_plain.append({
            "Method": METHOD_LABEL[method],
            "ASR (%)": asr_tex.replace("$\\pm$", "±"),
            "Action change (%)": action_tex.replace("$\\pm$", "±"),
            "All-QoS (%)": qos_tex.replace("$\\pm$", "±"),
            "URLLC violation (%)": urllc_tex.replace("$\\pm$", "±"),
            "eMBB throughput loss (%)": loss_tex.replace("$\\pm$", "±"),
        })

        if method == "obs_attack":
            method_tex = r"Observation-space attack~\cite{10697477}"
        elif method == "maloran":
            method_tex = r"MalO-RAN~\cite{LACAVA2025111727}"
        elif method == "clean":
            method_tex = r"Clean"
        elif method == "context_attack":
            method_tex = r"ART-RAN: context"
        elif method == "lifecycle_attack":
            method_tex = r"ART-RAN: lifecycle"
        elif method == "reasoning_attack":
            method_tex = r"ART-RAN: reasoning"
        else:
            method_tex = METHOD_LABEL[method]

        rows_tex.append(
            f"{method_tex} & {asr_tex} & {action_tex} & {qos_tex} & "
            f"{urllc_tex} & {loss_tex} \\\\"
        )

    pd.DataFrame(rows_plain).to_csv(
        outdir / "table_attack_summary.csv", index=False
    )

    tex = r"""\begin{table*}[t]
\centering
\caption{Macro-averaged ART-RAN attack impact over the evaluation scenarios. Values are mean $\pm$ 95\% confidence interval across independent PPO training seeds. eMBB throughput loss is computed relative to the paired clean trajectory.}
\label{tab:attack-summary}
\setlength{\tabcolsep}{4.2pt}
\renewcommand{\arraystretch}{1.12}
\begin{tabular}{lccccc}
\toprule
Method &
ASR (\%) &
Action change (\%) &
All-QoS (\%) $\uparrow$ &
URLLC viol. (\%) $\downarrow$ &
eMBB loss (\%) $\downarrow$ \\
\midrule
""" + "\n".join(rows_tex) + r"""
\bottomrule
\end{tabular}
\end{table*}
"""
    (outdir / "table_attack_summary.tex").write_text(tex)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_seeds(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--runs-root",
        type=Path,
        default=Path("runs"),
        help="Root containing v062_seed0, v062_seed1, ...",
    )
    p.add_argument(
        "--seeds",
        default="0,1,2,3,4",
        help="Comma-separated independent PPO training seeds.",
    )
    p.add_argument(
        "--attack-csv",
        type=Path,
        default=None,
        help="Episode-level paired attack evaluation CSV.",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("paper_figures"),
    )
    args = p.parse_args()

    seeds = parse_seeds(args.seeds)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    learning = load_learning_runs(args.runs_root, seeds)
    plot_learning_curves(learning, args.out_dir)
    print("[ok] Figure 1: PPO learning curves")

    if args.attack_csv is not None:
        attack = load_attack_results(args.attack_csv)
        plot_attack_bars(attack, args.out_dir)
        make_summary_table(attack, args.out_dir)
        print("[ok] Figure 2: attack QoS impact")
        print("[ok] Table I: attack summary")
    else:
        print(
            "[note] --attack-csv not provided; attack figure/table were not "
            "generated. This avoids fabricating attack results."
        )

    print(f"[done] outputs: {args.out_dir.resolve()}")


if __name__ == "__main__":
    main()
