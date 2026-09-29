"""ART-RAN v0.6.2: Maskable-PPO training on the COMMAG-grounded environment.

Scientific role
---------------
This script trains the *downstream Near-RT actuator* used by ART-RAN.  The PPO controller is not the paper's novelty; it is the controlled actuator through
which an upstream rApp policy is translated into feasible PRB-allocation choices.

Data split and scenario policy
------------------------------
* Training data: rows labelled ``split == 'train'`` (exp1--exp4 in the current preparation pipeline).
* Validation data: rows labelled ``split == 'val'`` (exp5).
* ``mobility_shift`` is intentionally NOT used by this script for model selection.  Keep it for the final held-out distribution-shift evaluation.
* Five training environments are used, one per TRAIN_SCENARIOS entry.  Because  DummyVecEnv advances each environment once per vector step, the five training
  scenarios receive equal sampling weight rather than weight proportional to the number of rows in the dataset.

Observation/action model
------------------------
The current v0.6.2 environment exposes a 29-D observation: 22-D RAN state + 7-D rApp policy context. The discrete action catalog contains the empirically supported PRB allocations.
Action masking enforces the rApp-derived minimum-allocation and maximum-change constraints before MaskablePPO chooses an action.

Reproducibility
---------------
``--seed`` is only a pseudo-random seed; it has no statistical meaning.  The recommended paper protocol is to train multiple independent seeds (for example
0, 1, 2, 3, 4) and aggregate across those runs.  Validation trajectory seeds are intentionally fixed across training seeds so every trained policy is
compared on the same validation trajectories.

Validation semantics
--------------------
Validation is performed only after completed PPO updates (at rollout boundaries), not in the middle of a rollout.  For every visited validation
state, all currently admissible actions are evaluated with the environment's side-effect-free ``counterfactual`` method.  This produces one-step diagnostic
quantities such as regret and oracle_score; these are NOT trajectory-level oracles and are not be presented as the primary ART-RAN attack metrics.

The headline downstream metrics for our paper remain interpretable RAN quantities (e.g., slice throughput, QoS-violation rate, URLLC reliability,
PRB/action deviation).  The one-step oracle diagnostics are useful for checking controller quality and model selection.

Performance notes
-----------------
* The ExtraTrees empirical twin is queried one state/action at a time.  Its  ``n_jobs`` is therefore set to 1 by default to avoid per-prediction thread 
scheduling overhead.  Use ``--twin-n-jobs`` to benchmark another value. 
  * Validation environments are constructed once and reused.  Reconstructing a COMMAGPolicyEnv for every validation episode is expensive because the
  environment precomputes eligible trajectory then starts. 
  * DummyVecEnv is deliberate: it keeps one in-process copy of the ~6.2-GB twin.  A process-based vector environment could replicate the model and dramatically  increase memory pressure.
* With the default five environments and n_steps=1024, one PPO rollout contains 5,120 transitions.  The default 102,400 steps is exactly 20 full rollouts.
  The default validation interval, 20,480 steps, is exactly four rollouts.

Outputs
-------
The run directory contains:
* ``training_config.json``              exact run configuration and versions
* ``validation_learning_curves.csv``    checkpoint-level means/std/SE
* ``validation_episode_metrics.csv``    per-episode validation measurements

The ``*_se`` columns describe variation across validation episodes within one
trained seed.  For the paper's primary uncertainty bars, aggregate independent
training seeds rather than treating within-seed episodes as independent model
replicates.
* ``maskable_ppo_best.zip``             best validation-return checkpoint
* ``best_vecnormalize.pkl``             normalization state matching best model
* ``best_checkpoint.json``              best-checkpoint metadata
* ``maskable_ppo_<actual_steps>.zip``   final policy
* ``vecnormalize.pkl``                  normalization state matching final model
* ``tb/``                               TensorBoard logs when available/enabled

Important assumption
--------------------
``COMMAGPolicyEnv.counterfactual(action)`` must be side-effect free.  The
validation logic relies on that contract when comparing admissible actions
before executing the PPO-selected action.
"""

from __future__ import annotations

import argparse
import json
import platform
import random
import time
from importlib.metadata import PackageNotFoundError, version as package_version
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import MaskablePPO

from art_ran_commag.env import COMMAGPolicyEnv
from art_ran_commag.twin import COMMAGResponseModel


# -----------------------------------------------------------------------------
# Experiment definition
# -----------------------------------------------------------------------------

TRAIN_SCENARIOS = [
    "nominal",
    "urllc_burst",
    "embb_surge",
    "channel_shift",
    "mixed",
]

# Validation/model-selection scenarios only.
# mobility_shift remains untouched here and is reserved for final evaluation.
VAL_SCENARIOS = [
    "nominal",
    "urllc_burst",
    "embb_surge",
    "channel_shift",
]

EXPECTED_OBS_DIM = 29
TRAIN_ENV_SEED_STRIDE = 101

# Fixed PPO settings used by the current v0.6.2 experimental design.
LEARNING_RATE = 1.5e-4
GAMMA = 0.99
GAE_LAMBDA = 0.95
CLIP_RANGE = 0.15
ENT_COEF = 0.003
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
TARGET_KL = 0.02
POLICY_HIDDEN = [128, 128]


# -----------------------------------------------------------------------------
# Small utilities
# -----------------------------------------------------------------------------

def _installed_version(name: str) -> str | None:
    """Return an installed package version without making it a hard dependency."""
    try:
        return package_version(name)
    except PackageNotFoundError:
        return None


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def _finite_stats(values) -> dict[str, float | int]:
    """Return n/mean/std/SE after dropping non-finite values."""
    x = np.asarray(values, dtype=float)
    x = x[np.isfinite(x)]

    if len(x) == 0:
        return {
            "n": 0,
            "mean": np.nan,
            "std": np.nan,
            "se": np.nan,
        }

    if len(x) == 1:
        return {
            "n": 1,
            "mean": float(x[0]),
            "std": np.nan,
            "se": np.nan,
        }

    std = float(x.std(ddof=1))
    return {
        "n": int(len(x)),
        "mean": float(x.mean()),
        "std": std,
        "se": float(std / np.sqrt(len(x))),
    }


def _check_optional_logging(progress_requested: bool, tensorboard_requested: bool):
    """Disable optional UI/logging features gracefully if extras are unavailable."""
    progress_ok = progress_requested
    tensorboard_ok = tensorboard_requested

    if progress_requested:
        try:
            import rich  # noqa: F401
            import tqdm  # noqa: F401
        except ImportError:
            progress_ok = False
            print(
                "[warning] progress bar disabled because 'rich' and/or 'tqdm' "
                "is not installed."
            )

    if tensorboard_requested:
        try:
            import tensorboard  # noqa: F401
        except ImportError:
            tensorboard_ok = False
            print(
                "[warning] TensorBoard logging disabled because 'tensorboard' "
                "is not installed."
            )

    return progress_ok, tensorboard_ok


# -----------------------------------------------------------------------------
# Validation episode
# -----------------------------------------------------------------------------

def evaluate_episode(
    model: MaskablePPO,
    vecnorm: VecNormalize,
    env: COMMAGPolicyEnv,
    seed: int,
) -> dict[str, float]:
    """Evaluate one deterministic episode on a cached raw validation environment.

    The PPO model was trained on normalized observations, while ``env`` is a raw
    Gymnasium environment.  Therefore observations are transformed using the
    *current training VecNormalize statistics* before ``model.predict``.  Calling
    ``normalize_obs`` directly does not update those statistics.

    The environment's ``counterfactual`` method is used only for one-step
    diagnostics.  It must not mutate the environment state.
    """

    obs, _ = env.reset(seed=seed)
    rows: list[dict[str, float]] = []
    done = False

    while not done:
        mask = np.asarray(env.action_masks(), dtype=bool)
        valid = np.flatnonzero(mask)

        if len(valid) == 0:
            raise RuntimeError(
                "Validation encountered a state with no admissible action. "
                "The strict environment/mask contract has been violated."
            )

        # One-step diagnostic oracle over the currently admissible actions.
        # This is not a trajectory-level oracle and is never exposed to PPO.
        oracle = [env.counterfactual(int(a)) for a in valid]
        oracle_rewards = np.asarray([x[0] for x in oracle], dtype=float)

        if not np.all(np.isfinite(oracle_rewards)):
            raise RuntimeError("Non-finite counterfactual reward encountered.")

        best = float(oracle_rewards.max())
        worst = float(oracle_rewards.min())

        # A state is "jointly feasible" if at least one admissible action can
        # satisfy all empirically calibrated slice QoS targets immediately.
        feasible = any(bool(x[1]["all_qos_met"]) for x in oracle)

        obs_n = vecnorm.normalize_obs(
            np.asarray(obs, dtype=np.float32)[None, :]
        )[0]

        action, _ = model.predict(
            obs_n,
            deterministic=True,
            action_masks=mask,
        )
        action = int(np.asarray(action).item())

        if not mask[action]:
            raise RuntimeError(
                f"MaskablePPO selected inadmissible action index {action}."
            )

        obs, reward, terminated, truncated, info = env.step(action)
        reward = float(reward)

        if best > worst + 1e-8:
            oracle_score = (reward - worst) / (best - worst)
        else:
            oracle_score = 1.0

        qos_violations = np.asarray(info["qos_violations"], dtype=bool)
        throughput = np.asarray(info["throughput"], dtype=float)

        if qos_violations.shape[0] < 3 or throughput.shape[0] < 3:
            raise RuntimeError(
                "Environment info must expose three slice QoS indicators and "
                "three slice throughput values."
            )

        rows.append(
            {
                "reward": reward,
                "regret": float(best - reward),
                "oracle_score": float(np.clip(oracle_score, 0.0, 1.0)),
                "feasible": float(feasible),
                "feasible_success": float(
                    feasible and bool(info["all_qos_met"])
                ),
                "urllc_violation": float(qos_violations[2]),
                "all_qos": float(bool(info["all_qos_met"])),
                "churn": float(info["action_churn"]),
                "embb_throughput_mbps": float(throughput[0]),
                "mmtc_throughput_mbps": float(throughput[1]),
                "urllc_throughput_mbps": float(throughput[2]),
            }
        )

        done = bool(terminated or truncated)

    d = pd.DataFrame(rows)
    feasible_count = float(d["feasible"].sum())

    return {
        "return": float(d["reward"].sum()),
        "mean_reward": float(d["reward"].mean()),
        "regret": float(d["regret"].mean()),
        "oracle_score": float(d["oracle_score"].mean()),
        "fcsr": (
            float(d["feasible_success"].sum() / feasible_count)
            if feasible_count > 0
            else np.nan
        ),
        "feasible_rate": float(d["feasible"].mean()),
        "urllc_violation": float(d["urllc_violation"].mean()),
        "all_qos": float(d["all_qos"].mean()),
        "churn": float(d["churn"].mean()),
        "embb_throughput_mbps": float(d["embb_throughput_mbps"].mean()),
        "mmtc_throughput_mbps": float(d["mmtc_throughput_mbps"].mean()),
        "urllc_throughput_mbps": float(d["urllc_throughput_mbps"].mean()),
    }


# -----------------------------------------------------------------------------
# Rollout-boundary validation callback
# -----------------------------------------------------------------------------

class ValidationCallback(BaseCallback):
    """Validate at completed-rollout boundaries and save the best checkpoint.

    ``BaseCallback._on_step`` is called while SB3 is collecting a rollout, i.e.
    *before* the PPO update corresponding to that rollout.  To avoid attaching a
    validation point to a policy that has not yet consumed the reported samples,
    this callback evaluates from ``_on_rollout_start``.  Except at t=0, that hook
    occurs after the previous rollout has been optimized.

    Model selection uses the unweighted mean of raw episode return across the
    validation scenarios.  All scenarios have the same horizon and reward
    definition, and each scenario receives equal weight.
    """

    def __init__(
        self,
        vecnorm: VecNormalize,
        eval_envs: dict[str, COMMAGPolicyEnv],
        run_dir: Path,
        every: int = 20480,
        n_eps: int = 8,
        eval_seed_base: int = 1000,
    ):
        super().__init__()

        self.vecnorm = vecnorm
        self.eval_envs = eval_envs
        self.run_dir = Path(run_dir)
        self.every = int(every)
        self.n_eps = int(n_eps)
        self.eval_seed_base = int(eval_seed_base)

        self.aggregate_csv = self.run_dir / "validation_learning_curves.csv"
        self.episode_csv = self.run_dir / "validation_episode_metrics.csv"

        self.next_eval = self.every
        self.last_eval_timestep: int | None = None
        self.best_score = -np.inf
        self.rows: list[dict] = []
        self.episode_rows: list[dict] = []

    def _on_step(self) -> bool:
        # Required by BaseCallback.  Validation itself is intentionally done at
        # rollout boundaries in _on_rollout_start().
        return True

    def _on_rollout_start(self) -> None:
        timesteps = int(self.model.num_timesteps)

        # t=0 precedes any training update, so do not label it as a trained
        # validation checkpoint.  It can be evaluated separately if needed.
        if timesteps <= 0 or timesteps < self.next_eval:
            return

        self.evaluate_checkpoint(timesteps)

        # Rollout sizes can jump over a requested interval.  Move the next
        # threshold forward until it is strictly greater than current time.
        while self.next_eval <= timesteps:
            self.next_eval += self.every

    def _write_outputs(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(self.rows).to_csv(self.aggregate_csv, index=False)
        pd.DataFrame(self.episode_rows).to_csv(self.episode_csv, index=False)

    def evaluate_checkpoint(self, timesteps: int) -> None:
        """Evaluate the current policy and optionally update the best checkpoint."""
        timesteps = int(timesteps)

        if self.last_eval_timestep == timesteps:
            return

        checkpoint_rows = []

        for scenario in VAL_SCENARIOS:
            env = self.eval_envs[scenario]
            results = []

            for episode_index in range(self.n_eps):
                eval_seed = self.eval_seed_base + episode_index
                result = evaluate_episode(
                    model=self.model,
                    vecnorm=self.vecnorm,
                    env=env,
                    seed=eval_seed,
                )
                results.append(result)

                self.episode_rows.append(
                    {
                        "timesteps": timesteps,
                        "scenario": scenario,
                        "episode_index": episode_index,
                        "eval_seed": eval_seed,
                        **result,
                    }
                )

            row = {
                "timesteps": timesteps,
                "scenario": scenario,
            }

            for key in results[0]:
                stats = _finite_stats([r[key] for r in results])
                row[f"{key}_n"] = stats["n"]
                row[f"{key}_mean"] = stats["mean"]
                row[f"{key}_std"] = stats["std"]
                row[f"{key}_se"] = stats["se"]

            self.rows.append(row)
            checkpoint_rows.append(row)

        self.last_eval_timestep = timesteps
        self._write_outputs()

        current = pd.DataFrame(checkpoint_rows)

        print(f"\n=== VALIDATION AFTER PPO UPDATE @ {timesteps} STEPS ===")
        print(
            current[
                [
                    "scenario",
                    "return_mean",
                    "oracle_score_mean",
                    "regret_mean",
                    "fcsr_mean",
                    "feasible_rate_mean",
                    "urllc_violation_mean",
                    "all_qos_mean",
                    "embb_throughput_mbps_mean",
                ]
            ]
            .round(4)
            .to_string(index=False)
        )

        scenario_returns = current["return_mean"].to_numpy(dtype=float)
        scenario_returns = scenario_returns[np.isfinite(scenario_returns)]
        selection_score = (
            float(scenario_returns.mean())
            if len(scenario_returns)
            else -np.inf
        )

        print(
            "[validation] selection score "
            f"(mean raw return across scenarios) = {selection_score:.6f}"
        )

        if selection_score > self.best_score:
            self.best_score = selection_score

            self.model.save(self.run_dir / "maskable_ppo_best")
            self.vecnorm.save(self.run_dir / "best_vecnormalize.pkl")

            _write_json(
                self.run_dir / "best_checkpoint.json",
                {
                    "timesteps": timesteps,
                    "selection_metric": "mean_raw_validation_return",
                    "selection_score": selection_score,
                    "scenario_return_mean": {
                        str(r["scenario"]): float(r["return_mean"])
                        for r in checkpoint_rows
                    },
                },
            )

            print(
                f"[validation] new best checkpoint saved at {timesteps} steps"
            )


# -----------------------------------------------------------------------------
# Main training routine
# -----------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Train the ART-RAN v0.6.2 MaskablePPO downstream actuator on the "
            "COMMAG-grounded environment."
        )
    )

    parser.add_argument(
        "--prepared",
        default="prepared_v062",
        help="Directory containing commag_wide.parquet, twin, action catalog, and calibration.",
    )
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Output directory. Default: runs/v062_seed<seed>.",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=102400,
        help=(
            "Requested training transitions. With 5 envs and n_steps=1024, "
            "102400 is exactly 20 PPO rollouts."
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help=(
            "Pseudo-random training seed. It has no statistical meaning; use "
            "multiple independent values (e.g., 0..4) for final experiments."
        ),
    )
    parser.add_argument(
        "--eval-every",
        type=int,
        default=20480,
        help="Validation interval in environment transitions.",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=8,
        help="Fixed-seed validation episodes per scenario at each checkpoint.",
    )
    parser.add_argument(
        "--eval-seed-base",
        type=int,
        default=1000,
        help="Base seed for fixed validation trajectories; keep identical across training seeds.",
    )
    parser.add_argument(
        "--n-steps",
        type=int,
        default=1024,
        help="PPO rollout length per vector environment.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=512,
        help="PPO minibatch size.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=160,
        help="Episode horizon used by both training and validation environments.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="SB3 policy device, e.g. cpu, cuda, or auto.",
    )
    parser.add_argument(
        "--twin-n-jobs",
        type=int,
        default=1,
        help=(
            "ExtraTrees inference threads. One is recommended for repeated "
            "single-row predictions; benchmark before changing."
        ),
    )
    parser.add_argument(
        "--allow-existing-run-dir",
        action="store_true",
        help=(
            "Allow writing into a non-empty run directory. For independent "
            "paper runs, prefer a fresh directory."
        ),
    )
    parser.add_argument(
        "--no-progress-bar",
        action="store_true",
        help="Disable the SB3 tqdm/rich progress bar.",
    )
    parser.add_argument(
        "--no-tensorboard",
        action="store_true",
        help="Disable TensorBoard logging.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # Basic argument validation
    # ------------------------------------------------------------------
    if args.steps <= 0:
        raise ValueError("--steps must be positive")
    if args.n_steps <= 0:
        raise ValueError("--n-steps must be positive")
    if args.batch_size <= 1:
        raise ValueError("--batch-size must be > 1")
    if args.horizon <= 0:
        raise ValueError("--horizon must be positive")
    if args.eval_every <= 0:
        raise ValueError("--eval-every must be positive")
    if args.eval_episodes <= 0:
        raise ValueError("--eval-episodes must be positive")
    if args.twin_n_jobs == 0:
        raise ValueError("--twin-n-jobs cannot be zero")

    prep = Path(args.prepared)
    run = (
        Path(args.run_dir)
        if args.run_dir is not None
        else Path(f"runs/v062_seed{args.seed}")
    )

    required = [
        prep / "commag_wide.parquet",
        prep / "action_catalog.csv",
        prep / "calibration.json",
        prep / "commag_twin.joblib",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing required prepared artifacts:\n  " + "\n  ".join(missing)
        )

    if run.exists() and any(run.iterdir()):
        if not args.allow_existing_run_dir:
            raise RuntimeError(
                f"Run directory is not empty: {run}. "
                "Use a fresh directory, remove the old smoke run, or pass "
                "--allow-existing-run-dir intentionally."
            )
        print(
            f"[warning] writing into existing non-empty run directory: {run}"
        )
    run.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Optional logging features
    # ------------------------------------------------------------------
    progress_bar, tensorboard_enabled = _check_optional_logging(
        progress_requested=not args.no_progress_bar,
        tensorboard_requested=not args.no_tensorboard,
    )
    tensorboard_log = str(run / "tb") if tensorboard_enabled else None

    # ------------------------------------------------------------------
    # Load data, action catalog, calibration, and empirical response twin
    # ------------------------------------------------------------------
    print("[setup] loading COMMAG traces...")
    df = pd.read_parquet(prep / "commag_wide.parquet")

    if "split" not in df.columns:
        raise KeyError("commag_wide.parquet does not contain the required 'split' column")

    train = df[df["split"] == "train"].reset_index(drop=True)
    val = df[df["split"] == "val"].reset_index(drop=True)

    if len(train) == 0 or len(val) == 0:
        raise RuntimeError(
            f"Prepared split is empty: train={len(train)}, val={len(val)}"
        )

    actions = pd.read_csv(prep / "action_catalog.csv").to_numpy(dtype=np.int32)
    if actions.ndim != 2 or actions.shape[1] != 3 or len(actions) < 2:
        raise RuntimeError(
            f"Expected action_catalog.csv to contain >=2 three-slice actions; got {actions.shape}."
        )

    cal = json.loads((prep / "calibration.json").read_text())
    for key in ["resource_budget", "max_change_default"]:
        if key not in cal:
            raise KeyError(f"calibration.json is missing required key: {key}")

    action_sums = actions.sum(axis=1)
    if not np.all(action_sums == int(cal["resource_budget"])):
        raise RuntimeError(
            "Action catalog is inconsistent with calibrated resource budget. "
            f"Action sums={np.unique(action_sums).tolist()}, "
            f"resource_budget={cal['resource_budget']}"
        )

    print("[setup] loading empirical twin...")
    twin = COMMAGResponseModel.load(prep / "commag_twin.joblib")

    if not hasattr(twin.model, "n_jobs"):
        raise RuntimeError(
            "Loaded twin estimator does not expose n_jobs; expected ExtraTreesRegressor."
        )

    # Single-row predictions dominate the environment.  A single ExtraTrees
    # inference thread generally avoids joblib scheduling overhead here.
    twin.model.n_jobs = int(args.twin_n_jobs)

    print(
        f"[setup] train rows={len(train):,}, val rows={len(val):,}, "
        f"actions={len(actions)}"
    )
    print(f"[setup] train scenarios={TRAIN_SCENARIOS}")
    print(f"[setup] validation scenarios={VAL_SCENARIOS}")
    print(f"[setup] resource_budget={cal['resource_budget']}")
    print(f"[setup] max_change={cal['max_change_default']}")
    print(f"[setup] twin inference n_jobs={twin.model.n_jobs}")

    # ------------------------------------------------------------------
    # Reproducibility
    # ------------------------------------------------------------------
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # ------------------------------------------------------------------
    # Training environments
    # ------------------------------------------------------------------
    def make_env(scenario: str, seed: int):
        def _make():
            return COMMAGPolicyEnv(
                train,
                twin,
                actions,
                cal,
                scenario=scenario,
                horizon=args.horizon,
                seed=seed,
            )

        return _make

    env_fns = [
        make_env(
            scenario=scenario,
            seed=args.seed + i * TRAIN_ENV_SEED_STRIDE,
        )
        for i, scenario in enumerate(TRAIN_SCENARIOS)
    ]

    vec_env = DummyVecEnv(env_fns)
    vec_env = VecNormalize(
        vec_env,
        norm_obs=True,
        norm_reward=True,
        clip_obs=10.0,
        clip_reward=10.0,
        gamma=GAMMA,
    )

    obs_shape = tuple(vec_env.observation_space.shape)
    if obs_shape != (EXPECTED_OBS_DIM,):
        raise RuntimeError(
            f"Expected {EXPECTED_OBS_DIM}-D observation for v0.6.2; got {obs_shape}."
        )

    action_n = getattr(vec_env.action_space, "n", None)
    if action_n != len(actions):
        raise RuntimeError(
            f"Environment action_space.n={action_n}, catalog size={len(actions)}."
        )

    # ------------------------------------------------------------------
    # Cached validation environments
    # ------------------------------------------------------------------
    print("[setup] constructing cached validation environments...")
    eval_envs = {
        scenario: COMMAGPolicyEnv(
            val,
            twin,
            actions,
            cal,
            scenario=scenario,
            horizon=args.horizon,
            seed=args.eval_seed_base,
        )
        for scenario in VAL_SCENARIOS
    }

    # ------------------------------------------------------------------
    # Rollout/batch consistency
    # ------------------------------------------------------------------
    n_envs = len(TRAIN_SCENARIOS)
    rollout_size = args.n_steps * n_envs

    if rollout_size % args.batch_size != 0:
        raise ValueError(
            f"rollout_size={rollout_size} must be divisible by "
            f"batch_size={args.batch_size} for clean minibatches."
        )

    if args.steps % rollout_size != 0:
        expected_actual = int(np.ceil(args.steps / rollout_size) * rollout_size)
        print(
            f"[warning] requested steps ({args.steps}) are not a multiple of one "
            f"rollout ({rollout_size}). SB3 will complete the final rollout, so "
            f"actual steps will be at least {expected_actual}."
        )

    if args.eval_every % rollout_size != 0:
        print(
            f"[warning] eval-every={args.eval_every} is not a multiple of the "
            f"rollout size {rollout_size}. Validation will occur at the first "
            "completed rollout boundary after each threshold."
        )

    print(
        f"[setup] requested device={args.device}, n_envs={n_envs}, "
        f"n_steps={args.n_steps}, rollout_size={rollout_size}, "
        f"batch_size={args.batch_size}, requested_steps={args.steps}"
    )

    # ------------------------------------------------------------------
    # Validation callback
    # ------------------------------------------------------------------
    callback = ValidationCallback(
        vecnorm=vec_env,
        eval_envs=eval_envs,
        run_dir=run,
        every=args.eval_every,
        n_eps=args.eval_episodes,
        eval_seed_base=args.eval_seed_base,
    )

    # ------------------------------------------------------------------
    # Maskable PPO
    # ------------------------------------------------------------------
    model = MaskablePPO(
        "MlpPolicy",
        vec_env,
        learning_rate=LEARNING_RATE,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        gamma=GAMMA,
        gae_lambda=GAE_LAMBDA,
        clip_range=CLIP_RANGE,
        ent_coef=ENT_COEF,
        vf_coef=VF_COEF,
        max_grad_norm=MAX_GRAD_NORM,
        target_kl=TARGET_KL,
        policy_kwargs={
            "activation_fn": torch.nn.Tanh,
            "net_arch": {
                "pi": POLICY_HIDDEN,
                "vf": POLICY_HIDDEN,
            },
        },
        seed=args.seed,
        verbose=1,
        device=args.device,
        tensorboard_log=tensorboard_log,
    )

    # ------------------------------------------------------------------
    # Persist exact run configuration before training starts
    # ------------------------------------------------------------------
    config = {
        "art_ran_version": "v0.6.2",
        "python": platform.python_version(),
        "versions": {
            "numpy": np.__version__,
            "pandas": pd.__version__,
            "torch": torch.__version__,
            "stable_baselines3": _installed_version("stable-baselines3"),
            "sb3_contrib": _installed_version("sb3-contrib"),
            "scikit_learn": _installed_version("scikit-learn"),
            "gymnasium": _installed_version("gymnasium"),
        },
        "seed": args.seed,
        "training_env_seeds": {
            scenario: args.seed + i * TRAIN_ENV_SEED_STRIDE
            for i, scenario in enumerate(TRAIN_SCENARIOS)
        },
        "eval_seed_base": args.eval_seed_base,
        "eval_seeds": [
            args.eval_seed_base + i for i in range(args.eval_episodes)
        ],
        "steps_requested": args.steps,
        "n_envs": n_envs,
        "n_steps": args.n_steps,
        "rollout_size": rollout_size,
        "batch_size": args.batch_size,
        "horizon": args.horizon,
        "eval_every": args.eval_every,
        "eval_episodes": args.eval_episodes,
        "device_requested": args.device,
        "twin_n_jobs": args.twin_n_jobs,
        "train_scenarios": TRAIN_SCENARIOS,
        "val_scenarios": VAL_SCENARIOS,
        "held_out_scenario_not_used": "mobility_shift",
        "observation_dim": EXPECTED_OBS_DIM,
        "action_catalog": actions.tolist(),
        "resource_budget": int(cal["resource_budget"]),
        "max_change_default": int(cal["max_change_default"]),
        "ppo": {
            "learning_rate": LEARNING_RATE,
            "gamma": GAMMA,
            "gae_lambda": GAE_LAMBDA,
            "clip_range": CLIP_RANGE,
            "ent_coef": ENT_COEF,
            "vf_coef": VF_COEF,
            "max_grad_norm": MAX_GRAD_NORM,
            "target_kl": TARGET_KL,
            "activation": "Tanh",
            "actor_hidden": POLICY_HIDDEN,
            "critic_hidden": POLICY_HIDDEN,
            "norm_obs": True,
            "norm_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
        "validation_model_selection_metric": "mean_raw_validation_return",
        "tensorboard_enabled": tensorboard_enabled,
        "progress_bar_enabled": progress_bar,
    }
    _write_json(run / "training_config.json", config)

    # ------------------------------------------------------------------
    # Learn
    # ------------------------------------------------------------------
    print("[train] starting MaskablePPO...")
    train_start = time.perf_counter()

    model.learn(
        total_timesteps=args.steps,
        callback=callback,
        progress_bar=progress_bar,
    )

    training_wall_seconds = time.perf_counter() - train_start
    actual_steps = int(model.num_timesteps)

    # The final update may terminate the SB3 loop without another
    # _on_rollout_start callback.  Always validate the actual final policy once.
    if callback.last_eval_timestep != actual_steps:
        callback.evaluate_checkpoint(actual_steps)

    # ------------------------------------------------------------------
    # Save final policy and normalization state
    # ------------------------------------------------------------------
    final_model_base = run / f"maskable_ppo_{actual_steps}"
    model.save(final_model_base)
    vec_env.save(run / "vecnormalize.pkl")

    config.update(
        {
            "steps_actual": actual_steps,
            "learn_wall_seconds_including_checkpoint_validation": training_wall_seconds,
            "effective_fps_including_checkpoint_validation": (
                float(actual_steps / training_wall_seconds)
                if training_wall_seconds > 0
                else np.nan
            ),
            "final_model": str(final_model_base.with_suffix(".zip")),
            "final_vecnormalize": str(run / "vecnormalize.pkl"),
            "best_model": str(run / "maskable_ppo_best.zip"),
            "best_vecnormalize": str(run / "best_vecnormalize.pkl"),
            "best_validation_score": float(callback.best_score),
            "best_checkpoint_metadata": str(run / "best_checkpoint.json"),
        }
    )
    _write_json(run / "training_config.json", config)

    print("\nTraining complete.")
    print(f"Requested steps: {args.steps}")
    print(f"Actual steps:    {actual_steps}")
    print(f"Final model:     {final_model_base.with_suffix('.zip')}")
    print(f"VecNormalize:    {run / 'vecnormalize.pkl'}")
    print(f"Best model:      {run / 'maskable_ppo_best.zip'}")
    print(f"Validation:      {run / 'validation_learning_curves.csv'}")
    print(f"Episode metrics: {run / 'validation_episode_metrics.csv'}")
    print(
        "Effective learn throughput (includes checkpoint validation time): "
        f"{config['effective_fps_including_checkpoint_validation']:.3f} transitions/s"
    )

    # Explicitly close raw validation envs and vectorized training env.
    for env in eval_envs.values():
        env.close()
    vec_env.close()


if __name__ == "__main__":
    main()
