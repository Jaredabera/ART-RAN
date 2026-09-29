from __future__ import annotations



import numpy as np

import pandas as pd



try:

    import gymnasium as gym

    from gymnasium import spaces

except ImportError as e:

    raise ImportError("Install with pip install -e '.[colab]'") from e



from .calibration import utilities, violations

from .reasoner import ReferenceReasoner





class COMMAGPolicyEnv(gym.Env):

    """

    Trace-driven semi-counterfactual COMMAG environment.



    Exogenous requested-PRB pressure, CQI, scheduler state, and inferred

    arrivals are taken from sequential COMMAG traces. The empirical twin

    predicts only immediate action-sensitive throughput and grant ratio.

    Queue state evolves through a physical balance equation, avoiding

    recursive learned next-buffer prediction.



    The Non-RT rApp does not directly select PRB allocations. It produces

    slice priorities and control constraints that define the admissible

    action set for the downstream Near-RT controller.

    """



    metadata = {"render_modes": []}



    def __init__(

        self,

        df,

        twin,

        actions,

        cal,

        scenario="mixed",

        horizon=160,

        non_rt_interval=20,

        seed=2027,

    ):

        super().__init__()



        self.df = df.reset_index(drop=True).copy()

        self.twin = twin

        self.actions = np.asarray(actions, dtype=np.int32)

        self.cal = cal



        self.scenario = str(scenario)

        self.horizon = int(horizon)

        self.interval = int(non_rt_interval)

        self.rng = np.random.default_rng(seed)



        if self.actions.ndim != 2 or self.actions.shape[1] != 3:

            raise ValueError(

                f"Expected action catalog of shape (N, 3), "

                f"got {self.actions.shape}"

            )



        if len(self.actions) == 0:

            raise ValueError("Action catalog is empty")



        self.reasoner = ReferenceReasoner(self.actions, self.cal)



        self.action_space = spaces.Discrete(len(self.actions))

        self.observation_space = spaces.Box(

            low=-10.0,

            high=10.0,

            shape=(29,),

            dtype=np.float32,

        )



        self._index_trajectories()



    # ------------------------------------------------------------------

    # Trajectory indexing

    # ------------------------------------------------------------------



    def _index_trajectories(self):
        """
        Index sequential COMMAG trajectories and pre-compute eligible
        episode starts for the configured scenario.

        A horizon-H episode requires H action rows plus one terminal
        observation row, hence at least H+1 samples from the same
        rf/tr/exp/bs trajectory.

        Scenario-specific environments use the scenario label only as an
        initial-condition constraint. After reset, the episode follows the
        recorded COMMAG trajectory naturally and may leave that regime.

        Mixed training retains the previous rule that mobility-shift samples
        are excluded from the complete H+1 window.
        """
        self.groups = []

        for key, g in self.df.groupby(
            ["rf", "tr", "exp", "bs"],
            sort=False,
        ):
            g = g.sort_values("time_bin").reset_index(drop=True)

            if len(g) >= self.horizon + 1:
                self.groups.append((key, g))

        if not self.groups:
            raise ValueError(
                f"No trajectories contain at least "
                f"{self.horizon + 1} samples"
            )

        # Scenario eligibility is static for this environment, so compute
        # it once instead of rescanning every trajectory on every reset.
        self._eligible_by_group = [
            self._eligible_starts(g) for _, g in self.groups
        ]

        # Pre-compute global-start sampling metadata. Sampling a trajectory
        # uniformly and then a start within it would over-represent groups
        # with relatively few eligible starts. The cumulative counts below
        # let reset() sample uniformly over the complete eligible-start set.
        self._eligible_counts = np.asarray(
            [len(starts) for starts in self._eligible_by_group],
            dtype=np.int64,
        )
        self._eligible_cumulative = np.cumsum(self._eligible_counts)
        self._total_eligible = int(self._eligible_counts.sum())

        if self._total_eligible == 0:
            raise ValueError(
                f"No eligible {self.horizon}-step episodes for "
                f"scenario='{self.scenario}'"
            )

    def _eligible_starts(self, g):
        """
        Return valid episode starts for the configured scenario.

        A horizon-H episode contains H control transitions and therefore
        requires H+1 rows from the same COMMAG trajectory.

        For a scenario-specific environment, the scenario constrains only
        the initial state. This models an episode initialized under the
        requested operating condition (e.g., URLLC burst) while allowing
        the subsequent H transitions to evolve according to the recorded
        trace.

        For mixed training, mobility_shift remains excluded from the full
        H+1 window, preserving the previous training-regime separation.
        """
        window_len = self.horizon + 1

        if len(g) < window_len:
            return np.empty(0, dtype=int)

        labels = g["scenario"].astype(str).to_numpy()
        max_start = len(g) - window_len
        candidate_starts = np.arange(max_start + 1, dtype=int)

        if self.scenario != "mixed":
            # Scenario-specific evaluation: constrain the initial state only.
            return candidate_starts[
                labels[candidate_starts] == self.scenario
            ].astype(int)

        # Mixed training: preserve the prior requirement that mobility-shift
        # samples do not appear anywhere in the H+1 episode window.
        invalid = (labels == "mobility_shift").astype(np.int32)
        cumulative = np.concatenate(
            [
                np.zeros(1, dtype=np.int64),
                np.cumsum(invalid, dtype=np.int64),
            ]
        )
        invalid_per_window = (
            cumulative[window_len:]
            - cumulative[:-window_len]
        )

        return np.flatnonzero(
            invalid_per_window == 0
        ).astype(int)

    # ------------------------------------------------------------------

    # Gymnasium interface

    # ------------------------------------------------------------------



    def reset(self, *, seed=None, options=None):

        super().reset(seed=seed)



        if seed is not None:

            self.rng = np.random.default_rng(seed)



        # Uniformly sample over all eligible episode starts. This avoids
        # the trajectory-first bias that over-represents trajectories with
        # relatively few eligible start positions.
        global_idx = int(
            self.rng.integers(self._total_eligible)
        )

        gi = int(
            np.searchsorted(
                self._eligible_cumulative,
                global_idx,
                side="right",
            )
        )

        previous_total = (
            0
            if gi == 0
            else int(self._eligible_cumulative[gi - 1])
        )
        local_idx = global_idx - previous_total

        starts = self._eligible_by_group[gi]
        self.key, self.g = self.groups[gi]
        self.pos = int(starts[local_idx])
        self.end = self.pos + self.horizon



        r = self._row()



        self.buffer = np.asarray(

            [

                r[f"dl_buffer [bytes]_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        self.tput = np.asarray(

            [

                r[f"tx_brate downlink [Mbps]_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        self.ratio = np.asarray(

            [

                r[f"ratio_granted_req_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        # Initialize the previous action using the nearest supported

        # allocation in the empirical action catalog.

        obs_action = np.asarray(

            [

                r[f"slice_prb_s{s}"]

                for s in range(3)

            ],

            dtype=int,

        )



        distance = np.square(

            self.actions - obs_action[None, :]

        ).sum(axis=1)



        self.prev_action = self.actions[

            int(np.argmin(distance))

        ].copy()



        self.age = 10**9

        self.policy = None

        self._update_policy()



        info = {

            "scenario": str(r["scenario"]),

            "trajectory": self.key,

            "start_pos": int(self.pos),

        }



        return self._obs(), info



    def _row(self):

        return self.g.iloc[self.pos]



    # ------------------------------------------------------------------

    # Non-RT rApp policy

    # ------------------------------------------------------------------



    def _update_policy(self):

        if self.policy is None or self.age >= self.interval:

            u = utilities(

                self.tput,

                self.ratio,

                self.buffer,

                self.cal,

            )



            self.policy = self.reasoner.propose(

                self.buffer,

                self._row(),

                u,

            )



            self.age = 0



    # ------------------------------------------------------------------

    # Near-RT admissible action set

    # ------------------------------------------------------------------



    def action_masks(self):

        """

        Return the actions permitted by the current rApp policy.



        No silent relaxation is allowed. If the rApp generates a policy

        for which no supported Near-RT action exists, the policy is

        operationally infeasible and must be exposed explicitly.

        """

        min_alloc = np.asarray(

            self.policy.min_alloc,

            dtype=np.int32,

        )



        mask = np.all(

            self.actions >= min_alloc[None, :],

            axis=1,

        )



        change = np.abs(

            self.actions - self.prev_action[None, :]

        ).sum(axis=1)



        mask &= change <= self.policy.max_change



        if not np.any(mask):

            raise RuntimeError(

                "rApp generated an infeasible policy: "

                f"scenario={self.scenario}, "

                f"trajectory={self.key}, "

                f"pos={self.pos}, "

                f"min_alloc={min_alloc.tolist()}, "

                f"max_change={self.policy.max_change}, "

                f"prev_action={self.prev_action.tolist()}"

            )



        return mask.astype(bool)



    # ------------------------------------------------------------------

    # Observation

    # ------------------------------------------------------------------



    def _obs(self):

        r = self._row()



        u = utilities(

            self.tput,

            self.ratio,

            self.buffer,

            self.cal,

        )



        targets = np.asarray(

            [

                self.cal[f"qos_target_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        deficit = np.maximum(targets - u, 0.0)



        features = []



        for s in range(3):

            features += [

                np.log1p(self.buffer[s])

                / np.log1p(self.cal[f"buffer_ref_s{s}"]),



                float(r[f"sum_requested_prbs_s{s}"])

                / self.cal[f"req_ref_s{s}"],



                float(self.tput[s])

                / self.cal[f"tput_ref_s{s}"],



                float(self.ratio[s]),



                (

                    float(r[f"dl_cqi_s{s}"])

                    / self.cal[f"cqi_ref_s{s}"]

                    if pd.notna(r[f"dl_cqi_s{s}"])

                    else 0.0

                ),



                float(self.prev_action[s])

                / max(

                    float(self.cal["resource_budget"]),

                    1.0,

                ),



                float(deficit[s]),

            ]



        total_req = sum(

            float(r[f"sum_requested_prbs_s{s}"])

            for s in range(3)

        )



        total_ref = sum(

            self.cal[f"req_ref_s{s}"]

            for s in range(3)

        )



        features.append(

            total_req / max(total_ref, 1e-6)

        )



        ran_state = np.asarray(

            features,

            dtype=np.float32,

        )



        policy_state = np.asarray(

            self.policy.vector(

                self.cal["resource_budget"]

            ),

            dtype=np.float32,

        )



        obs = np.concatenate(

            [ran_state, policy_state]

        )



        if obs.shape != (29,):

            raise RuntimeError(

                f"Expected 29-D observation, got {obs.shape}: "

                f"RAN={ran_state.shape}, "

                f"policy={policy_state.shape}"

            )



        return np.clip(

            obs,

            -10.0,

            10.0,

        ).astype(np.float32)



    # ------------------------------------------------------------------

    # Counterfactual RAN response

    # ------------------------------------------------------------------



    def counterfactual(self, action_idx):

        action_idx = int(action_idx)



        if not 0 <= action_idx < len(self.actions):

            raise IndexError(

                f"Invalid action index {action_idx}; "

                f"catalog size={len(self.actions)}"

            )



        action = self.actions[action_idx].copy()

        r = self._row()



        tput, ratio = self.twin.predict_from_row(

            r,

            self.buffer,

            action,

        )



        dt = float(

            r.get(

                "dt_sec",

                self.cal.get(

                    "time_bin_ms",

                    250,

                ) / 1000.0,

            )

        )



        arrivals = np.asarray(

            [

                r[f"arrival_bytes_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        service = tput * 1e6 / 8.0 * dt



        next_buffer = (

            np.maximum(

                self.buffer - service,

                0.0,

            )

            + np.maximum(

                arrivals,

                0.0,

            )

        )



        u = utilities(

            tput,

            ratio,

            next_buffer,

            self.cal,

        )



        v = violations(

            u,

            self.cal,

        )



        churn = float(

            np.abs(

                action - self.prev_action

            ).sum()

            / max(

                float(self.cal["resource_budget"]),

                1.0,

            )

        )



        score = float(

            np.dot(

                self.policy.priority,

                u,

            )

        )



        reward = (

            2.0 * score

            - 1.10 * v[0]

            - 0.90 * v[1]

            - 1.55 * v[2]

            - 0.04 * churn

        )



        targets = np.asarray(

            [

                self.cal[f"qos_target_s{s}"]

                for s in range(3)

            ],

            dtype=float,

        )



        qos_violations = u < targets



        info = {

            "action": action,

            "throughput": tput,

            "grant_ratio": ratio,

            "next_buffer": next_buffer,

            "utility": u,

            "violation_deficit": v,



            # Preferred terminology.

            "qos_violations": qos_violations,

            "all_qos_met": bool(

                np.all(u >= targets)

            ),



            # Backward-compatible aliases for existing scripts.

            "sla_violations": qos_violations,

            "all_sla_met": bool(

                np.all(u >= targets)

            ),



            "action_churn": churn,

            "rapp_priority": (

                self.policy.priority.copy()

            ),

            "rapp_min_alloc": (

                self.policy.min_alloc.copy()

            ),

            "rapp_max_change": (

                self.policy.max_change

            ),

        }



        return float(reward), info



    # ------------------------------------------------------------------

    # Environment transition

    # ------------------------------------------------------------------



    def step(self, action_idx):

        action_idx = int(action_idx)



        if not 0 <= action_idx < len(self.actions):

            raise IndexError(

                f"Invalid action index {action_idx}"

            )



        # Enforce the rApp policy envelope. MaskablePPO should already

        # respect this mask; this check prevents accidental bypass by

        # evaluation or external control code.

        valid_mask = self.action_masks()



        if not valid_mask[action_idx]:

            raise ValueError(

                "Attempted to execute an action outside the "

                "current rApp policy envelope: "

                f"action={self.actions[action_idx].tolist()}, "

                f"scenario={self.scenario}, "

                f"trajectory={self.key}, "

                f"pos={self.pos}"

            )



        r = self._row()

        scenario_now = str(r["scenario"])



        reward, info = self.counterfactual(

            action_idx

        )



        self.buffer = info[

            "next_buffer"

        ].copy()



        self.tput = info[

            "throughput"

        ].copy()



        self.ratio = info[

            "grant_ratio"

        ].copy()



        self.prev_action = info[

            "action"

        ].copy()



        self.pos += 1

        self.age += 1



        truncated = (

            self.pos >= self.end

            or self.pos >= len(self.g) - 1

        )



        terminated = False



        # Scenario-specific episodes are constrained only at reset. The

        # current trace state may naturally move into another scenario.

        self._update_policy()



        info["scenario"] = scenario_now

        info["trajectory"] = self.key

        info["position"] = int(self.pos)



        return (

            self._obs(),

            reward,

            terminated,

            truncated,

            info,

        )