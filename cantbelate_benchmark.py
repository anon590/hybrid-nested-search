"""
"Can't Be Late" testbed for the LLM-driven Hybrid Nested Search (paper.pdf, Sec. 6).

Cloud-scheduling analog of cleanup_benchmark.py. The artifact under search is a
per-step decision rule
    decide(obs, params) -> int in {0:NONE, 1:SPOT, 2:ON_DEMAND}
that, at each tick of a deadline-constrained job, chooses a cheap-but-preemptible
SPOT instance, a reliable ON_DEMAND instance, or waiting (NONE). The STRUCTURE
(how `obs` maps to the choice) is proposed by the LLM; the continuous holes
`params` (slack buffers, pressure thresholds, hysteresis bands) are tuned by the
inner CMA-ES.

The harness fitness is the total dollar COST of finishing the job (spot ~$0.97/h,
on-demand ~$3.06/h; waiting is free but burns deadline budget). CMA-ES is a
*minimizer*, so evaluate() returns the LOSS = cost directly (lower is better),
exactly like benchmarks.py. The simulator applies a STRONG GUARANTEE that forces
ON_DEMAND whenever the deadline would otherwise be missed, so the deadline is a
HARD constraint the policy cannot violate -- the objective is cleanly cost only.
Brittle rules (any raised exception, e.g. running off the trace) score +inf.

OBJECTIVE-BLINDNESS: the rule sees only the CURRENT per-step state (remaining
work, slack to deadline, restart overhead, whether spot is up right now). It does
NOT see the trace's future spot-availability pattern or the prices, so it cannot
analytically place the buffers/thresholds that trade "gamble on spot returning"
against "commit to on-demand and stop paying relaunch overhead". That is the
tuning gap Delta the hybrid arm de-aliases.

Three "regimes" play the role of the meta-optimizer's function suite. They share
one deadline (70h over a 48h job) and one trace pool, and differ ONLY in the
restart-overhead the simulator charges per instance switch -- which is exactly
what sets how much the (non-obvious) thrash-vs-commit threshold matters:

  cheap_restart  -- overhead 0.02h: relaunching spot is almost free, so the
                    spot-greedy default is already near-optimal (Delta ~ 0 tie).
  moderate       -- overhead 0.20h: textbook setting.
  costly_restart -- overhead 0.40h: each switch wastes real compute, so the
                    optimal commit threshold is delicate and non-obvious
                    (large Delta -> hybrid wins).
"""

from __future__ import annotations

import os
import sys
import glob
import json
import logging

import numpy as np

# --------------------------------------------------------------------------- #
# Locate the GEPA repo's Can't Be Late simulator (sky_spot) + downloaded traces.
# --------------------------------------------------------------------------- #
_GEPA_ROOT = os.environ.get("GEPA_ROOT", os.path.expanduser("~/gepa"))
_SIM_DIR = os.path.join(
    _GEPA_ROOT, "examples", "adrs", "can_be_late", "utils", "simulator")
if not os.path.isdir(_SIM_DIR):
    raise FileNotFoundError(
        f"Can't Be Late simulator not found under {_SIM_DIR!r}. "
        "Set GEPA_ROOT to your gepa checkout.")
if _SIM_DIR not in sys.path:
    sys.path.insert(0, _SIM_DIR)

# The strong-guarantee override logs a WARNING on essentially every tick; silence
# the whole simulator package so a 100-eval inner loop does not flood the console.
logging.getLogger("sky_spot").setLevel(logging.CRITICAL)

from sky_spot.env import TraceEnv               # noqa: E402  (after sys.path setup)
from sky_spot.task import SingleTask            # noqa: E402
from sky_spot.strategies.strategy import Strategy  # noqa: E402
from sky_spot.utils import ClusterType          # noqa: E402

_TRACE_ROOT = os.path.join(
    _SIM_DIR, "real", "ddl=search+task=48+overhead=0.20", "real")

TASK_DURATION_H = 48.0          # job compute time (hours), fixed across regimes
DEADLINE_H = float(os.environ.get("CBL_DEADLINE_H", "70"))  # fixed deadline
N_TRACES = int(os.environ.get("CBL_N_TRACES", "15"))        # traces per regime
_BROKEN_LOSS = float("inf")

# action int <-> ClusterType.  NONE=0, SPOT=1, ON_DEMAND=2  (independent of the
# enum's internal .value, so the candidate's encoding is unambiguous).
_ACT = {0: ClusterType.NONE, 1: ClusterType.SPOT, 2: ClusterType.ON_DEMAND}
_CT_TO_INT = {ClusterType.NONE: 0, ClusterType.SPOT: 1, ClusterType.ON_DEMAND: 2}


# --------------------------------------------------------------------------- #
# Regimes: same deadline + trace pool, different restart overhead -> different
# optimal thrash-vs-commit threshold.
# --------------------------------------------------------------------------- #
REGIMES = {
    "cheap_restart": dict(overhead=0.02, desc="cheap restarts (low-Delta)"),
    "moderate": dict(overhead=0.20, desc="textbook restart overhead"),
    "costly_restart": dict(overhead=0.40, desc="costly restarts (high-Delta)"),
}

ORDER = ["cheap_restart", "moderate", "costly_restart"]


# --------------------------------------------------------------------------- #
# Deterministic trace selection (a fixed, reproducible subset of the pool).
# All overhead folders hold identical spot-availability data, so one pool serves
# every regime; the overhead is applied by the simulator, not the trace.
# --------------------------------------------------------------------------- #
_TRACE_CACHE = None


def _trace_pool():
    global _TRACE_CACHE
    if _TRACE_CACHE is None:
        allt = sorted(glob.glob(os.path.join(
            _TRACE_ROOT, "*", "traces", "random_start", "*.json")))
        usable = []
        need_h = max(DEADLINE_H, TASK_DURATION_H)
        for t in allt:
            try:
                d = json.load(open(t))
                gap = d["metadata"]["gap_seconds"]
                if len(d["data"]) * gap / 3600.0 >= need_h:
                    usable.append(t)
            except Exception:
                continue
        if not usable:
            raise RuntimeError(
                f"No usable traces >= {need_h}h under {_TRACE_ROOT}. "
                "Did the real_traces.tar.gz download/extract succeed?")
        # even stride across the sorted pool -> spread over envs and trace ids
        stride = max(1, len(usable) // N_TRACES)
        _TRACE_CACHE = usable[::stride][:N_TRACES]
    return _TRACE_CACHE


# --------------------------------------------------------------------------- #
# Thin Strategy that delegates the heuristic decision to the candidate rule.
# Registered ONCE (NAME unique); the candidate fn + params are swapped per call.
# --------------------------------------------------------------------------- #
class _EvalStrategy(Strategy):
    NAME = "hybrid_eval"

    def set_candidate(self, decide, params):
        self._decide = decide
        self._params = params

    def _step(self, last_cluster_type, has_spot):
        env = self.env
        remaining_task_time = self.task_duration - sum(self.task_done_time)
        remaining_time = self.deadline - env.elapsed_seconds
        obs = {
            "remaining_task_time": float(remaining_task_time),   # work left (s)
            "remaining_time": float(remaining_time),             # time to ddl (s)
            "slack": float(remaining_time - remaining_task_time),  # spare time (s)
            "restart_overhead": float(self.restart_overhead),     # per-switch (s)
            "remaining_restart_overhead": float(self.remaining_restart_overhead),
            "has_spot": bool(has_spot),                           # spot up NOW
            "last_cluster_type": _CT_TO_INT.get(last_cluster_type, 0),
            "gap_seconds": float(env.gap_seconds),                # tick length (s)
            "elapsed_seconds": float(env.elapsed_seconds),
            "deadline": float(self.deadline),
            "task_duration": float(self.task_duration),
        }
        a = int(self._decide(obs, self._params))
        return _ACT.get(a, ClusterType.NONE)


class CantBeLateBenchmark:
    """Self-contained cost harness: a decision rule is scored by the mean dollar
    cost of finishing the job over the regime's fixed trace subset. evaluate()
    returns the LOSS = cost so it plugs straight into synth_meta_optimizer.tune."""

    def __init__(self, name, n_traces=None, deadline_h=None):
        spec = REGIMES[name]
        self.name = name
        self.overhead = spec["overhead"]
        self.desc = spec["desc"]
        self.deadline_h = deadline_h if deadline_h is not None else DEADLINE_H
        pool = _trace_pool()
        self.traces = pool if n_traces is None else pool[:n_traces]

    # -- one rollout (minimal single-region loop; deadline hard-enforced) ----
    def _rollout_cost(self, decide, params, trace_file):
        from types import SimpleNamespace
        args = SimpleNamespace(
            deadline_hours=self.deadline_h,
            restart_overhead_hours=[self.overhead],
            inter_task_overhead=[0.0],
        )
        env = TraceEnv(trace_file, 0)
        task = SingleTask({"duration": TASK_DURATION_H, "checkpoint_size_gb": 50})
        strat = _EvalStrategy(args)
        strat.set_candidate(decide, params)
        env.reset()
        strat.reset(env, task)
        steps = 0
        max_steps = len(env.trace) + 5
        while not strat.task_done:
            request_type = strat.step()
            env.step(request_type)
            steps += 1
            if steps > max_steps:
                raise RuntimeError("policy did not terminate within the trace")
        strat.step()                 # realize the final tick
        env.step(ClusterType.NONE)
        return float(env.accumulated_cost)

    # -- harness interface used by tune() / untuned_score() -----------------
    def evaluate(self, decide, params):
        """Mean finishing cost over the regime's traces, as LOSS (lower=better).
        Any exception on any trace -> +inf (brittle-policy penalty)."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                costs = [self._rollout_cost(decide, params, t) for t in self.traces]
        except Exception:
            return _BROKEN_LOSS
        if not costs or any(not np.isfinite(c) for c in costs):
            return _BROKEN_LOSS
        return float(np.mean(costs))


# --------------------------------------------------------------------------- #
# Hand-written seed structures (the families named in the prompt). Used for:
# (a) the pure-CMA-ES / vanilla-numerical arm, and (b) --selftest. Each reads its
# constants ONLY from `params`, like an LLM proposal must.
# --------------------------------------------------------------------------- #
def seed_slack_buffer(obs, params):
    """(S1) Slack-threshold rule: gamble on SPOT (or wait) while there is plenty
    of slack to the deadline; once slack falls below `buffer` seconds, commit to
    ON_DEMAND rather than risk a forced end-of-run block. One tunable hole; this
    is the fixed seed structure for the pure-CMA-ES arm."""
    if obs["remaining_task_time"] <= 1e-3:
        return 0
    if obs["slack"] <= params["buffer"]:
        return 2
    return 1 if obs["has_spot"] else 0


def seed_hysteresis(obs, params):
    """(S2) Two-threshold hysteresis ladder. Above `hi` slack: always wait for
    SPOT (free). Below `lo` slack: always ON_DEMAND. In the band between, use
    ON_DEMAND only when spot is down (hedge), else ride SPOT."""
    if obs["remaining_task_time"] <= 1e-3:
        return 0
    slack = obs["slack"]
    hi, lo = params["hi"], params["lo"]
    if slack <= lo:
        return 2
    if slack >= hi:
        return 1 if obs["has_spot"] else 0
    if obs["has_spot"]:
        return 1
    return 2


def seed_pressure(obs, params):
    """(S3) Time-pressure controller: pressure = remaining_work / remaining_time.
    When pressure exceeds `p_thresh` (deadline closing in), commit to ON_DEMAND;
    otherwise ride SPOT when available and wait when not."""
    if obs["remaining_task_time"] <= 1e-3:
        return 0
    rt = max(obs["remaining_time"], 1e-9)
    pressure = obs["remaining_task_time"] / rt
    if pressure >= params["p_thresh"]:
        return 2
    return 1 if obs["has_spot"] else 0


# Parameter manifests (Def. 1). Times in seconds (gap = 600s); 'scale' "cont"
# for bounded coefficients, "log" for multiplicative.
_DAY_S = 86400.0
SEED_SLACK_BUFFER = {
    "fn": seed_slack_buffer,
    "name": "S1 slack-threshold buffer",
    "manifest": {
        "buffer": dict(bounds=(0.0, _DAY_S), guess=7200.0, scale="cont"),
    },
}
SEED_HYSTERESIS = {
    "fn": seed_hysteresis,
    "name": "S2 two-threshold hysteresis",
    "manifest": {
        "hi": dict(bounds=(0.0, 2 * _DAY_S), guess=43200.0, scale="cont"),
        "lo": dict(bounds=(0.0, _DAY_S), guess=7200.0, scale="cont"),
    },
}
SEED_PRESSURE = {
    "fn": seed_pressure,
    "name": "S3 time-pressure controller",
    "manifest": {
        "p_thresh": dict(bounds=(0.50, 1.0), guess=0.90, scale="cont"),
    },
}

SEEDS = {
    "slack_buffer": SEED_SLACK_BUFFER,
    "hysteresis": SEED_HYSTERESIS,
    "pressure": SEED_PRESSURE,
}

# Fixed seed structure for the pure-CMA-ES / vanilla-numerical arm.
SEED_FIXED = SEED_SLACK_BUFFER


# --------------------------------------------------------------------------- #
# Pure-CMA-ES / vanilla-numerical reference (paper Sec. 6.2, arm 2)
# --------------------------------------------------------------------------- #
def pure_cmaes(bench, seed_structure=SEED_FIXED, budget=100, seed=1):
    """CMA-ES over the parameters of a FIXED seed structure -- no structural
    search. Capped at the seed structure's ceiling. Returns
    (cost_best, best_params, n_evals)."""
    from synth_meta_optimizer import tune
    loss, params, evals = tune(bench, seed_structure, budget=budget, seed=seed)
    return float(loss), params, int(evals)


# --------------------------------------------------------------------------- #
# Self-test (no LLM): exercise the full machinery on the hand-written sketches.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from synth_meta_optimizer import tune, untuned_score, fmt

    regime = sys.argv[1] if len(sys.argv) > 1 else "costly_restart"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 40
    bench = CantBeLateBenchmark(regime)
    print(f"=== Can't Be Late self-test | regime={regime} ({bench.desc}) "
          f"| ddl={bench.deadline_h}h overhead={bench.overhead}h "
          f"| n_traces={len(bench.traces)} | B_in={budget} ===")
    for key, cand in SEEDS.items():
        loss_untuned, _ = untuned_score(bench, cand)
        loss_tuned, params, ev = pure_cmaes(bench, cand, budget=budget, seed=1)
        gap = loss_untuned - loss_tuned
        print(f"  {cand['name']:30s} cost_guess={fmt(loss_untuned):>10}  "
              f"cost_tuned={fmt(loss_tuned):>10}  Delta={fmt(gap):>10}  [{ev} evals]")
    print("Self-test complete (machinery OK).")
