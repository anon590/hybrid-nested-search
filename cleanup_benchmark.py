"""
Cleanup testbed for the LLM-driven Hybrid Nested Search (paper.pdf, Sec. 6.1).

This is the social-dilemma analog of benchmarks.py. Where the meta-optimizer
hides a 2-D function behind a black-box gradient oracle, here we hide the
Cleanup *dynamics constants* (waste spawn rate, depletion threshold, apple
respawn rate, team size) behind a self-play rollout. The artifact under search
is a TEAM POLICY  policy(env, agent_id, params)  whose STRUCTURE maps the global
pollution level to a number of cleaners, and whose continuous holes `params`
(threshold breakpoints, per-band cleaner counts, controller gains) are tuned by
the inner CMA-ES.

The harness fitness is the utilitarian efficiency U = collective reward per
timestep (Perolat et al.'s social-outcome metric, already implemented in
CleanupEnv.compute_metrics). CMA-ES is a *minimizer*, so evaluate() returns the
LOSS = -U; the driver flips the sign back to U for reporting. Brittle policies
(any raised exception during a rollout) score +inf, matching benchmarks.py.

Several "regimes" play the role of the meta-optimizer's function suite. They
share one small map and differ only in the hidden dynamics, so the *optimal*
number of cleaners differs across regimes. A regime where the river tolerates a
lot of pollution makes the LLM's default guess near-optimal (the Delta ~ 0 tie
control of Eq. (4)); a regime with fast pollution and a tight depletion
threshold needs aggressive, non-obvious cleaning (large Delta -> hybrid wins).
"""

from __future__ import annotations

import numpy as np

from cleanup_env import make_cleanup, CleanupEnv, CleanupAction
from cleanup_helpers import (
    waste_ratio, assign_roles, clean_action, harvest_action,
)

# --------------------------------------------------------------------------- #
# Regimes: same map, hidden dynamics differ -> different optimal cleaner count
# --------------------------------------------------------------------------- #
# Each regime is a CleanupEnv configuration. `small=True` uses CLEANUP_MAP_SMALL
# (11x13, 24 river cells, 45 apple spawns). The hidden constants below are what
# the LLM never sees; they set where the parameter optimum lands.
REGIMES = {
    "light": dict(
        small=True, n_agents=4, max_steps=300, seeds=(0, 1, 2),
        dynamics=dict(waste_spawn_prob=0.20, threshold_depletion=0.60,
                      apple_respawn_prob=0.05),
        desc="slow pollution, tolerant river (low-Delta control)"),
    "moderate": dict(
        small=True, n_agents=6, max_steps=300, seeds=(0, 1, 2),
        dynamics=dict(waste_spawn_prob=0.50, threshold_depletion=0.40,
                      apple_respawn_prob=0.05),
        desc="textbook Cleanup dynamics (moderate Delta)"),
    "heavy": dict(
        small=True, n_agents=7, max_steps=300, seeds=(0, 1, 2),
        dynamics=dict(waste_spawn_prob=0.90, threshold_depletion=0.30,
                      apple_respawn_prob=0.06),
        desc="fast pollution, tight depletion (high Delta)"),
    # optional large-map stress regime (not in default ORDER; heavier compute)
    "heavy_large": dict(
        small=False, n_agents=10, max_steps=400, seeds=(0, 1),
        dynamics=dict(waste_spawn_prob=0.90, threshold_depletion=0.30,
                      apple_respawn_prob=0.05),
        desc="large map, fast pollution (high Delta, costly)"),
}

ORDER = ["light", "moderate", "heavy"]

# Loss returned for a rollout that raises (penalize brittle code), mirroring
# benchmarks.py which returns inf on any exception.
_BROKEN_LOSS = float("inf")


class CleanupBenchmark:
    """Self-play harness: a team policy is scored by the efficiency U it
    achieves, averaged over the regime's fixed evaluation seeds. evaluate()
    returns the LOSS = -U so it plugs straight into synth_meta_optimizer.tune."""

    def __init__(self, name, max_steps=None, seeds=None):
        spec = REGIMES[name]
        self.name = name
        self.small = spec["small"]
        self.n_agents = spec["n_agents"]
        self.max_steps = max_steps if max_steps is not None else spec["max_steps"]
        self.seeds = tuple(seeds) if seeds is not None else tuple(spec["seeds"])
        self.dynamics = dict(spec["dynamics"])
        self.desc = spec["desc"]

    # -- env construction ---------------------------------------------------
    def _make_env(self, seed):
        return make_cleanup(
            n_agents=self.n_agents, small=self.small,
            max_steps=self.max_steps, seed=seed, **self.dynamics)

    # -- one self-play episode ---------------------------------------------
    def _rollout_U(self, policy_fn, params, seed):
        env = self._make_env(seed)
        env.reset(seed=seed)
        n = env.n_agents
        ep_rewards = {i: [] for i in range(n)}
        ep_timeouts = {i: [] for i in range(n)}
        for _ in range(self.max_steps):
            actions = {}
            for i in range(n):
                a = policy_fn(env, i, params)
                a = int(a)
                if a < 0 or a >= env.action_space_n:
                    a = int(CleanupAction.STAND)
                actions[i] = a
            _, rewards, _, _, info = env.step(actions)
            for i in range(n):
                ep_rewards[i].append(rewards[i])
                ep_timeouts[i].append(info[i]["timeout"] > 0)
        m = CleanupEnv.compute_metrics(ep_rewards, ep_timeouts)
        return float(m["efficiency"])

    # -- harness interface used by tune() / untuned_score() -----------------
    def evaluate(self, policy_fn, params):
        """Mean efficiency over the regime's seeds, returned as LOSS = -U.
        Any exception on any seed -> +inf (brittle policy penalty)."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                Us = [self._rollout_U(policy_fn, params, s) for s in self.seeds]
        except Exception:
            return _BROKEN_LOSS
        if not Us or any(not np.isfinite(u) for u in Us):
            return _BROKEN_LOSS
        return -float(np.mean(Us))

    def utility(self, policy_fn, params):
        """Convenience: signed efficiency U (higher is better) = -loss."""
        return -self.evaluate(policy_fn, params)


# --------------------------------------------------------------------------- #
# Hand-written seed structures (the three sketches named in paper Sec. 6.1).
# Used for: (a) the pure-CMA-ES / vanilla-numerical arm, and (b) --selftest.
# Each reads its constants ONLY from `params`, just like an LLM proposal must.
# --------------------------------------------------------------------------- #
def _dispatch(env, agent_id, n_cleaners):
    roles = assign_roles(env, n_cleaners)
    if roles[agent_id] == "clean":
        return clean_action(env, agent_id)
    return harvest_action(env, agent_id)


def seed_fixed_fraction(env, agent_id, params):
    """(S3) Fixed roles: a constant fraction of the team cleans, regardless of
    pollution. One tunable hole. This is the *seed structure* for pure CMA-ES:
    it has no waste-adaptivity, so its ceiling caps the numerical-only arm."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    n_c = params["clean_fraction"] * env.n_agents
    return _dispatch(env, agent_id, n_c)


def seed_proportional(env, agent_id, params):
    """(S2) Proportional controller: n_c = round(gain * waste_ratio * n_agents)."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    n_c = params["gain"] * waste_ratio(env) * env.n_agents
    return _dispatch(env, agent_id, n_c)


def seed_ladder(env, agent_id, params):
    """(S1) Waste-adaptive threshold ladder mapping pollution -> cleaner count.
    Breakpoints b1>b2>b3 and per-band counts c_hi>c_mid>c_lo>c_base."""
    if int(env.agent_timeout[agent_id]) > 0:
        return int(CleanupAction.STAND)
    wr = waste_ratio(env)
    b1, b2, b3 = params["b1"], params["b2"], params["b3"]
    if wr >= b1:
        n_c = params["c_hi"]
    elif wr >= b2:
        n_c = params["c_mid"]
    elif wr >= b3:
        n_c = params["c_lo"]
    else:
        n_c = params["c_base"]
    return _dispatch(env, agent_id, n_c)


# Parameter manifests (Def. 1). 'scale': "cont" for bounded coefficients,
# "log" for multiplicative gains.
SEED_FIXED_FRACTION = {
    "fn": seed_fixed_fraction,
    "name": "S3 fixed-fraction roles",
    "manifest": {
        "clean_fraction": dict(bounds=(0.0, 1.0), guess=0.3, scale="cont"),
    },
}
SEED_PROPORTIONAL = {
    "fn": seed_proportional,
    "name": "S2 proportional controller",
    "manifest": {
        "gain": dict(bounds=(0.0, 3.0), guess=1.0, scale="cont"),
    },
}
SEED_LADDER = {
    "fn": seed_ladder,
    "name": "S1 waste-adaptive ladder",
    "manifest": {
        "b1": dict(bounds=(0.30, 0.95), guess=0.60, scale="cont"),
        "b2": dict(bounds=(0.15, 0.60), guess=0.35, scale="cont"),
        "b3": dict(bounds=(0.02, 0.30), guess=0.10, scale="cont"),
        "c_hi": dict(bounds=(0.0, 12.0), guess=5.0, scale="cont"),
        "c_mid": dict(bounds=(0.0, 10.0), guess=3.0, scale="cont"),
        "c_lo": dict(bounds=(0.0, 8.0), guess=2.0, scale="cont"),
        "c_base": dict(bounds=(0.0, 6.0), guess=1.0, scale="cont"),
    },
}

SEEDS = {
    "fixed_fraction": SEED_FIXED_FRACTION,
    "proportional": SEED_PROPORTIONAL,
    "ladder": SEED_LADDER,
}


# --------------------------------------------------------------------------- #
# Pure-CMA-ES / vanilla-numerical reference (paper Sec. 6.2, arm 2)
# --------------------------------------------------------------------------- #
def pure_cmaes(bench, seed_structure=SEED_FIXED_FRACTION, budget=100, seed=1):
    """CMA-ES over the parameters of a FIXED seed structure -- no structural
    search. Isolates the value of structural discovery: this arm is capped at
    its seed structure's ceiling. Returns (U_best, best_params, n_evals)."""
    from synth_meta_optimizer import tune
    loss, params, evals = tune(bench, seed_structure, budget=budget, seed=seed)
    return -float(loss), params, int(evals)


# --------------------------------------------------------------------------- #
# Self-test (no LLM): exercise the full machinery on hand-written sketches.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    import sys
    from synth_meta_optimizer import tune, untuned_score, fmt

    regime = sys.argv[1] if len(sys.argv) > 1 else "moderate"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    bench = CleanupBenchmark(regime, max_steps=200, seeds=(0, 1))
    print(f"=== Cleanup self-test | regime={regime} ({bench.desc}) "
          f"| n_agents={bench.n_agents} steps={bench.max_steps} "
          f"seeds={bench.seeds} | B_in={budget} ===")
    for key, cand in SEEDS.items():
        u_untuned = -untuned_score(bench, cand)[0]
        u_tuned, params, ev = pure_cmaes(bench, cand, budget=budget, seed=1)
        gap = u_tuned - u_untuned
        print(f"  {cand['name']:28s} U_guess={u_untuned:7.3f}  "
              f"U_tuned={u_tuned:7.3f}  Delta={gap:7.3f}  [{ev} evals]")
    print("Self-test complete (machinery OK).")
