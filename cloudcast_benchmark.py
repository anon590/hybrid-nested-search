"""
Cloudcast testbed for the LLM-driven Hybrid Nested Search (paper.pdf, Sec. 6).

This is the multi-cloud broadcast-routing analog of cleanup_benchmark.py /
benchmarks.py. The artifact under search is a routing ALGORITHM
    search_algorithm(src, dsts, G, num_partitions, params) -> BroadCastTopology
whose STRUCTURE (independent shortest paths, shared relay tree, k-shortest
multipath, ...) is proposed by the LLM, and whose continuous holes `params`
(cost/throughput blend weight, relay/detour tolerances, hub counts) are tuned by
the inner CMA-ES.

The harness fitness is the total transfer COST in dollars returned by the
upstream BCSimulator (egress fees + instance cost). CMA-ES is a *minimizer*, so
evaluate() returns the LOSS = cost directly (lower is better), exactly like
benchmarks.py. Brittle algorithms (any raised exception, an incomplete topology
that leaves some (dst, partition) unrouted, or a pathological blow-up that trips
the per-evaluation timeout) score +inf.

Two "regimes" play the role of the meta-optimizer's function suite. They differ
in the cost geometry that sets where the parameter optimum lands:

  intra  -- source and destinations in ONE provider: intra-cloud egress is cheap
            and the cost-only shortest-path tree is already near-optimal, so the
            LLM's default constants are about as good as tuned ones
            (the Delta ~ 0 tie control of Eq. (4): structure dominates, tuning
            barely moves cost).
  inter  -- cross-provider broadcast: expensive, heterogeneous egress with more
            sharing/relay opportunities, so the right blend weights and relay
            tolerances are non-obvious -> a larger tuning gap Delta.

NOTE the LLM is NOT blind to the per-edge cost/throughput (it receives the
graph, exactly as the upstream GEPA example does). The tuning gap arises because
TOTAL cost is a *global* function of the chosen topology (shared-edge partition
accounting, ingress/egress proportional flow throttling, instance cost from
runtime) that the LLM cannot optimize per-edge by inspection -- only the inner
solver, which actually runs the simulator, can place the continuous holes.
"""

from __future__ import annotations

import os
import sys
import io
import json
import signal
import threading
import contextlib

import numpy as np

# --------------------------------------------------------------------------- #
# Locate the GEPA repo that ships the Cloudcast simulator + data.
# --------------------------------------------------------------------------- #
_GEPA_ROOT = os.environ.get(
    "GEPA_ROOT",
    os.path.expanduser("~/gepa"),
)
if not os.path.isdir(os.path.join(_GEPA_ROOT, "examples", "adrs", "cloudcast")):
    raise FileNotFoundError(
        f"Cloudcast example not found under GEPA_ROOT={_GEPA_ROOT!r}. "
        "Set GEPA_ROOT to your gepa checkout.")
if _GEPA_ROOT not in sys.path:
    sys.path.insert(0, _GEPA_ROOT)

import networkx as nx  # noqa: E402  (after sys.path setup)
from examples.adrs.cloudcast.utils.cloudcast.utils import make_nx_graph        # noqa: E402
from examples.adrs.cloudcast.utils.cloudcast.broadcast import BroadCastTopology  # noqa: E402
from examples.adrs.cloudcast.utils.cloudcast.simulator import BCSimulator      # noqa: E402

_CONFIG_DIR = os.path.join(
    _GEPA_ROOT, "examples", "adrs", "cloudcast", "utils", "cloudcast", "config")

NUM_VMS = int(os.environ.get("CLOUDCAST_NUM_VMS", "2"))
# Wall-clock budget for ONE search_algorithm call (guards pathological structures
# such as nx.all_simple_paths over the ~70-node mesh). +inf loss on timeout.
_EVAL_TIMEOUT_S = int(os.environ.get("CLOUDCAST_EVAL_TIMEOUT", "20"))
_BROKEN_LOSS = float("inf")


# --------------------------------------------------------------------------- #
# Regimes: same network, different cost geometry -> different optimal routing.
# --------------------------------------------------------------------------- #
REGIMES = {
    "intra": dict(
        configs=["intra_aws", "intra_azure", "intra_gcp"],
        desc="intra-cloud (cheap egress, low-Delta)"),
    "inter": dict(
        configs=["inter_agz", "inter_gaz2"],
        desc="inter-cloud (expensive egress, high-Delta)"),
}

ORDER = ["intra", "inter"]


# --------------------------------------------------------------------------- #
# Shared network graph (building it from CSV is slow; cache and hand out copies).
# --------------------------------------------------------------------------- #
_G_CACHE = None


def _base_graph():
    global _G_CACHE
    if _G_CACHE is None:
        with contextlib.redirect_stdout(io.StringIO()):
            _G_CACHE = make_nx_graph(num_vms=NUM_VMS)
    return _G_CACHE.copy()


def _load_config(name):
    with open(os.path.join(_CONFIG_DIR, f"{name}.json")) as f:
        return json.load(f)


class _Timeout:
    """SIGALRM-based wall-clock guard. No-op off the main thread (where SIGALRM
    is unavailable) so the harness still runs, just without the guard."""

    def __init__(self, seconds):
        self.seconds = max(1, int(seconds))
        self._armed = False

    def __enter__(self):
        if threading.current_thread() is threading.main_thread():
            try:
                signal.signal(signal.SIGALRM, self._raise)
                signal.alarm(self.seconds)
                self._armed = True
            except (ValueError, AttributeError):
                self._armed = False
        return self

    def __exit__(self, *exc):
        if self._armed:
            signal.alarm(0)
        return False

    @staticmethod
    def _raise(signum, frame):
        raise TimeoutError("search_algorithm exceeded the per-evaluation budget")


class CloudcastBenchmark:
    """Broadcast-routing harness: an algorithm is scored by the mean transfer
    COST over the regime's scenario configs. evaluate() returns the LOSS = cost
    so it plugs straight into synth_meta_optimizer.tune (a minimizer)."""

    def __init__(self, name):
        spec = REGIMES[name]
        self.name = name
        self.desc = spec["desc"]
        self.config_names = list(spec["configs"])
        self.configs = [_load_config(c) for c in self.config_names]

    # -- one scenario -------------------------------------------------------
    def _cost_one(self, fn, params, config):
        G = _base_graph()
        with _Timeout(_EVAL_TIMEOUT_S):
            topo = fn(config["source_node"], config["dest_nodes"], G,
                      config["num_partitions"], params)
        if topo is None:
            return _BROKEN_LOSS
        topo.set_num_partitions(config["num_partitions"])
        # Completeness: every destination/partition must carry a path, else the
        # simulator would iterate a None and the broadcast is invalid anyway.
        for dst in config["dest_nodes"]:
            paths = topo.paths.get(dst)
            if not paths:
                return _BROKEN_LOSS
            for i in range(config["num_partitions"]):
                if not paths.get(str(i)):
                    return _BROKEN_LOSS
        sim = BCSimulator(num_vms=NUM_VMS)
        with contextlib.redirect_stdout(io.StringIO()):
            _, cost = sim.evaluate_path(topo, config)
        return float(cost)

    # -- harness interface used by tune() / untuned_score() -----------------
    def evaluate(self, fn, params):
        """Mean cost over the regime's scenarios, returned as LOSS (lower=better).
        Any exception / incomplete topology / timeout on any scenario -> +inf."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                costs = [self._cost_one(fn, params, c) for c in self.configs]
        except Exception:
            return _BROKEN_LOSS
        if not costs or any(not np.isfinite(c) for c in costs):
            return _BROKEN_LOSS
        return float(np.mean(costs))


# --------------------------------------------------------------------------- #
# Hand-written seed structures (the families named in the prompt). Used for:
# (a) the pure-CMA-ES / vanilla-numerical arm, and (b) --selftest. Each reads its
# constants ONLY from `params`, just like an LLM proposal must, and each returns
# a COMPLETE topology so it never spuriously scores +inf.
# --------------------------------------------------------------------------- #
def _weighted_paths(src, dsts, G, num_partitions, weight_of):
    """Shared helper: per-destination Dijkstra under a custom edge weight; all
    partitions follow that destination's path (shares common prefixes)."""
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    for _, _, d in h.edges(data=True):
        d["w"] = float(weight_of(d))
    bc = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="w")
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(num_partitions):
                bc.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    return bc


def seed_blended(src, dsts, G, num_partitions, params):
    """(S1) Cost/throughput-blended shortest paths. tput_weight=0 recovers the
    cost-only Dijkstra baseline; raising it trades egress for faster (cheaper
    instance-time) links. The single tunable hole caps the pure-CMA-ES arm."""
    w = params["tput_weight"]
    return _weighted_paths(
        src, dsts, G, num_partitions,
        lambda d: d["cost"] + w * (1.0 / (d.get("throughput") or 1e-9)))


def seed_power(src, dsts, G, num_partitions, params):
    """(S2) Power-law cost weighting: weight = cost ** alpha. alpha>1 sharpens
    preference for the very cheapest hops (more relaying); alpha<1 flattens it."""
    alpha = params["alpha"]
    return _weighted_paths(
        src, dsts, G, num_partitions,
        lambda d: float(max(d["cost"], 1e-9)) ** alpha)


def seed_congestion(src, dsts, G, num_partitions, params):
    """(S3) Congestion-relieved cost: weight = cost + relief / throughput, a
    throughput-aware bias that spreads load off bandwidth-poor links."""
    relief = params["relief"]
    return _weighted_paths(
        src, dsts, G, num_partitions,
        lambda d: d["cost"] + relief / (d.get("throughput") or 1e-9))


# Parameter manifests (Def. 1). 'scale': "log" for multiplicative gains,
# "cont" for bounded/additive coefficients.
SEED_BLENDED = {
    "fn": seed_blended,
    "name": "S1 cost/throughput-blended paths",
    "manifest": {
        "tput_weight": dict(bounds=(0.0, 5.0), guess=0.5, scale="cont"),
    },
}
SEED_POWER = {
    "fn": seed_power,
    "name": "S2 power-law cost weighting",
    "manifest": {
        "alpha": dict(bounds=(0.25, 3.0), guess=1.0, scale="cont"),
    },
}
SEED_CONGESTION = {
    "fn": seed_congestion,
    "name": "S3 congestion-relieved cost",
    "manifest": {
        "relief": dict(bounds=(1e-4, 1e1), guess=1e-2, scale="log"),
    },
}

SEEDS = {
    "blended": SEED_BLENDED,
    "power": SEED_POWER,
    "congestion": SEED_CONGESTION,
}

# The fixed seed structure for the pure-CMA-ES / vanilla-numerical arm: a single
# routing rule with one tunable hole, so its ceiling = single-path weighted
# routing (no relay / multipath structure) -- the arm cannot restructure.
SEED_FIXED = SEED_BLENDED


# --------------------------------------------------------------------------- #
# Pure-CMA-ES / vanilla-numerical reference (paper Sec. 6.2, arm 2)
# --------------------------------------------------------------------------- #
def pure_cmaes(bench, seed_structure=SEED_FIXED, budget=100, seed=1):
    """CMA-ES over the parameters of a FIXED seed structure -- no structural
    search. Isolates the value of structural discovery: this arm is capped at
    its seed structure's ceiling. Returns (cost_best, best_params, n_evals)."""
    from synth_meta_optimizer import tune
    loss, params, evals = tune(bench, seed_structure, budget=budget, seed=seed)
    return float(loss), params, int(evals)


# --------------------------------------------------------------------------- #
# Self-test (no LLM): exercise the full machinery on the hand-written sketches.
# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    from synth_meta_optimizer import tune, untuned_score, fmt

    regime = sys.argv[1] if len(sys.argv) > 1 else "inter"
    budget = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    bench = CloudcastBenchmark(regime)
    print(f"=== Cloudcast self-test | regime={regime} ({bench.desc}) "
          f"| configs={bench.config_names} | B_in={budget} ===")
    for key, cand in SEEDS.items():
        loss_untuned, _ = untuned_score(bench, cand)
        loss_tuned, params, ev = pure_cmaes(bench, cand, budget=budget, seed=1)
        gap = loss_untuned - loss_tuned
        print(f"  {cand['name']:32s} cost_guess={fmt(loss_untuned):>10}  "
              f"cost_tuned={fmt(loss_tuned):>10}  Delta={fmt(gap):>10}  [{ev} evals]")
    print("Self-test complete (machinery OK).")
