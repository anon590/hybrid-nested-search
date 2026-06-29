"""
Synthetic experiment for the Hybrid Nested Search framework (paper.pdf).

Setup from synth_exp.md ("Rosenbrock Meta-Optimizer" benchmark):

    f(x,y) = (1-x)^2 + 100 (y - x^2)^2          # global min 0 at (1,1)

We isolate the paper's thesis (LLMs propose *structure*, a zero-order inner
solver tunes the *continuous holes*) with NO LLM in the loop. The structures
(candidate programs) are given; CMA-ES plays every role:

  RUN 1 -- Vanilla CMA-ES over the objective itself.
           Optimizes (x, y) directly. "Trivial for CMA-ES, impossible for an
           LLM" (synth_exp.md). Reference: confirms the reachable optimum is 0.

  RUN 2 -- CMA-ES over the parameters of Candidate 1 (Standard GD; tau_1).
           manifest: {lr}.

  RUN 3 -- CMA-ES over the parameters of Candidate 2 (GD + Momentum; tau_2).
           manifest: {lr, beta}.

Candidate 2 uses CLASSIC HEAVY-BALL momentum (v = beta*v + g), the form whose
dynamics synth_exp.md's prose describes: at the LLM's default guess it
*overshoots the valley and diverges to infinity*. (The EMA variant in the md
code listing does not diverge and is dropped from the analysis.)

The point (Prop. 1 / Eq. 3-4): the *untuned* guess for the better structure
(tau_2) diverges, so vanilla joint search ranks tau_2 BELOW tau_1 (parametric
aliasing) and discards momentum. The inner CMA-ES de-aliases the ranking: once
tuned, tau_2 reaches ~0 and is correctly recognized as the better structure.

Run:  uv run --with numpy --with cma python3 synth_meta_optimizer.py
"""

import numpy as np
import cma


# --------------------------------------------------------------------------- #
# Benchmark (structure verbatim from synth_exp.md)
# --------------------------------------------------------------------------- #
class RosenbrockBenchmark:
    def __init__(self, steps=1000):
        self.steps = steps
        # Starting point deliberately placed far from the global minimum (1, 1)
        self.x0, self.y0 = -1.2, 1.0

    def rosenbrock(self, x, y):
        """The classic banana function. Global min at (1, 1) where f = 0."""
        return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2

    def rosenbrock_grad(self, x, y):
        """Gradient oracle provided to the candidate algorithm."""
        dx = -2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2)
        dy = 200.0 * (y - x ** 2)
        return dx, dy

    def evaluate(self, optimizer_func, params):
        """Run a candidate optimizer; return final Rosenbrock value (lower=better)."""
        try:
            with np.errstate(over="ignore", invalid="ignore"):
                final_x, final_y = optimizer_func(
                    self.rosenbrock_grad, self.x0, self.y0, self.steps, params
                )
                final_x = np.clip(final_x, -1e5, 1e5)
                final_y = np.clip(final_y, -1e5, 1e5)
                val = self.rosenbrock(final_x, final_y)
            return float(val) if np.isfinite(val) else float("inf")
        except Exception:
            return float("inf")  # penalize broken code / math overflow


# --------------------------------------------------------------------------- #
# Candidate programs (the "structures" an LLM would emit)
# --------------------------------------------------------------------------- #
def candidate1_gd(grad_fn, x, y, steps, params):
    """Candidate 1 (tau_1): Standard Gradient Descent."""
    lr = params["lr"]
    for _ in range(steps):
        dx, dy = grad_fn(x, y)
        x -= lr * dx
        y -= lr * dy
    return x, y


def candidate2_momentum(grad_fn, x, y, steps, params):
    """Candidate 2 (tau_2): GD with classic HEAVY-BALL momentum.

    v = beta*v + g  -> amplifies the step by 1/(1-beta). At the default guess
    (lr=0.01, beta=0.9) this overshoots the Rosenbrock valley and diverges.
    """
    lr = params["lr"]
    beta = params["beta"]
    vx, vy = 0.0, 0.0
    for _ in range(steps):
        dx, dy = grad_fn(x, y)
        vx = beta * vx + dx
        vy = beta * vy + dy
        x -= lr * vx
        y -= lr * vy
    return x, y


# Parameter manifests (Def. 1 / Sec. 2.1). 'scale': lr is a gain -> log-scaled
# hole (paper: type "log"); beta is a plain fraction -> linear ("cont").
CAND1 = {
    "fn": candidate1_gd,
    "name": "Candidate 1: Standard GD (tau_1)",
    "manifest": {"lr": dict(bounds=(1e-5, 0.1), guess=0.001, scale="log")},
}
CAND2 = {
    "fn": candidate2_momentum,
    "name": "Candidate 2: GD + Momentum (tau_2)",
    "manifest": {
        "lr": dict(bounds=(1e-5, 0.1), guess=0.01, scale="log"),
        "beta": dict(bounds=(0.0, 0.99), guess=0.9, scale="cont"),
    },
}


# --------------------------------------------------------------------------- #
# Inner solver: CMA-ES Tune() over a manifest  (Sec. 2.3)
# --------------------------------------------------------------------------- #
def _to_value(z, spec):
    """Map normalized coordinate z in [0,1] to an actual parameter value."""
    lo, hi = spec["bounds"]
    if spec["scale"] == "log":
        return 10.0 ** (np.log10(lo) + z * (np.log10(hi) - np.log10(lo)))
    return lo + z * (hi - lo)


def _to_z(value, spec):
    """Inverse map: actual value -> normalized coordinate (the warm start)."""
    lo, hi = spec["bounds"]
    if spec["scale"] == "log":
        z = (np.log10(value) - np.log10(lo)) / (np.log10(hi) - np.log10(lo))
    else:
        z = (value - lo) / (hi - lo)
    return float(np.clip(z, 0.0, 1.0))


def tune(benchmark, candidate, budget=100, sigma0=0.25, seed=1, return_history=False):
    """CMA-ES over a structure's continuous holes, warm-started at the guess.

    Returns (best_value, best_params, n_evals[, history]).
    history is the best-so-far value after each evaluation (for plotting).
    """
    manifest = candidate["manifest"]
    fn = candidate["fn"]
    keys = list(manifest.keys())
    n = len(keys)
    dim = max(2, n)  # pycma needs dim >= 2; pad 1-hole structures with an inert coord
    z0 = [_to_z(manifest[k]["guess"], manifest[k]) for k in keys] + [0.5] * (dim - n)

    def to_params(z):
        return {k: _to_value(float(np.clip(z[i], 0.0, 1.0)), manifest[k])
                for i, k in enumerate(keys)}

    def objective(z):
        val = benchmark.evaluate(fn, to_params(z))
        return val if np.isfinite(val) else 1e12

    es = cma.CMAEvolutionStrategy(
        z0, sigma0,
        {"bounds": [0, 1], "maxfevals": budget, "seed": seed, "verbose": -9},
    )
    history = []
    best = np.inf
    while not es.stop():
        xs = es.ask()
        fs = [objective(x) for x in xs]
        es.tell(xs, fs)
        for fv in fs:
            best = min(best, fv)
            history.append(best)

    best_params = to_params(es.result.xbest)
    out = (float(es.result.fbest), best_params, int(es.result.evaluations))
    return out + (history,) if return_history else out


def untuned_score(benchmark, candidate):
    """Score of the LLM's raw guess (what vanilla joint search observes)."""
    params = {k: candidate["manifest"][k]["guess"] for k in candidate["manifest"]}
    return benchmark.evaluate(candidate["fn"], params), params


def vanilla_cmaes_over_objective(benchmark, budget=2000, sigma0=0.5, seed=1):
    """RUN 1: optimize (x, y) directly -- the pure numerical reference."""
    def objective(p):
        v = benchmark.rosenbrock(p[0], p[1])
        return v if np.isfinite(v) else 1e12

    es = cma.CMAEvolutionStrategy(
        [benchmark.x0, benchmark.y0], sigma0,
        {"maxfevals": budget, "ftarget": 1e-14, "seed": seed, "verbose": -9},
    )
    es.optimize(objective)
    return float(es.result.fbest), es.result.xbest, int(es.result.evaluations)


# --------------------------------------------------------------------------- #
# Experiment driver
# --------------------------------------------------------------------------- #
def fmt(v):
    if not np.isfinite(v):
        return "inf"
    return f"{v:.6e}" if (v >= 1e3 or (0 < v < 1e-3)) else f"{v:.6f}"


def main():
    B_IN = 100
    bench = RosenbrockBenchmark(steps=1000)

    print("=" * 74)
    print("HYBRID NESTED SEARCH -- Rosenbrock Meta-Optimizer synthetic experiment")
    print(f"  f(x,y)=(1-x)^2+100(y-x^2)^2 | start=({bench.x0},{bench.y0}) "
          f"| steps={bench.steps} | global min 0 @ (1,1)")
    print("=" * 74)

    print("\n[RUN 1] Vanilla CMA-ES over the objective f(x,y) directly")
    v1, xy, n1 = vanilla_cmaes_over_objective(bench)
    print(f"        best f = {fmt(v1)}   at (x,y)=({xy[0]:.6f},{xy[1]:.6f})   [{n1} evals]")

    print("\n[Untuned guesses]  g_van(tau) = f(tau, theta_guess)   (Eq. 3, the alias)")
    u1, p1 = untuned_score(bench, CAND1)
    u2, p2 = untuned_score(bench, CAND2)
    print(f"        {CAND1['name']:36s} guess={p1}  ->  {fmt(u1)}")
    print(f"        {CAND2['name']:36s} guess={p2}  ->  {fmt(u2)}")

    print(f"\n[RUN 2] CMA-ES over Candidate 1 parameters  (B_in={B_IN})")
    f1, bp1, e1 = tune(bench, CAND1, budget=B_IN, seed=1)
    print(f"        F_hat(tau_1) = {fmt(f1)}   at { {k: round(v,6) for k,v in bp1.items()} }   [{e1} evals]")

    print(f"\n[RUN 3] CMA-ES over Candidate 2 parameters  (B_in={B_IN})")
    f2, bp2, e2 = tune(bench, CAND2, budget=B_IN, seed=1)
    print(f"        F_hat(tau_2) = {fmt(f2)}   at { {k: round(v,6) for k,v in bp2.items()} }   [{e2} evals]")

    seeds = range(1, 9)
    r1 = [tune(bench, CAND1, budget=B_IN, seed=s)[0] for s in seeds]
    r2 = [tune(bench, CAND2, budget=B_IN, seed=s)[0] for s in seeds]
    print(f"\n[Robustness over {len(list(seeds))} seeds]  (best tuned f)")
    print(f"        tau_1 (GD)       : median={fmt(np.median(r1))}  min={fmt(np.min(r1))}  max={fmt(np.max(r1))}")
    print(f"        tau_2 (Momentum) : median={fmt(np.median(r2))}  min={fmt(np.min(r2))}  max={fmt(np.max(r2))}")

    print("\n" + "=" * 74)
    print("VERDICT -- parametric aliasing vs. hybrid de-aliasing")
    print("=" * 74)
    van_pick = CAND1["name"] if u1 <= u2 else CAND2["name"]
    hyb_pick = CAND1["name"] if f1 <= f2 else CAND2["name"]
    print(f"  Vanilla (untuned) sees: tau_1={fmt(u1)}  tau_2={fmt(u2)}  -> picks {van_pick}")
    print(f"  Hybrid  (tuned)   sees: tau_1={fmt(f1)}  tau_2={fmt(f2)}  -> picks {hyb_pick}")
    if hyb_pick == CAND2["name"] and van_pick == CAND1["name"]:
        print("  -> ALIAS BROKEN: hybrid recovers the superior structure vanilla discarded.")
        print("     Confirms Prop. 1 / Eq. 4 (advantage ~ E[Delta]).")
    print("=" * 74)


if __name__ == "__main__":
    main()
