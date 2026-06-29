"""
Test-function suite for the hidden meta-optimizer experiment.

Each problem is a 2-D objective with global minimum value 0. The LLM is BLIND
to all of this: it only ever receives a black-box gradient oracle grad_fn(x,y)
and a starting point. The functions span the regimes from synth_exp.md / the
method note's "matched class" discussion:

  rosenbrock  -- curved narrow valley           (curvature; momentum helps)
  ellipsoid   -- ill-conditioned (cond 1e4)      (per-coordinate scale; adaptive helps)
  rastrigin   -- regular multimodal lattice      (escape / large-or-annealed steps)
  ackley      -- flat outer region + center funnel(gradient normalization)
  schwefel    -- DECEPTIVE multimodal            (gradient points away from global min)

For multimodal problems we evaluate from several FIXED starts and average the
final loss, so the inner CMA-ES is rewarded for a structure that works generally
rather than one that luckily lands in the basin of a single start (Sec. 5).
"""

import numpy as np

# Default coordinate clip: points outside are scored at the clipped boundary, so
# a candidate cannot "win" by diverging. Schwefel overrides this with its
# canonical [-500, 500] box (see PROBLEMS) -- its loss is UNBOUNDED BELOW in the
# tail (the x*sin(sqrt|x|) amplitude grows with |x|), so without a tight box an
# optimizer can drive the loss arbitrarily negative instead of finding the basin.
COORD_CLIP = 1e5


# --------------------------------------------------------------------------- #
# Closed-form objectives + analytic gradients (the oracle handed to the LLM)
# --------------------------------------------------------------------------- #
def _rosenbrock(x, y):
    return (1.0 - x) ** 2 + 100.0 * (y - x ** 2) ** 2


def _rosenbrock_grad(x, y):
    return (-2.0 * (1.0 - x) - 400.0 * x * (y - x ** 2), 200.0 * (y - x ** 2))


COND = 1.0e4  # ellipsoid condition number


def _ellipsoid(x, y):
    return x ** 2 + COND * y ** 2


def _ellipsoid_grad(x, y):
    return (2.0 * x, 2.0 * COND * y)


def _rastrigin(x, y):
    return (20.0 + (x ** 2 - 10.0 * np.cos(2 * np.pi * x))
            + (y ** 2 - 10.0 * np.cos(2 * np.pi * y)))


def _rastrigin_grad(x, y):
    return (2.0 * x + 20.0 * np.pi * np.sin(2 * np.pi * x),
            2.0 * y + 20.0 * np.pi * np.sin(2 * np.pi * y))


def _ackley(x, y):
    s = x * x + y * y
    return (-20.0 * np.exp(-0.2 * np.sqrt(0.5 * s))
            - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
            + 20.0 + np.e)


def _ackley_grad(x, y):
    s = x * x + y * y
    r = np.sqrt(0.5 * s) + 1e-12
    e1 = np.exp(-0.2 * r)
    e2 = np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y)))
    dx = 2.0 * e1 * x / r + np.pi * np.sin(2 * np.pi * x) * e2
    dy = 2.0 * e1 * y / r + np.pi * np.sin(2 * np.pi * y) * e2
    return (dx, dy)


_SCH_OFFSET = 418.9829 * 2


def _schwefel(x, y):
    return _SCH_OFFSET - (x * np.sin(np.sqrt(abs(x))) + y * np.sin(np.sqrt(abs(y))))


def _schwefel_grad(x, y):
    def g(t):
        a = np.sqrt(abs(t) + 1e-12)
        return -(np.sin(a) + 0.5 * a * np.cos(a))
    return (g(x), g(y))


# --------------------------------------------------------------------------- #
# Registry. 'starts' are fixed (reproducible); 'sigma0' scales the direct-CMA-ES
# reference to the problem's domain.
# --------------------------------------------------------------------------- #
PROBLEMS = {
    "rosenbrock": dict(
        f=_rosenbrock, grad=_rosenbrock_grad, starts=[(-1.2, 1.0)],
        xmin=(1.0, 1.0), sigma0=0.5, desc="curved narrow valley"),
    "ellipsoid": dict(
        f=_ellipsoid, grad=_ellipsoid_grad, starts=[(-3.0, 2.0)],
        xmin=(0.0, 0.0), sigma0=1.0, desc="ill-conditioned (cond 1e4)"),
    "rastrigin": dict(
        f=_rastrigin, grad=_rastrigin_grad,
        starts=[(3.5, -2.5), (4.2, 3.7), (-3.1, 1.8)],
        xmin=(0.0, 0.0), sigma0=2.0, desc="multimodal lattice"),
    "ackley": dict(
        f=_ackley, grad=_ackley_grad,
        starts=[(8.0, -8.0), (5.0, 6.5), (-7.0, 3.0)],
        xmin=(0.0, 0.0), sigma0=3.0, desc="flat outer region + funnel"),
    "schwefel": dict(
        f=_schwefel, grad=_schwefel_grad,
        starts=[(-300.0, 100.0), (150.0, -250.0)],
        xmin=(420.9687, 420.9687), sigma0=120.0, desc="deceptive multimodal",
        clip=500.0),  # canonical box; outside it the loss runs off to -inf
}

ORDER = ["rosenbrock", "ellipsoid", "rastrigin", "ackley", "schwefel"]


class Benchmark:
    """Objective-blind harness: candidates see only grad and a start; we score
    the final loss, averaged over the problem's fixed starts."""

    def __init__(self, name, steps=1000):
        spec = PROBLEMS[name]
        self.name = name
        self.steps = steps
        self.f = spec["f"]
        self.grad = spec["grad"]
        self.starts = spec["starts"]
        self.xmin = spec["xmin"]
        self.sigma0 = spec["sigma0"]
        self.desc = spec["desc"]
        self.clip = float(spec.get("clip", COORD_CLIP))  # per-problem domain box
        self.fopt = float(spec.get("fopt", 0.0))         # known optimum value
        # kept for back-compat with code that expects a single start
        self.x0, self.y0 = self.starts[0]

    def evaluate(self, optimizer_func, params):
        """Run the candidate from every start; return the MEAN final loss.
        Any divergence / exception on any start -> inf (penalize brittle code).
        The final point is clipped to the problem's domain box and the loss is
        floored at the known optimum, so a candidate cannot score below f* by
        diverging into an unbounded tail (see COORD_CLIP / schwefel)."""
        losses = []
        for (x0, y0) in self.starts:
            try:
                with np.errstate(over="ignore", invalid="ignore"):
                    fx, fy = optimizer_func(self.grad, x0, y0, self.steps, params)
                    fx = np.clip(fx, -self.clip, self.clip)
                    fy = np.clip(fy, -self.clip, self.clip)
                    val = self.f(fx, fy)
                if not np.isfinite(val):
                    return float("inf")
                losses.append(max(float(val) - self.fopt, 0.0))
            except Exception:
                return float("inf")
        return float(np.mean(losses))


def direct_cmaes(bench, budget=2000, seed=1):
    """Reference: CMA-ES applied DIRECTLY to f(x,y) (needs the objective itself,
    so it is not a synthesis method). Best-so-far final loss + #evals."""
    import cma
    x0, y0 = bench.starts[0]
    es = cma.CMAEvolutionStrategy(
        [x0, y0], bench.sigma0,
        {"maxfevals": budget, "ftarget": 1e-12, "seed": seed, "verbose": -9})

    def obj(p):
        v = bench.f(np.clip(p[0], -bench.clip, bench.clip),
                    np.clip(p[1], -bench.clip, bench.clip))
        return max(float(v) - bench.fopt, 0.0) if np.isfinite(v) else 1e12

    es.optimize(obj)
    return float(es.result.fbest), int(es.result.evaluations)
