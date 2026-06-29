"""
Complementary experiment: HYBRID NESTED SEARCH with GEPA as the OUTER operator.

Our other cloud drivers use a simple 3-shot autoresearch loop as the structural
operator M (Algorithm 1). Here we swap M for GEPA's reflective, Pareto-frontier
program evolution (the upstream ADRS engine) while keeping our CMA-ES as the
inner Tune(). This tests whether the de-aliasing advantage (Prop. 1) survives
when the structural optimizer is strong and reflective, not a toy loop.

Two arms, SAME GEPA engine / seed / budget -- only the inner loop + the feedback
GEPA sees differ:

  GEPA-HYBRID (ours):  evaluator returns the CMA-ES *tuned* cost; the reflection
      LM sees tuned costs and tuned params, and only chooses STRUCTURE + ranges.
  GEPA-VANILLA:        evaluator returns the *untuned* cost at the program's own
      declared constants; the reflection LM must pick structure AND constants.
      (~the upstream ADRS example.)

The artifact GEPA evolves is a program string that declares a tunable MANIFEST
and a `search_algorithm(src, dsts, G, num_partitions, params)` reading constants
ONLY from `params`:

    MANIFEST = [{"param","low","high","guess","scale"}, ...]
    def search_algorithm(src, dsts, G, num_partitions, params): ...

`nx` (networkx), `np`, `math`, and `BroadCastTopology` are pre-injected; a small
import whitelist is also allowed so the reflection LM can write idiomatic code.

Run (uv; uses the LOCAL gepa repo, not the stale system v0.0.4):
  GEMINI_API_KEY=...  uv run --isolated \
    --with "$GEPA_ROOT" --with litellm \
    --with numpy --with cma --with networkx --with pandas --with google-genai \
    python3 gepa_hybrid_cloudcast.py

  ... python3 gepa_hybrid_cloudcast.py --selftest   # no API: validates the
                                                    # evaluator + tune wiring
Knobs (env): MODEL (gemini-3.5-flash), REGIMES_RUN (intra,inter),
MAX_METRIC_CALLS (30), B_IN (100), MINIBATCH (2).
"""

import os
import sys
import ast
import json
import math
import importlib
import builtins
import contextlib
import io

import numpy as np

from synth_meta_optimizer import tune, untuned_score, fmt
from cloudcast_benchmark import (
    CloudcastBenchmark, ORDER, pure_cmaes, SEED_FIXED,
    nx, BroadCastTopology, _Timeout, _EVAL_TIMEOUT_S,
)
from llm_cloudcast_synth import normalize_manifest, _FORBIDDEN_NAMES, _SAFE_BUILTINS

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
REGIMES_RUN = [r.strip() for r in os.environ.get("REGIMES_RUN", "intra,inter").split(",") if r.strip()]
MAX_METRIC_CALLS = int(os.environ.get("MAX_METRIC_CALLS", "30"))
B_IN = int(os.environ.get("B_IN", "100"))
MINIBATCH = int(os.environ.get("MINIBATCH", "2"))
# Arms (and regimes) run as separate processes; cap concurrent workers. Default 2
# == "vanilla and hybrid in parallel"; raise to 4 to also overlap regimes (watch
# the LLM provider's rate limit).
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "2"))
SEED = 1


def _default_log_path():
    import re
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return os.environ.get("LOG_PATH") or f"gepa_cloudcast_log_{slug}.json"


LOG_PATH = _default_log_path()


# --------------------------------------------------------------------------- #
# Sandbox for GEPA-evolved programs. Like the autoresearch VALIDATE gate but a
# small import whitelist is allowed (the reflection LM writes idiomatic code, and
# the upstream ADRS programs import networkx). Dangerous modules/names stay blocked,
# and the per-evaluation _Timeout guards runaway structures.
# --------------------------------------------------------------------------- #
_SAFE_IMPORTS = {
    "networkx", "numpy", "math", "collections", "heapq", "itertools",
    "functools", "operator", "random",
}


def _safe_import(name, *args, **kwargs):
    root = name.split(".")[0]
    if root not in _SAFE_IMPORTS:
        raise ImportError(f"import of {name!r} is not allowed in the sandbox")
    return importlib.import_module(name)


def validate_program(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    funcs = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    if "search_algorithm" not in funcs:
        return False, "missing search_algorithm function"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                if a.name.split(".")[0] not in _SAFE_IMPORTS:
                    return False, f"import not allowed: {a.name}"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in _SAFE_IMPORTS:
                return False, f"import not allowed: from {node.module}"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder attribute access"
        elif isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"forbidden name: {node.id}"
    return True, "ok"


def compile_program(code):
    """Compile a GEPA program string -> (search_algorithm fn, normalized manifest).
    Raises ValueError on a sandbox violation or a missing/uncallable function."""
    ok, msg = validate_program(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    sb = dict(_SAFE_BUILTINS)
    sb["__import__"] = _safe_import
    ns = {"__builtins__": sb, "np": np, "numpy": np, "math": math,
          "nx": nx, "networkx": nx, "BroadCastTopology": BroadCastTopology}
    exec(compile(code, "<gepa-program>", "exec"), ns)
    fn = ns.get("search_algorithm")
    if not callable(fn):
        raise ValueError("search_algorithm is not callable")
    raw = ns.get("MANIFEST", [])
    try:
        manifest = normalize_manifest(raw) if raw else {}
    except Exception:
        manifest = {}              # malformed manifest -> treat as 0-hole structure
    return fn, manifest


# --------------------------------------------------------------------------- #
# Seed sketch (the cost/throughput-blended Dijkstra, with a declared MANIFEST).
# --------------------------------------------------------------------------- #
SEED_PROGRAM = '''MANIFEST = [
    {"param": "tput_weight", "low": 0.0, "high": 5.0, "guess": 0.5, "scale": "cont"},
]


def search_algorithm(src, dsts, G, num_partitions, params):
    w = params["tput_weight"]
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    for u, v, d in h.edges(data=True):
        d["w"] = d["cost"] + w * (1.0 / (d.get("throughput") or 1e-9))
    bc = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="w")
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(num_partitions):
                bc.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    return bc
'''


# --------------------------------------------------------------------------- #
# Objective / background shown to GEPA's reflection LM (arm-specific clause).
# --------------------------------------------------------------------------- #
OBJECTIVE = ("Evolve the STRUCTURE of a multi-cloud broadcast routing algorithm "
             "search_algorithm(src, dsts, G, num_partitions, params) that delivers "
             "a fixed data volume from one source region to many destinations at "
             "minimum total dollar cost (egress + a small instance cost). G is a "
             "networkx DiGraph whose edges carry 'cost' ($/GB) and 'throughput' "
             "(Gbps). An edge used by several (dst, partition) pairs is billed once.")

_BG_COMMON = """The program MUST keep this contract:
  - Define `def search_algorithm(src, dsts, G, num_partitions, params)` returning a
    BroadCastTopology with a path for EVERY (destination, partition).
  - Read every tunable numeric constant from the `params` dict.
  - Declare a top-level `MANIFEST = [ {"param","low","high","guess","scale"}, ... ]`
    listing those constants (scale: "log" for multiplicative gains, "cont" for
    bounded/additive). The param names MUST match the keys read from `params`.
  - `nx` (networkx), `np`, `math`, and `BroadCastTopology` are already available;
    you may import only from {networkx, numpy, math, collections, heapq, itertools,
    functools, operator, random}. No other imports, file/network I/O, or eval/exec.
  - Avoid exponential enumeration over the ~70-node mesh (there is a hard time budget)."""

_BG_TUNED = """A separate zero-order tuner (CMA-ES) will choose the VALUES of your
constants within the MANIFEST ranges; you choose only the STRUCTURE and the ranges,
and set each `guess` to a sensible default. The cost reported back to you is AFTER
the tuner optimized the constants -- so prefer high-ceiling structures even if their
default constants look mediocre."""

_BG_UNTUNED = """There is NO tuner: the `guess` values in your MANIFEST are used
as-is. You must choose both a good STRUCTURE and good concrete constant values. The
cost reported back to you is your program at exactly those guessed constants."""


def background_for(arm):
    clause = _BG_TUNED if arm == "hybrid" else _BG_UNTUNED
    return f"{_BG_COMMON}\n\n{clause}"


# --------------------------------------------------------------------------- #
# Reflection LM (minimal; mirrors examples/adrs/.../utils/lm.py)
# --------------------------------------------------------------------------- #
_STUB_PROGRAM = "```python\n" + SEED_PROGRAM + "```"


def _stub_lm(prompt):
    """Module-level (picklable) no-API reflection LM for testing the loop wiring."""
    return _STUB_PROGRAM


def make_reflection_lm(model):
    if os.environ.get("GEPA_STUB_LM") == "1":
        return _stub_lm
    if "gemini" in model:
        from google import genai
        from google.genai.types import HttpOptions
        client = genai.Client(http_options=HttpOptions(api_version="v1"))

        def call(prompt):
            text = (prompt if isinstance(prompt, str)
                    else "\n".join(f"{m.get('role','user')}: {m.get('content','')}"
                                    for m in prompt))
            r = client.models.generate_content(model=model, contents=text)
            return r.text or ""
        return call

    import litellm

    def call(prompt):
        msgs = prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}]
        return litellm.completion(model=model, messages=msgs).choices[0].message.content or ""
    return call


# --------------------------------------------------------------------------- #
# Evaluator factory: wraps the inner CMA-ES tune, with a per-program cache so a
# given structure is tuned ONCE (not once per example). Both untuned and tuned
# costs are recorded for analysis; only the arm-appropriate one is scored/fed back.
# --------------------------------------------------------------------------- #
def make_evaluator(bench, arm):
    cache = {}

    def _entry(code):
        ent = cache.get(code)
        if ent is not None:
            return ent
        try:
            fn, manifest = compile_program(code)
            cand = {"fn": fn, "name": "gepa", "manifest": manifest}
            guess = {k: manifest[k]["guess"] for k in manifest}
            untuned = bench.evaluate(fn, guess)
            # CMA-ES fires ONLY in the hybrid arm (and only if the structure has
            # holes); the vanilla arm never runs the inner loop -- faithful and
            # cheaper. Vanilla records tuned=None (its delivered pick is tuned once
            # post-hoc in run_gepa_arm for the de-aliasing comparison only).
            if arm == "hybrid" and manifest:
                tuned, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
                tuned, tparams = float(tuned), tparams
            elif arm == "hybrid":
                tuned, tparams = float(untuned), dict(guess)   # 0-hole structure
            else:
                tuned, tparams = None, None                    # vanilla: not tuned
            ent = dict(ok=True, fn=fn, manifest=manifest, guess=guess,
                       untuned=float(untuned), tuned=tuned, tparams=tparams)
        except Exception as e:
            ent = dict(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}")
        cache[code] = ent
        return ent

    def evaluate(candidate, example, **kwargs):
        code = candidate["program"] if isinstance(candidate, dict) else candidate
        ent = _entry(code)
        if not ent["ok"]:
            return 0.0, {"scores": {"cost": float("inf")}, "Error": ent["err"],
                         "Hint": "Fix the contract: a top-level MANIFEST and "
                                 "def search_algorithm(src,dsts,G,num_partitions,params)."}
        params = ent["tparams"] if arm == "hybrid" else ent["guess"]
        cfg = bench.configs[example["i"]]
        cost_i = bench._cost_one(ent["fn"], params, cfg)
        score = 1.0 / (1.0 + cost_i) if math.isfinite(cost_i) else 0.0
        side = {
            "scores": {"cost_on_config": cost_i},
            "Input": {"config": bench.config_names[example["i"]]},
        }
        if arm == "hybrid":
            side["Output"] = {
                "tuned_cost_mean": round(ent["tuned"], 3),
                "untuned_cost_mean": round(ent["untuned"], 3),
                "tuning_gain_delta": round(ent["untuned"] - ent["tuned"], 3),
                "n_tunable_holes": len(ent["manifest"]),
                "tuned_params": {k: round(v, 4) for k, v in ent["tparams"].items()},
            }
            side["Feedback"] = (
                f"After CMA-ES tuned your {len(ent['manifest'])} constant(s), mean cost "
                f"= {ent['tuned']:.2f} (was {ent['untuned']:.2f} at your guesses). "
                "Improve the STRUCTURE / expose more useful holes; the tuner sets values.")
        else:
            side["Output"] = {
                "cost_mean_at_your_guesses": round(ent["untuned"], 3),
                "n_constants": len(ent["manifest"]),
            }
            side["Feedback"] = (
                f"Your program at its declared constants has mean cost "
                f"= {ent['untuned']:.2f}. Improve both the structure and the constants.")
        return score, side

    evaluate._cache = cache
    return evaluate


# --------------------------------------------------------------------------- #
# Run one GEPA arm over one regime.
# --------------------------------------------------------------------------- #
def run_gepa_arm(bench, arm, reflection_lm, run_dir):
    from gepa.optimize_anything import (
        EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything)

    dataset = [{"i": i} for i in range(len(bench.configs))]
    evaluator = make_evaluator(bench, arm)
    cfg = GEPAConfig(
        engine=EngineConfig(
            run_dir=run_dir, seed=0, max_metric_calls=MAX_METRIC_CALLS,
            track_best_outputs=True, display_progress_bar=False,
            parallel=False, raise_on_exception=False),
        reflection=ReflectionConfig(
            reflection_minibatch_size=MINIBATCH, reflection_lm=reflection_lm,
            skip_perfect_score=False),
    )
    result = optimize_anything(
        seed_candidate={"program": SEED_PROGRAM},
        evaluator=evaluator, dataset=dataset, valset=dataset,
        objective=OBJECTIVE, background=background_for(arm), config=cfg)

    # Assemble entries from EVERY distinct structure GEPA explored (the evaluator
    # cache), not just the kept frontier -- so best-discovered + E[Delta] are
    # complete and consistent with the autoresearch arms (which min over proposals).
    entries = []
    for idx, (code, ent) in enumerate(evaluator._cache.items()):
        if not ent.get("ok"):
            continue
        entries.append(dict(name=f"gepa#{idx}", code=code,
                            untuned=ent["untuned"], tuned=ent["tuned"],
                            tuned_params=ent["tparams"], n_holes=len(ent["manifest"])))
    # Vanilla never tuned during the run; tune ONLY its delivered pick once so the
    # de-aliasing (H2) "tune both picks" comparison is still available.
    if arm == "vanilla" and entries:
        pick = min(entries, key=lambda e: e["untuned"])
        try:
            fn, manifest = compile_program(pick["code"])
            if manifest:
                tv, tp, _ = tune(bench, {"fn": fn, "name": "v", "manifest": manifest},
                                 budget=B_IN, seed=SEED)
                pick["tuned"], pick["tuned_params"] = float(tv), tp
            else:
                pick["tuned"], pick["tuned_params"] = pick["untuned"], {}
        except Exception:
            pick["tuned"] = pick["untuned"]
    return entries, result


# --------------------------------------------------------------------------- #
# Reporting (cost, lower is better)
# --------------------------------------------------------------------------- #
def report_regime(name, hybrid, vanilla, pure_cost, pure_params):
    print("\n" + "=" * 84)
    print(f"GEPA-as-outer-operator RESULTS for regime '{name}'  "
          f"(max_metric_calls={MAX_METRIC_CALLS}, B_in={B_IN})")
    print("=" * 84)
    print(f"pure CMA-ES (fixed seed {SEED_FIXED['name']}): cost = {fmt(pure_cost)}")
    if not (hybrid and vanilla):
        print("  (an arm produced no valid candidate -- skipping head-to-head)")
        return None
    hyb = min(hybrid, key=lambda e: e["tuned"])
    van = min(vanilla, key=lambda e: e["untuned"])
    dh, dv = hyb["tuned"], van["untuned"]
    print(f"  GEPA-vanilla : kept {len(vanilla):2d} structures -> best delivers "
          f"{fmt(dv)} (untuned guess)")
    print(f"  GEPA-hybrid  : kept {len(hybrid):2d} structures -> best delivers "
          f"{fmt(dh)} (CMA-ES tuned)")
    trio = {"GEPA-hybrid": dh, "GEPA-vanilla": dv, "pure CMA-ES": pure_cost}
    winner = min(trio, key=trio.get)
    print(f"  => LOWEST delivered cost: {winner} ({fmt(trio[winner])})")
    van_pick_tuned = van.get("tuned")   # vanilla's pick is tuned once post-hoc
    vpt = fmt(van_pick_tuned) if van_pick_tuned is not None else "n/a"
    print(f"  DE-ALIASING (tune both picks): vanilla-pick-tuned={vpt}  "
          f"hybrid-pick-tuned={fmt(hyb['tuned'])}")
    # E[Delta] over the HYBRID exploration (vanilla isn't tuned per-structure).
    finite = [e["untuned"] - e["tuned"] for e in hybrid
              if e["tuned"] is not None
              and np.isfinite(e["untuned"]) and np.isfinite(e["tuned"])]
    e_delta = float(np.mean(finite)) if finite else float("nan")
    print(f"  REGIME E[Delta] = {fmt(e_delta)} over {len(finite)} hybrid structures")
    return dict(regime=name, pure_cost=pure_cost, van_delivered=dv, hyb_delivered=dh,
                van_pick_tuned=van_pick_tuned, hyb_pick_tuned=hyb["tuned"],
                mean_delta=e_delta, winner=winner)


# --------------------------------------------------------------------------- #
def _arm_worker(payload):
    """Run ONE (regime, arm) GEPA loop in its own process. Returns picklable
    entries (code strings + costs). Each worker builds its own reflection LM, and
    runs on the process's main thread -- so the per-eval SIGALRM timeout still
    applies (unlike thread-based parallelism)."""
    regime, arm = payload
    bench = CloudcastBenchmark(regime)
    lm = make_reflection_lm(MODEL)
    entries, _ = run_gepa_arm(bench, arm, lm, f"runs/gepa_cc_{regime}_{arm}")
    return regime, arm, entries


def live_run():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"Outer operator: GEPA  |  reflection LM: '{MODEL}'  |  "
          f"max_metric_calls={MAX_METRIC_CALLS}  |  inner CMA-ES B_in={B_IN}  |  "
          f"arms/regimes parallel (<= {MAX_PARALLEL} workers)")
    print(f"  (logging to {LOG_PATH})")
    # Pure-CMA-ES reference per regime: cheap, computed sequentially in the parent.
    pure = {}
    for name in REGIMES_RUN:
        bench = CloudcastBenchmark(name)
        pc, pp, _ = pure_cmaes(bench, SEED_FIXED, budget=B_IN, seed=SEED)
        pure[name] = (pc, pp)
        print(f"[pure CMA-ES] {name} ({bench.desc}): cost={fmt(pc)}")

    combos = [(name, arm) for name in REGIMES_RUN for arm in ("vanilla", "hybrid")]
    print(f"\nLaunching {len(combos)} GEPA arm-runs in parallel "
          f"({MAX_PARALLEL} at a time)...")
    collected = {}
    with ProcessPoolExecutor(max_workers=min(MAX_PARALLEL, len(combos))) as ex:
        futs = {ex.submit(_arm_worker, c): c for c in combos}
        for fut in as_completed(futs):
            rname, arm, entries = fut.result()
            collected[(rname, arm)] = entries
            print(f"  [done] {rname}/{arm}: {len(entries)} structures explored")

    results = {}
    for name in REGIMES_RUN:
        pc, pp = pure[name]
        results[name] = {"hybrid": collected.get((name, "hybrid"), []),
                         "vanilla": collected.get((name, "vanilla"), []),
                         "pure_cost": pc, "pure_params": pp}
    with open(LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


def selftest_run():
    """No-API: validate compile_program + the evaluator/tune wiring on the seed
    (and a couple of hand-written program strings) for both arms."""
    print("SELF-TEST (no API): validating the GEPA<->CMA-ES evaluator wiring.\n")
    EXTRA = {
        "no-manifest baseline (0 holes)": '''
def search_algorithm(src, dsts, G, num_partitions, params):
    h = G.copy()
    h.remove_edges_from(list(h.in_edges(src)) + list(nx.selfloop_edges(h)))
    bc = BroadCastTopology(src, dsts, num_partitions)
    for dst in dsts:
        path = nx.dijkstra_path(h, src, dst, weight="cost")
        for i in range(len(path) - 1):
            s, t = path[i], path[i + 1]
            for j in range(num_partitions):
                bc.append_dst_partition_path(dst, j, [s, t, G[s][t]])
    return bc
''',
        "illegal import (should be rejected)": '''
import os
def search_algorithm(src, dsts, G, num_partitions, params):
    return None
''',
    }
    bench = CloudcastBenchmark(REGIMES_RUN[0] if REGIMES_RUN else "inter")
    print(f"regime={bench.name}  configs={bench.config_names}")
    for arm in ("vanilla", "hybrid"):
        ev = make_evaluator(bench, arm)
        progs = {"SEED_PROGRAM": SEED_PROGRAM, **EXTRA}
        print(f"\n--- arm={arm} ---")
        for label, code in progs.items():
            try:
                s, side = ev({"program": code}, {"i": 0})
                ent = ev._cache.get(code, {})
                if ent.get("ok"):
                    tval = ent["tuned"]
                    tstr = fmt(tval) if tval is not None else "n/a (vanilla)"
                    print(f"  {label:38s} score={s:.5f}  untuned={fmt(ent['untuned'])} "
                          f"tuned={tstr} holes={len(ent['manifest'])}")
                else:
                    print(f"  {label:38s} REJECTED: {ent.get('err')}")
            except Exception as e:
                print(f"  {label:38s} EXC {type(e).__name__}: {e}")
    print("\nSelf-test complete: compile_program, manifest parsing, and the "
          "tuned/untuned evaluator paths all work. Run live (with GEMINI_API_KEY) "
          "to exercise GEPA's reflection loop.")


def main():
    print("=" * 84)
    print("HYBRID NESTED SEARCH with GEPA as the OUTER structural operator (Cloudcast)")
    print("=" * 84)
    if "--selftest" in sys.argv:
        selftest_run()
        return
    results = live_run()
    rows = []
    for name in REGIMES_RUN:
        if name in results:
            r = results[name]
            rows.append(report_regime(name, r["hybrid"], r["vanilla"],
                                       r["pure_cost"], r["pure_params"]))
    rows = [r for r in rows if r]
    if rows:
        print("\n" + "=" * 84)
        print("SUMMARY (GEPA outer operator; delivered cost, lower better)")
        print(f"  {'regime':8s} {'E[Delta]':>9} {'GEPA-van':>9} {'pureCMA':>9} "
              f"{'GEPA-hyb':>9}  winner")
        for r in rows:
            print(f"  {r['regime']:8s} {fmt(r['mean_delta']):>9} "
                  f"{fmt(r['van_delivered']):>9} {fmt(r['pure_cost']):>9} "
                  f"{fmt(r['hyb_delivered']):>9}  {r['winner']}")


if __name__ == "__main__":
    main()
