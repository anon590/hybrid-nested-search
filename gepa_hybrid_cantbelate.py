"""
Complementary experiment: HYBRID NESTED SEARCH with GEPA as the OUTER operator,
on the "Can't Be Late" cloud-scheduling task (sibling of gepa_hybrid_cloudcast.py).

We swap our 3-shot autoresearch loop for GEPA's reflective, Pareto-frontier program
evolution as the structural operator M (Algorithm 1), keeping our CMA-ES as the
inner Tune(). Two arms, SAME GEPA engine / seed / budget -- only the inner loop +
the feedback differ:

  GEPA-HYBRID (ours):  evaluator returns the CMA-ES *tuned* cost; the reflection LM
      sees tuned costs and only chooses STRUCTURE + ranges.
  GEPA-VANILLA:        evaluator returns the *untuned* cost at the program's own
      declared constants (no inner loop; ~the upstream ADRS baseline).

CMA-ES fires ONLY in the hybrid arm (vanilla is tune-free; its delivered pick is
tuned once post-hoc for the de-aliasing comparison). Arms/regimes run as separate
processes (so each keeps its own main thread; cheap rollouts, no shared state).

The artifact GEPA evolves declares a tunable MANIFEST and a decision rule reading
constants ONLY from `params`:

    MANIFEST = [{"param","low","high","guess","scale"}, ...]
    def decide(obs, params):  # return 0=NONE, 1=SPOT, 2=ON_DEMAND for this tick

`np` and `math` are pre-injected; a small import whitelist is allowed. The
objective/background carry NO structure hints (no family menu) -- matching the
de-hinted prompts.

Run (uv; uses the LOCAL gepa repo):
  GEMINI_API_KEY=...  uv run --isolated \
    --with "$GEPA_ROOT" --with litellm \
    --with numpy --with cma --with google-genai \
    python3 gepa_hybrid_cantbelate.py

  ... python3 gepa_hybrid_cantbelate.py --selftest   # no API: validates the wiring
Knobs (env): MODEL (gemini-3.5-flash), REGIMES_RUN (cheap_restart,moderate,
costly_restart), MAX_METRIC_CALLS (30), B_IN (100), MINIBATCH (2),
GEPA_TRACES (6 -- traces used as GEPA examples + tune set), MAX_PARALLEL (2).
"""

import os
import sys
import ast
import json
import math
import importlib

import numpy as np

from synth_meta_optimizer import tune, untuned_score, fmt
from cantbelate_benchmark import CantBeLateBenchmark, ORDER, pure_cmaes, SEED_FIXED
from llm_cantbelate_synth import normalize_manifest, _FORBIDDEN_NAMES, _SAFE_BUILTINS

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
REGIMES_RUN = [r.strip() for r in
               os.environ.get("REGIMES_RUN", ",".join(ORDER)).split(",") if r.strip()]
MAX_METRIC_CALLS = int(os.environ.get("MAX_METRIC_CALLS", "30"))
B_IN = int(os.environ.get("B_IN", "100"))
MINIBATCH = int(os.environ.get("MINIBATCH", "2"))
# Traces used as GEPA examples AND as the inner-tune set. Kept small so the
# max_metric_calls budget yields enough proposals (raise for a less noisy estimate).
GEPA_TRACES = int(os.environ.get("GEPA_TRACES", "6"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "2"))
SEED = 1


def _default_log_path():
    import re
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return os.environ.get("LOG_PATH") or f"gepa_cantbelate_log_{slug}.json"


LOG_PATH = _default_log_path()


# --------------------------------------------------------------------------- #
# Sandbox for GEPA-evolved programs. A decision rule needs no imports, but a small
# whitelist is tolerated; while-loops are FORBIDDEN (a rule runs every tick and
# there is no per-eval timeout here, so a stray loop would hang the rollout).
# --------------------------------------------------------------------------- #
_SAFE_IMPORTS = {
    "numpy", "math", "collections", "heapq", "itertools", "functools",
    "operator", "random",
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
    if "decide" not in funcs:
        return False, "missing decide function"
    for node in ast.walk(tree):
        if isinstance(node, ast.While):
            return False, "while-loops are not allowed"
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
    """Compile a GEPA program string -> (decide fn, normalized manifest)."""
    ok, msg = validate_program(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    sb = dict(_SAFE_BUILTINS)
    sb["__import__"] = _safe_import
    ns = {"__builtins__": sb, "np": np, "numpy": np, "math": math}
    exec(compile(code, "<gepa-program>", "exec"), ns)
    fn = ns.get("decide")
    if not callable(fn):
        raise ValueError("decide is not callable")
    raw = ns.get("MANIFEST", [])
    try:
        manifest = normalize_manifest(raw) if raw else {}
    except Exception:
        manifest = {}
    return fn, manifest


# --------------------------------------------------------------------------- #
# Seed sketch (slack-threshold buffer, with a declared MANIFEST).
# --------------------------------------------------------------------------- #
SEED_PROGRAM = '''MANIFEST = [
    {"param": "buffer", "low": 0.0, "high": 86400.0, "guess": 7200.0, "scale": "cont"},
]


def decide(obs, params):
    if obs["remaining_task_time"] <= 1e-3:
        return 0
    if obs["slack"] <= params["buffer"]:
        return 2
    return 1 if obs["has_spot"] else 0
'''


# --------------------------------------------------------------------------- #
# Objective / background for GEPA's reflection LM (de-hinted: NO family menu).
# --------------------------------------------------------------------------- #
OBJECTIVE = ("Evolve the STRUCTURE of a per-timestep scheduling rule "
             "decide(obs, params) for the 'Can't Be Late' problem: finish a compute "
             "job before a hard deadline at minimum dollar cost. Each tick choose "
             "0=NONE (wait, free, no progress), 1=SPOT (cheaper per hour but "
             "preemptible; progresses only while spot is available, and relaunching "
             "after an outage costs restart_overhead of lost compute), or 2=ON_DEMAND "
             "(more expensive per hour but always progresses).")

_BG_COMMON = """The program MUST keep this contract:
  - Define `def decide(obs, params)` returning an int in {0, 1, 2} for the current tick.
  - `obs` is a dict (all times in SECONDS) with keys: remaining_task_time,
    remaining_time, slack (= remaining_time - remaining_task_time), restart_overhead,
    remaining_restart_overhead, has_spot (bool: spot available NOW),
    last_cluster_type (0/1/2), gap_seconds, elapsed_seconds, deadline, task_duration.
  - Read every tunable numeric constant from the `params` dict.
  - Declare a top-level `MANIFEST = [ {"param","low","high","guess","scale"}, ... ]`
    listing those constants (scale: "log" for multiplicative gains, "cont" for
    bounded/additive, e.g. time buffers in seconds or fractions in [0,1]). The param
    names MUST match the keys read from `params`.
  - Use only `obs`, `params`, `np`, `math` (you may import only from {numpy, math,
    collections, heapq, itertools, functools, operator, random}). No while-loops,
    no file/network I/O, no eval/exec, no dunder attributes.
  - You are BLIND to the future spot pattern and the exact prices, so you cannot
    compute the optimal thresholds analytically. A separate STRONG GUARANTEE forces
    ON_DEMAND if the deadline would otherwise be missed -- you cannot miss the
    deadline, so your only job is to MINIMIZE COST."""

_BG_TUNED = """A separate zero-order tuner (CMA-ES) will choose the VALUES of your
constants within the MANIFEST ranges; you choose only the STRUCTURE and the ranges,
and set each `guess` to a sensible default. The cost reported back to you is AFTER
the tuner optimized the constants -- prefer high-ceiling structures even if their
default constants look mediocre."""

_BG_UNTUNED = """There is NO tuner: the `guess` values in your MANIFEST are used
as-is. You must choose both a good STRUCTURE and good concrete constant values. The
cost reported back to you is your rule at exactly those guessed constants."""


def background_for(arm):
    clause = _BG_TUNED if arm == "hybrid" else _BG_UNTUNED
    return f"{_BG_COMMON}\n\n{clause}"


# --------------------------------------------------------------------------- #
# Reflection LM (minimal; gemini via google.genai, else litellm).
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
# Evaluator factory: wraps the inner CMA-ES tune with a per-program cache; CMA-ES
# fires only in the hybrid arm. Both costs recorded; only the arm one is scored/fed.
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
                                 "def decide(obs, params) returning 0/1/2."}
        params = ent["tparams"] if arm == "hybrid" else ent["guess"]
        try:
            cost_i = bench._rollout_cost(ent["fn"], params, bench.traces[example["i"]])
        except Exception:
            cost_i = float("inf")
        score = 1.0 / (1.0 + cost_i) if math.isfinite(cost_i) else 0.0
        side = {"scores": {"cost_on_trace": cost_i},
                "Input": {"trace": os.path.basename(bench.traces[example["i"]])}}
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
            side["Output"] = {"cost_mean_at_your_guesses": round(ent["untuned"], 3),
                              "n_constants": len(ent["manifest"])}
            side["Feedback"] = (
                f"Your rule at its declared constants has mean cost = {ent['untuned']:.2f}. "
                "Improve both the structure and the constants.")
        return score, side

    evaluate._cache = cache
    return evaluate


# --------------------------------------------------------------------------- #
# Run one GEPA arm over one regime.
# --------------------------------------------------------------------------- #
def run_gepa_arm(bench, arm, reflection_lm, run_dir):
    from gepa.optimize_anything import (
        EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything)

    dataset = [{"i": i} for i in range(len(bench.traces))]
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
def report_regime(name, hybrid, vanilla, pure_cost, desc):
    print("\n" + "=" * 84)
    print(f"GEPA-as-outer-operator RESULTS for regime '{name}'  ({desc})")
    print(f"  max_metric_calls={MAX_METRIC_CALLS}, B_in={B_IN}, traces={GEPA_TRACES}")
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
    van_pick_tuned = van.get("tuned")
    vpt = fmt(van_pick_tuned) if van_pick_tuned is not None else "n/a"
    print(f"  DE-ALIASING (tune both picks): vanilla-pick-tuned={vpt}  "
          f"hybrid-pick-tuned={fmt(hyb['tuned'])}")
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
    """Run ONE (regime, arm) GEPA loop in its own process (own main thread)."""
    regime, arm = payload
    bench = CantBeLateBenchmark(regime, n_traces=GEPA_TRACES)
    lm = make_reflection_lm(MODEL)
    entries, _ = run_gepa_arm(bench, arm, lm, f"runs/gepa_cbl_{regime}_{arm}")
    return regime, arm, entries


def live_run():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"Outer operator: GEPA  |  reflection LM: '{MODEL}'  |  "
          f"max_metric_calls={MAX_METRIC_CALLS}  |  inner CMA-ES B_in={B_IN}  |  "
          f"traces={GEPA_TRACES}  |  arms/regimes parallel (<= {MAX_PARALLEL} workers)")
    print(f"  (logging to {LOG_PATH})")
    pure, desc = {}, {}
    for name in REGIMES_RUN:
        bench = CantBeLateBenchmark(name, n_traces=GEPA_TRACES)
        pc, _pp, _ = pure_cmaes(bench, SEED_FIXED, budget=B_IN, seed=SEED)
        pure[name] = pc
        desc[name] = bench.desc
        print(f"[pure CMA-ES] {name} ({bench.desc}): cost={fmt(pc)}")

    combos = [(name, arm) for name in REGIMES_RUN for arm in ("vanilla", "hybrid")]
    print(f"\nLaunching {len(combos)} GEPA arm-runs in parallel ({MAX_PARALLEL} at a time)...")
    collected = {}
    with ProcessPoolExecutor(max_workers=min(MAX_PARALLEL, len(combos))) as ex:
        futs = {ex.submit(_arm_worker, c): c for c in combos}
        for fut in as_completed(futs):
            rname, arm, entries = fut.result()
            collected[(rname, arm)] = entries
            print(f"  [done] {rname}/{arm}: {len(entries)} structures explored")

    results = {}
    for name in REGIMES_RUN:
        results[name] = {"hybrid": collected.get((name, "hybrid"), []),
                         "vanilla": collected.get((name, "vanilla"), []),
                         "pure_cost": pure[name], "desc": desc[name]}
    with open(LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


def selftest_run():
    """No-API: validate compile_program + the evaluator/tune wiring on the seed
    (and hand-written rules) for both arms."""
    print("SELF-TEST (no API): validating the GEPA<->CMA-ES evaluator wiring.\n")
    EXTRA = {
        "no-manifest greedy spot (0 holes)": '''
def decide(obs, params):
    if obs["remaining_task_time"] <= 1e-3:
        return 0
    return 1 if obs["has_spot"] else 0
''',
        "illegal while-loop (should be rejected)": '''
def decide(obs, params):
    while True:
        pass
    return 0
''',
    }
    bench = CantBeLateBenchmark(REGIMES_RUN[0] if REGIMES_RUN else "moderate",
                                n_traces=GEPA_TRACES)
    print(f"regime={bench.name}  traces={len(bench.traces)}  overhead={bench.overhead}h")
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
                    print(f"  {label:42s} score={s:.5f}  untuned={fmt(ent['untuned'])} "
                          f"tuned={tstr} holes={len(ent['manifest'])}")
                else:
                    print(f"  {label:42s} REJECTED: {ent.get('err')}")
            except Exception as e:
                print(f"  {label:42s} EXC {type(e).__name__}: {e}")
    print("\nSelf-test complete. Run live (with GEMINI_API_KEY) to exercise GEPA's "
          "reflection loop.")


def main():
    print("=" * 84)
    print("HYBRID NESTED SEARCH with GEPA as the OUTER structural operator (Can't Be Late)")
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
                                       r["pure_cost"], r.get("desc", "")))
    rows = [r for r in rows if r]
    if rows:
        print("\n" + "=" * 84)
        print("SUMMARY (GEPA outer operator; delivered cost, lower better)")
        print(f"  {'regime':16s} {'E[Delta]':>9} {'GEPA-van':>9} {'pureCMA':>9} "
              f"{'GEPA-hyb':>9}  winner")
        for r in rows:
            print(f"  {r['regime']:16s} {fmt(r['mean_delta']):>9} "
                  f"{fmt(r['van_delivered']):>9} {fmt(r['pure_cost']):>9} "
                  f"{fmt(r['hyb_delivered']):>9}  {r['winner']}")


if __name__ == "__main__":
    main()
