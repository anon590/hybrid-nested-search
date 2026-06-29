"""
Complementary experiment: HYBRID NESTED SEARCH with GEPA as the OUTER operator,
on the Cleanup social-dilemma task (the GEPA sibling of llm_cleanup_synth.py).

We swap our 3-shot autoresearch loop for GEPA's reflective, Pareto-frontier program
evolution as the structural operator M (Algorithm 1), keeping our CMA-ES as the
inner Tune(). Cleanup is a MAXIMIZATION of utilitarian efficiency U (collective
reward per timestep), so everything is oriented "higher U is better" (unlike the
cost-minimizing cloud drivers).

Two arms, SAME GEPA engine / seed / budget -- only the inner loop + feedback differ:

  GEPA-HYBRID (ours):  evaluator returns the CMA-ES *tuned* welfare U; the reflection
      LM sees tuned U and only chooses STRUCTURE + ranges.
  GEPA-VANILLA:        evaluator returns the *untuned* U at the policy's own declared
      constants (no inner loop).

CMA-ES fires ONLY in the hybrid arm (vanilla is tune-free; its delivered pick is
tuned once post-hoc for the de-aliasing comparison). Arms/regimes run as separate
processes.

The artifact GEPA evolves declares a tunable MANIFEST and a team policy that reads
constants ONLY from `params`, using injected SKILLS (it decides HOW MANY agents
clean vs harvest; it does not write navigation):

    MANIFEST = [{"param","low","high","guess","scale"}, ...]
    def policy(env, agent_id, params):  # return one int action in [0, 8]

OBJECTIVE-BLINDNESS is preserved: the validator forbids reading the hidden dynamics
constants (waste/apple spawn rates, depletion threshold), so the LLM cannot compute
the optimal cleaning effort analytically -- the source of the tuning gap Delta.

Run (uv; uses the LOCAL gepa repo):
  GEMINI_API_KEY=...  uv run --isolated \
    --with "$GEPA_ROOT" --with litellm \
    --with numpy --with cma --with google-genai \
    python3 gepa_hybrid_cleanup.py

  ... python3 gepa_hybrid_cleanup.py --selftest   # no API: validates the wiring
Knobs (env): MODEL (gemini-3.5-flash), REGIMES_RUN (light,moderate,heavy),
MAX_METRIC_CALLS (30), B_IN (100), MINIBATCH (2), MAX_PARALLEL (2).
"""

import os
import sys
import json
import math

import numpy as np

from synth_meta_optimizer import tune, untuned_score, fmt
from cleanup_benchmark import CleanupBenchmark, ORDER, pure_cmaes, SEED_FIXED_FRACTION
from llm_cleanup_synth import validate_code, _SAFE_BUILTINS, normalize_manifest
from cleanup_helpers import policy_namespace

MODEL = os.environ.get("MODEL", "gemini-3.5-flash")
REGIMES_RUN = [r.strip() for r in
               os.environ.get("REGIMES_RUN", ",".join(ORDER)).split(",") if r.strip()]
MAX_METRIC_CALLS = int(os.environ.get("MAX_METRIC_CALLS", "30"))
B_IN = int(os.environ.get("B_IN", "100"))
MINIBATCH = int(os.environ.get("MINIBATCH", "2"))
MAX_PARALLEL = int(os.environ.get("MAX_PARALLEL", "2"))
_BROKEN_U = -1.0e6     # welfare floor for a broken policy (GEPA maximizes U)
SEED = 1


def _default_log_path():
    import re
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return os.environ.get("LOG_PATH") or f"gepa_cleanup_log_{slug}.json"


LOG_PATH = _default_log_path()


# --------------------------------------------------------------------------- #
# Sandbox: reuse Cleanup's STRICT validator (requires `policy`, forbids imports /
# while / dunder / the hidden-dynamics attributes), which preserves the
# objective-blindness. Skills are injected via policy_namespace().
# --------------------------------------------------------------------------- #
def compile_program(code):
    """Compile a GEPA program string -> (policy fn, normalized manifest)."""
    ok, msg = validate_code(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    ns = {"__builtins__": _SAFE_BUILTINS}
    ns.update(policy_namespace())
    exec(compile(code, "<gepa-program>", "exec"), ns)
    fn = ns.get("policy")
    if not callable(fn):
        raise ValueError("policy is not callable")
    raw = ns.get("MANIFEST", [])
    try:
        manifest = normalize_manifest(raw) if raw else {}
    except Exception:
        manifest = {}
    return fn, manifest


# --------------------------------------------------------------------------- #
# Seed sketch (fixed-fraction roles, with a declared MANIFEST). STAND action = 7.
# --------------------------------------------------------------------------- #
SEED_PROGRAM = '''MANIFEST = [
    {"param": "clean_fraction", "low": 0.0, "high": 1.0, "guess": 0.3, "scale": "cont"},
]


def policy(env, agent_id, params):
    if int(env.agent_timeout[agent_id]) > 0:
        return 7
    n_c = params["clean_fraction"] * env.n_agents
    roles = assign_roles(env, n_c)
    if roles[agent_id] == "clean":
        return clean_action(env, agent_id)
    return harvest_action(env, agent_id)
'''


# --------------------------------------------------------------------------- #
# Objective / background for GEPA's reflection LM (de-hinted: NO family menu).
# --------------------------------------------------------------------------- #
OBJECTIVE = ("Evolve the STRUCTURE of a TEAM POLICY policy(env, agent_id, params) "
             "for a 'Cleanup' public-goods gridworld: N agents share reward; apples "
             "(each +1) regrow ONLY while river pollution is low enough; cleaning a "
             "waste cell costs the cleaner a little but lets apples regrow (a public "
             "good). Too few cleaners -> the river saturates and apples stop; too many "
             "-> nobody harvests. MAXIMIZE the utilitarian efficiency U (collective "
             "reward per timestep; HIGHER is better).")

_BG_COMMON = """The program MUST keep this contract:
  - Define `def policy(env, agent_id, params)` returning ONE integer action in [0, 8]
    for `agent_id` this timestep.
  - You are given SKILLS so you only decide HOW MANY agents clean vs harvest (you do
    NOT write navigation):
      waste_ratio(env) -> float in [0,1]: the global pollution fraction (your ONLY
        pollution signal).
      assign_roles(env, n_cleaners) -> {agent_id: "clean" | "harvest"} (the n_cleaners
        agents nearest the pollution clean; the rest harvest).
      clean_action(env, agent_id) -> int ;  harvest_action(env, agent_id) -> int.
      env.n_agents -> int (team size; you may scale counts by it).
      env.agent_timeout[i] -> int (>0 means agent i is frozen this step).
      Also `np` and `math`.
  - Read every tunable numeric constant from the `params` dict, and declare a
    top-level `MANIFEST = [ {"param","low","high","guess","scale"}, ... ]` listing
    them (scale: "log" multiplicative, "cont" bounded/additive like thresholds in
    [0,1] or cleaner counts). Cleaner counts may be fractional (rounded/clamped to
    [0, n_agents]).
  - Always handle the frozen case FIRST: if int(env.agent_timeout[agent_id]) > 0:
    return 7  (STAND).
  - You are BLIND to the exact dynamics (waste spawn rate, the pollution level at
    which apples stop regrowing, apple regrowth rate, and the team size are HIDDEN),
    so you CANNOT compute the optimal number of cleaners analytically. Use only
    waste_ratio(env) and env.n_agents. Do NOT read environment dynamics constants.
    No imports, no while-loops, no I/O, no dunder attributes.

Canonical body to fill in:
    def policy(env, agent_id, params):
        if int(env.agent_timeout[agent_id]) > 0:
            return 7
        wr = waste_ratio(env)
        n_cleaners = ...        # YOUR STRUCTURE: map wr (and env.n_agents) to a
                                # cleaner count using constants from params
        roles = assign_roles(env, n_cleaners)
        if roles[agent_id] == "clean":
            return clean_action(env, agent_id)
        return harvest_action(env, agent_id)"""

_BG_TUNED = """A separate zero-order tuner (CMA-ES) will choose the VALUES of your
constants within the MANIFEST ranges; you choose only the STRUCTURE and the ranges,
and set each `guess` to a sensible default. The welfare U reported back to you is
AFTER the tuner optimized the constants -- prefer high-ceiling structures even if
their default constants look mediocre."""

_BG_UNTUNED = """There is NO tuner: the `guess` values in your MANIFEST are used
as-is. You must choose both a good STRUCTURE and good concrete constant values. The
welfare U reported back to you is your policy at exactly those guessed constants."""


def background_for(arm):
    clause = _BG_TUNED if arm == "hybrid" else _BG_UNTUNED
    return f"{_BG_COMMON}\n\n{clause}"


# --------------------------------------------------------------------------- #
# Reflection LM (minimal; gemini via google.genai, else litellm).
# --------------------------------------------------------------------------- #
_STUB_PROGRAM = "```python\n" + SEED_PROGRAM + "```"


def _stub_lm(prompt):
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
# Evaluator factory. U-oriented (GEPA maximizes welfare). CMA-ES fires only in the
# hybrid arm; both untuned/tuned U recorded, only the arm one scored/fed back.
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
            untuned_U = -bench.evaluate(fn, guess)               # U = -loss
            if arm == "hybrid" and manifest:
                tuned_loss, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
                tuned_U = -float(tuned_loss)
            elif arm == "hybrid":
                tuned_U, tparams = float(untuned_U), dict(guess)  # 0-hole structure
            else:
                tuned_U, tparams = None, None                     # vanilla: not tuned
            ent = dict(ok=True, fn=fn, manifest=manifest, guess=guess,
                       untuned=float(untuned_U), tuned=tuned_U, tparams=tparams)
        except Exception as e:
            ent = dict(ok=False, err=f"{type(e).__name__}: {str(e)[:160]}")
        cache[code] = ent
        return ent

    def evaluate(candidate, example, **kwargs):
        code = candidate["program"] if isinstance(candidate, dict) else candidate
        ent = _entry(code)
        if not ent["ok"]:
            return _BROKEN_U, {"scores": {"welfare": _BROKEN_U}, "Error": ent["err"],
                               "Hint": "Fix the contract: a top-level MANIFEST and "
                                       "def policy(env, agent_id, params) returning an int in [0,8]."}
        params = ent["tparams"] if arm == "hybrid" else ent["guess"]
        try:
            U_i = float(bench._rollout_U(ent["fn"], params, bench.seeds[example["i"]]))
        except Exception:
            U_i = _BROKEN_U
        score = U_i if math.isfinite(U_i) else _BROKEN_U          # GEPA maximizes U
        side = {"scores": {"welfare_on_seed": U_i},
                "Input": {"seed": bench.seeds[example["i"]]}}
        if arm == "hybrid":
            side["Output"] = {
                "tuned_welfare_mean": round(ent["tuned"], 4),
                "untuned_welfare_mean": round(ent["untuned"], 4),
                "tuning_gain_delta": round(ent["tuned"] - ent["untuned"], 4),
                "n_tunable_holes": len(ent["manifest"]),
                "tuned_params": {k: round(v, 4) for k, v in ent["tparams"].items()},
            }
            side["Feedback"] = (
                f"After CMA-ES tuned your {len(ent['manifest'])} constant(s), mean welfare "
                f"U = {ent['tuned']:.4f} (was {ent['untuned']:.4f} at your guesses; higher "
                "is better). Improve the STRUCTURE / expose more useful holes; the tuner sets values.")
        else:
            side["Output"] = {"welfare_mean_at_your_guesses": round(ent["untuned"], 4),
                              "n_constants": len(ent["manifest"])}
            side["Feedback"] = (
                f"Your policy at its declared constants has mean welfare U = {ent['untuned']:.4f} "
                "(higher is better). Improve both the structure and the constants.")
        return score, side

    evaluate._cache = cache
    return evaluate


# --------------------------------------------------------------------------- #
# Run one GEPA arm over one regime.
# --------------------------------------------------------------------------- #
def run_gepa_arm(bench, arm, reflection_lm, run_dir):
    from gepa.optimize_anything import (
        EngineConfig, GEPAConfig, ReflectionConfig, optimize_anything)

    dataset = [{"i": i} for i in range(len(bench.seeds))]
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
    # Vanilla never tuned during the run; tune ONLY its delivered pick (max U) once
    # so the de-aliasing (H2) "tune both picks" comparison is still available.
    if arm == "vanilla" and entries:
        pick = max(entries, key=lambda e: e["untuned"])
        try:
            fn, manifest = compile_program(pick["code"])
            if manifest:
                tl, tp, _ = tune(bench, {"fn": fn, "name": "v", "manifest": manifest},
                                 budget=B_IN, seed=SEED)
                pick["tuned"], pick["tuned_params"] = -float(tl), tp
            else:
                pick["tuned"], pick["tuned_params"] = pick["untuned"], {}
        except Exception:
            pick["tuned"] = pick["untuned"]
    return entries, result


# --------------------------------------------------------------------------- #
# Reporting (welfare U, HIGHER is better)
# --------------------------------------------------------------------------- #
def report_regime(name, hybrid, vanilla, pure_U, desc):
    print("\n" + "=" * 84)
    print(f"GEPA-as-outer-operator RESULTS for regime '{name}'  ({desc})")
    print(f"  max_metric_calls={MAX_METRIC_CALLS}, B_in={B_IN}")
    print("=" * 84)
    print(f"pure CMA-ES (fixed seed {SEED_FIXED_FRACTION['name']}): U = {fmt(pure_U)}")
    if not (hybrid and vanilla):
        print("  (an arm produced no valid candidate -- skipping head-to-head)")
        return None
    hyb = max(hybrid, key=lambda e: e["tuned"])
    van = max(vanilla, key=lambda e: e["untuned"])
    dh, dv = hyb["tuned"], van["untuned"]
    print(f"  GEPA-vanilla : kept {len(vanilla):2d} structures -> best delivers "
          f"U = {fmt(dv)} (untuned guess)")
    print(f"  GEPA-hybrid  : kept {len(hybrid):2d} structures -> best delivers "
          f"U = {fmt(dh)} (CMA-ES tuned)")
    trio = {"GEPA-hybrid": dh, "GEPA-vanilla": dv, "pure CMA-ES": pure_U}
    winner = max(trio, key=trio.get)
    print(f"  => HIGHEST delivered welfare: {winner} (U = {fmt(trio[winner])})")
    van_pick_tuned = van.get("tuned")
    vpt = fmt(van_pick_tuned) if van_pick_tuned is not None else "n/a"
    print(f"  DE-ALIASING (tune both picks): vanilla-pick-tuned U={vpt}  "
          f"hybrid-pick-tuned U={fmt(hyb['tuned'])}")
    finite = [e["tuned"] - e["untuned"] for e in hybrid
              if e["tuned"] is not None
              and np.isfinite(e["untuned"]) and np.isfinite(e["tuned"])]
    e_delta = float(np.mean(finite)) if finite else float("nan")
    print(f"  REGIME E[Delta] = {fmt(e_delta)} over {len(finite)} hybrid structures")
    return dict(regime=name, pure_U=pure_U, van_delivered=dv, hyb_delivered=dh,
                van_pick_tuned=van_pick_tuned, hyb_pick_tuned=hyb["tuned"],
                mean_delta=e_delta, winner=winner)


# --------------------------------------------------------------------------- #
def _arm_worker(payload):
    """Run ONE (regime, arm) GEPA loop in its own process (own main thread)."""
    regime, arm = payload
    bench = CleanupBenchmark(regime)
    lm = make_reflection_lm(MODEL)
    entries, _ = run_gepa_arm(bench, arm, lm, f"runs/gepa_cleanup_{regime}_{arm}")
    return regime, arm, entries


def live_run():
    from concurrent.futures import ProcessPoolExecutor, as_completed
    print(f"Outer operator: GEPA  |  reflection LM: '{MODEL}'  |  "
          f"max_metric_calls={MAX_METRIC_CALLS}  |  inner CMA-ES B_in={B_IN}  |  "
          f"arms/regimes parallel (<= {MAX_PARALLEL} workers)")
    print(f"  (logging to {LOG_PATH})")
    pure, desc = {}, {}
    for name in REGIMES_RUN:
        bench = CleanupBenchmark(name)
        pu, _pp, _ = pure_cmaes(bench, SEED_FIXED_FRACTION, budget=B_IN, seed=SEED)
        pure[name] = pu
        desc[name] = bench.desc
        print(f"[pure CMA-ES] {name} ({bench.desc}): U={fmt(pu)}")

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
                         "pure_U": pure[name], "desc": desc[name]}
    with open(LOG_PATH, "w") as f:
        json.dump(results, f, indent=2)
    return results


def selftest_run():
    """No-API: validate compile_program + the evaluator/tune wiring (U-oriented)."""
    print("SELF-TEST (no API): validating the GEPA<->CMA-ES evaluator wiring.\n")
    EXTRA = {
        "proportional controller (1 hole)": '''MANIFEST = [
    {"param": "gain", "low": 0.0, "high": 3.0, "guess": 1.0, "scale": "cont"},
]


def policy(env, agent_id, params):
    if int(env.agent_timeout[agent_id]) > 0:
        return 7
    n_c = params["gain"] * waste_ratio(env) * env.n_agents
    roles = assign_roles(env, n_c)
    if roles[agent_id] == "clean":
        return clean_action(env, agent_id)
    return harvest_action(env, agent_id)
''',
        "reads hidden dynamics (should be rejected)": '''
def policy(env, agent_id, params):
    return 1 if env.waste_spawn_prob > 0.5 else 7
''',
    }
    bench = CleanupBenchmark(REGIMES_RUN[0] if REGIMES_RUN else "moderate")
    print(f"regime={bench.name}  n_agents={bench.n_agents}  steps={bench.max_steps}  seeds={bench.seeds}")
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
                    print(f"  {label:42s} score(U)={s:7.4f}  untuned U={fmt(ent['untuned'])} "
                          f"tuned U={tstr} holes={len(ent['manifest'])}")
                else:
                    print(f"  {label:42s} REJECTED: {ent.get('err')}")
            except Exception as e:
                print(f"  {label:42s} EXC {type(e).__name__}: {e}")
    print("\nSelf-test complete. Run live (with GEMINI_API_KEY) to exercise GEPA's "
          "reflection loop.")


def main():
    print("=" * 84)
    print("HYBRID NESTED SEARCH with GEPA as the OUTER structural operator (Cleanup)")
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
                                       r["pure_U"], r.get("desc", "")))
    rows = [r for r in rows if r]
    if rows:
        print("\n" + "=" * 84)
        print("SUMMARY (GEPA outer operator; delivered welfare U, HIGHER better)")
        print(f"  {'regime':10s} {'E[Delta]':>9} {'GEPA-van':>9} {'pureCMA':>9} "
              f"{'GEPA-hyb':>9}  winner")
        for r in rows:
            print(f"  {r['regime']:10s} {fmt(r['mean_delta']):>9} "
                  f"{fmt(r['van_delivered']):>9} {fmt(r['pure_U']):>9} "
                  f"{fmt(r['hyb_delivered']):>9}  {r['winner']}")


if __name__ == "__main__":
    main()
