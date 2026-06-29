"""
LLM-driven Hybrid Nested Search on the "Can't Be Late" cloud-scheduling problem.

Scheduling instantiation of the framework (paper.pdf, Sec. 6), the cloud sibling
of llm_cleanup_synth.py. The artifact under search is a per-step decision rule
    decide(obs, params) -> int in {0:NONE, 1:SPOT, 2:ON_DEMAND}
whose STRUCTURE (how the current state maps to the choice) is proposed by an LLM
(the outer structural operator M of Algorithm 1) and whose continuous holes
`params` (slack buffers, pressure thresholds, hysteresis bands) are tuned by the
inner CMA-ES.

Provider-agnostic outer operator (set LLM_PROVIDER), identical to the other
experiments:
  * gemini      -> google-genai SDK          (GEMINI_API_KEY / GOOGLE_API_KEY)
  * openrouter  -> any model via OpenAI API  (OPENROUTER_API_KEY) e.g. z-ai/glm-5.2
  * openai      -> OpenAI directly           (OPENAI_API_KEY)

THREE ARMS are compared (paper Sec. 6.2):

  HYBRID (ours, Arm 3):   feedback = CMA-ES *tuned* cost C_hat(tau).
  VANILLA AUTORESEARCH (Arm 1): feedback = *untuned* cost at the LLM's guess.
  PURE CMA-ES (vanilla numerical, Arm 2): CMA-ES over a FIXED seed structure
      (slack-threshold buffer); no structural search.

OBJECTIVE-BLINDNESS: the rule sees only the CURRENT per-step state and never the
trace's future spot pattern or the prices, so it cannot place the buffers /
thresholds analytically -- that is the tuning gap Delta the hybrid arm de-aliases.
The simulator hard-enforces the deadline (a strong-guarantee override), so the
objective is cleanly cost only.

Run (Gemini):
  uv run --isolated --with numpy --with cma --with google-genai \
      python3 llm_cantbelate_synth.py

Run (OpenRouter, e.g. z-ai/glm-5.2):
  LLM_PROVIDER=openrouter MODEL=z-ai/glm-5.2 \
  uv run --isolated --with numpy --with cma --with openai \
      python3 llm_cantbelate_synth.py

  python3 llm_cantbelate_synth.py --replay     # reuse the log, no API
  python3 llm_cantbelate_synth.py --selftest    # no API: hand-written sketches
"""

import os
import sys
import re
import ast
import json
import math
import builtins

import numpy as np

from synth_meta_optimizer import tune, untuned_score, fmt
from cantbelate_benchmark import (
    CantBeLateBenchmark, REGIMES, ORDER, pure_cmaes, SEED_FIXED, SEEDS,
)

# --------------------------------------------------------------------------- #
# Provider / model selection (identical to llm_cleanup_synth.py).
# --------------------------------------------------------------------------- #
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODEL = {
    "gemini": "gemini-3.5-flash",
    "openrouter": "z-ai/glm-5.2",
    "openai": "gpt-4o-mini",
}
MODEL = (os.environ.get("MODEL")
         or os.environ.get("GEMINI_MODEL")
         or _DEFAULT_MODEL.get(PROVIDER, "gemini-3.5-flash"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "32000"))

N_PROPOSALS = int(os.environ.get("N_PROPOSALS", "3"))
REGIMES_RUN = [r.strip() for r in
               os.environ.get("REGIMES_RUN", ",".join(ORDER)).split(",")
               if r.strip()]
B_IN = int(os.environ.get("B_IN", "100"))             # inner CMA-ES budget
PURE_BUDGET = int(os.environ.get("PURE_BUDGET", "120"))
SEED = 1


def _default_log_path():
    if PROVIDER == "gemini" and MODEL.startswith("gemini-3.5"):
        return "llm_cantbelate_log.json"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return f"llm_cantbelate_log_{slug}.json"


LOG_PATH = os.environ.get("LOG_PATH") or _default_log_path()


# --------------------------------------------------------------------------- #
# Prompts -- the LLM knows the scheduling problem; the only per-arm difference is
# who sets the constants (tuner vs the LLM itself).
# --------------------------------------------------------------------------- #
_HEAD = """You are an expert in cloud cost optimization proposing the STRUCTURE
(control flow + decision rule) of a scheduling policy for the "Can't Be Late"
problem: finish a long compute job before a hard deadline at minimum dollar cost.

At each timestep you choose ONE of three actions (return the integer):
    0 = NONE       wait this tick (no instance, no cost, no progress)
    1 = SPOT       cheaper per hour but preemptible: it makes progress ONLY while
                   spot is available right now, and relaunching after an outage
                   costs `restart_overhead` of lost compute time
    2 = ON_DEMAND  more expensive per hour but reliable: always makes progress

Write exactly this function:
    def decide(obs, params):
        # return 0, 1, or 2 for THIS timestep

`obs` is a dict of the CURRENT state (all times in SECONDS):
    obs["remaining_task_time"]          compute still needed to finish the job
    obs["remaining_time"]               time left until the deadline
    obs["slack"]                        remaining_time - remaining_task_time
                                        (spare time; negative-ish => must hurry)
    obs["restart_overhead"]             compute lost on each instance switch
    obs["remaining_restart_overhead"]   overhead still being paid off right now
    obs["has_spot"]                     True iff spot is AVAILABLE this tick
    obs["last_cluster_type"]            previous action (0/1/2)
    obs["gap_seconds"]                  length of one tick (e.g. 600)
    obs["elapsed_seconds"], obs["deadline"], obs["task_duration"]

You are BLIND to the future: you do NOT see how long spot will stay up/down next,
nor the exact prices. You CANNOT compute the optimal thresholds analytically;
expose them as tunable constants in `params`. A separate STRONG GUARANTEE will
force ON_DEMAND if the deadline would otherwise be missed, so you cannot miss the
deadline -- your only job is to MINIMIZE COST.

Rules:
  - `params` is a dict mapping each hyperparameter name to a float. Read tunable
    constants ONLY from `params`.
  - Use ONLY `obs`, `params`, `np` and `math`. No import statements, no I/O, no
    while-loops, no names starting with an underscore, no dunder attributes.
  - Always return an int in {0, 1, 2}."""

_CLAUSE_TUNED = """A separate zero-order tuner (CMA-ES) will choose the exact
VALUES of your constants WITHIN the [low, high] ranges you declare. You choose
only the STRUCTURE and the ranges; set each `guess` to a sensible default. The
feedback you receive for each structure is its total cost AFTER the tuner
optimized its constants (lower is better)."""

_CLAUSE_UNTUNED = """There is NO tuner. The constant VALUES you put in each
`guess` field are used AS-IS to run the policy -- you must choose both the
STRUCTURE AND good concrete constant values. The feedback you receive for each
structure is the total cost of your proposed policy evaluated at exactly those
values (lower is better)."""

_TAIL = """Propose a structure DISTINCT from any already tried.

Manifest 'scale': "log" for multiplicative gains, "cont" for additive or bounded
coefficients (time buffers in seconds, fractions/pressures in [0,1])."""

JSON_INSTRUCTION = """Return ONLY a JSON object with keys:
  "name":      short structure name (string),
  "rationale": <=2 sentences (string),
  "code":      the full `def decide(obs, params)` source (string),
  "manifest":  array of objects {"param","low","high","guess","scale"}.
The "param" names in the manifest MUST exactly match the keys read from `params`."""


def system_for(mode):
    clause = _CLAUSE_TUNED if mode == "tuned" else _CLAUSE_UNTUNED
    return f"{_HEAD}\n\n{clause}\n\n{_TAIL}"


def build_user_prompt(entries, mode):
    if not entries:
        hist = "No structures tried yet. Propose your first scheduling-policy structure."
    else:
        if mode == "tuned":
            desc = ("their CMA-ES *tuned* total cost (lower is better; the cost "
                    "AFTER a separate optimizer chose the best constants)")
            key = "tuned"
        else:
            desc = ("the total cost of the policy at the DEFAULT constant values "
                    "you guessed (lower is better; no tuning was applied)")
            key = "untuned"
        lines = [f"  - {e['name']}: cost = {fmt(e[key])}" for e in entries]
        hist = ("Structures already tried, with " + desc + ":\n" + "\n".join(lines)
                + "\n\nPropose a new, structurally DIFFERENT scheduling policy that "
                "could achieve a LOWER cost.")
    return hist + "\n\n" + JSON_INSTRUCTION


# --------------------------------------------------------------------------- #
# LLM call (constrained decoding) -- identical dispatch to the other experiments
# --------------------------------------------------------------------------- #
def _proposal_schema():
    from pydantic import BaseModel

    class ManifestItem(BaseModel):
        param: str
        low: float
        high: float
        guess: float
        scale: str

    class Proposal(BaseModel):
        name: str
        rationale: str
        code: str
        manifest: list[ManifestItem]

    return Proposal


def propose(client, entries, mode, idx):
    if PROVIDER == "gemini":
        return _propose_gemini(client, entries, mode, idx)
    return _propose_openai(client, entries, mode, idx)


def _propose_gemini(client, entries, mode, idx):
    from google.genai import types
    resp = client.models.generate_content(
        model=MODEL,
        contents=build_user_prompt(entries, mode),
        config=types.GenerateContentConfig(
            system_instruction=system_for(mode),
            temperature=0.95,
            seed=idx,
            response_mime_type="application/json",
            response_schema=_proposal_schema(),
            max_output_tokens=MAX_TOKENS,
        ),
    )
    if getattr(resp, "parsed", None) is not None:
        return resp.parsed.model_dump()
    return _extract_json(resp.text)


def _propose_openai(client, entries, mode, idx):
    messages = [
        {"role": "system", "content": system_for(mode)},
        {"role": "user", "content": build_user_prompt(entries, mode)},
    ]
    common = dict(
        model=MODEL,
        messages=messages,
        temperature=0.95,
        max_tokens=MAX_TOKENS,
        extra_headers={
            "HTTP-Referer": "https://github.com/anon590/hybrid-nested-search",
            "X-Title": "Hybrid Nested Search (Can't Be Late)",
        },
    )
    try:
        resp = client.chat.completions.create(
            response_format={"type": "json_object"}, seed=idx, **common)
    except Exception:
        resp = client.chat.completions.create(**common)
    content = (resp.choices[0].message.content or "").strip()
    if not content:
        raise ValueError("empty completion (output may have been truncated)")
    return _extract_json(content)


def _extract_json(text):
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
        text = re.sub(r"\n?```$", "", text)
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.S)
        if not m:
            raise
        return json.loads(m.group(0))


# --------------------------------------------------------------------------- #
# VALIDATE gate: AST sandbox + smoke test (Algorithm 1, line 6)
# --------------------------------------------------------------------------- #
_FORBIDDEN_NAMES = {
    "__import__", "eval", "exec", "open", "compile", "globals", "locals",
    "vars", "getattr", "setattr", "delattr", "input", "exit", "quit",
    "breakpoint", "help", "memoryview", "object",
}
_SAFE_BUILTINS = {
    n: getattr(builtins, n) for n in [
        "range", "len", "abs", "min", "max", "float", "int", "enumerate",
        "zip", "sum", "pow", "list", "tuple", "dict", "set", "frozenset",
        "bool", "round", "map", "filter", "sorted", "reversed", "any", "all",
    ]
}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


def validate_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    if "decide" not in {n.name for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)}:
        return False, "missing decide function"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed"
        if isinstance(node, ast.While):
            return False, "while-loops are not allowed"
        if isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return False, "dunder attribute access"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"forbidden name: {node.id}"
    return True, "ok"


# A tiny smoke benchmark (one trace) reused for the VALIDATE gate.
_SMOKE_BENCH = None


def _smoke_bench():
    global _SMOKE_BENCH
    if _SMOKE_BENCH is None:
        _SMOKE_BENCH = CantBeLateBenchmark("moderate", n_traces=1)
    return _SMOKE_BENCH


def compile_candidate(name, code, manifest):
    ok, msg = validate_code(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    ns = {"__builtins__": _SAFE_BUILTINS, "np": np, "numpy": np, "math": math}
    exec(compile(code, f"<llm:{name}>", "exec"), ns)
    fn = ns.get("decide")
    if not callable(fn):
        raise ValueError("decide is not callable")
    # Smoke test on one real trace: must run a full job without raising and stay
    # finite. (Out-of-range actions are coerced to NONE by the harness.)
    test_params = {k: v["guess"] for k, v in manifest.items()}
    bench = _smoke_bench()
    with np.errstate(all="ignore"):
        cost = bench._rollout_cost(fn, test_params, bench.traces[0])
    if not np.isfinite(cost):
        raise ValueError("smoke test produced a non-finite cost")
    return fn


def _first(d, keys, default=None):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_manifest(manifest):
    """Coerce an LLM-proposed manifest into {param: {bounds, guess, scale}}.
    Tolerant of formatting drift across providers (key aliases, list-or-dict)."""
    if isinstance(manifest, dict):
        items = [{"param": k, **(v if isinstance(v, dict) else {})}
                 for k, v in manifest.items()]
    else:
        items = list(manifest)

    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _first(item, ("param", "name", "parameter", "key"))
        if name is None:
            continue
        try:
            lo = float(_first(item, ("low", "min", "lower", "lo"), 0.0))
            hi = float(_first(item, ("high", "max", "upper", "hi"), 1.0))
            g = float(_first(item, ("guess", "default", "value", "init", "initial"),
                             (lo + hi) / 2.0))
        except (TypeError, ValueError):
            continue
        sc = str(_first(item, ("scale", "type"), "cont")).lower()
        if sc not in ("log", "cont"):
            sc = "log" if sc in ("logarithmic", "multiplicative") else "cont"
        if hi <= lo:
            hi = lo * 10.0 if lo > 0 else lo + 1.0
        if sc == "log" and lo <= 0:
            lo = min(1e-8, hi / 10.0)
        g = float(np.clip(g, lo, hi))
        out[str(name)] = {"bounds": (lo, hi), "guess": g, "scale": sc}
    if not out:
        raise ValueError("manifest has no usable parameters")
    return out


# --------------------------------------------------------------------------- #
# One autoresearch arm
# --------------------------------------------------------------------------- #
def run_arm(client, bench, mode, n, seed_base, label):
    """Run one LLM loop. `mode` selects the feedback signal ('tuned' or
    'untuned'). We always compute BOTH costs for analysis, but only the
    mode-appropriate one is shown to the LLM."""
    print(f"\n----- {label} arm  (feedback = {mode} cost) "
          + "-" * (40 - len(label)))
    entries = []
    for i in range(n):
        accepted = False
        for attempt in range(3):
            try:
                prop = propose(client, entries, mode, idx=seed_base + i * 10 + attempt)
                man = normalize_manifest(prop["manifest"])
                fn = compile_candidate(prop["name"], prop["code"], man)
                cand = {"fn": fn, "name": prop["name"], "manifest": man}
                untuned, _ = untuned_score(bench, cand)
                tuned, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
                entries.append(dict(
                    name=prop["name"], code=prop["code"], manifest=man,
                    rationale=prop.get("rationale", ""), untuned=untuned,
                    tuned=tuned, tuned_params=tparams))
                seen = untuned if mode == "untuned" else tuned
                print(f"[{label} {i+1}/{n}] {prop['name'][:32]:32s} "
                      f"LLM sees {mode} cost={fmt(seen):>10}  "
                      f"(untuned={fmt(untuned)}, tuned={fmt(tuned)})")
                accepted = True
                break
            except Exception as e:
                print(f"[{label} {i+1}/{n}] rejected ({attempt+1}/3): "
                      f"{type(e).__name__}: {str(e)[:90]}")
        if not accepted:
            print(f"[{label} {i+1}/{n}] giving up on this slot.")
    return entries


# --------------------------------------------------------------------------- #
# Reporting (cost, lower is better)
# --------------------------------------------------------------------------- #
def _arm_table(title, entries, rank_key):
    print(f"\n  {title}  (the LLM ranks by '{rank_key}', lower cost better)")
    print(f"    {'structure':32s} {'untuned cost':>13} {'tuned cost':>13}")
    print("    " + "-" * 62)
    best = min(entries, key=lambda x: x[rank_key])
    for e in sorted(entries, key=lambda e: e[rank_key]):
        mark = " *" if e is best else "  "
        print(f"  {mark}{e['name'][:32]:32s} {fmt(e['untuned']):>13} "
              f"{fmt(e['tuned']):>13}")


def _ratio(a, b):
    return a / b if b > 0 else float("inf")


def report_regime(name, hybrid, vanilla, pure_cost, pure_params, bench):
    print("\n" + "=" * 84)
    print(f"RESULTS for regime '{name}'  ({bench.desc})")
    print(f"  deadline={bench.deadline_h}h  overhead={bench.overhead}h  "
          f"n_traces={len(bench.traces)}  B_in={B_IN}")
    print("=" * 84)
    print(f"Pure CMA-ES (vanilla numerical): tunes a FIXED seed structure "
          f"({SEED_FIXED['name']})")
    print(f"  -> delivers cost = {fmt(pure_cost)}  at "
          f"{ {k: round(v, 2) for k, v in pure_params.items()} }  (no structural search)")

    if hybrid:
        _arm_table("HYBRID arm (tuned feedback)", hybrid, "tuned")
    if vanilla:
        _arm_table("VANILLA autoresearch arm (untuned feedback)", vanilla, "untuned")
    if not (hybrid and vanilla):
        print("\n(one LLM arm empty -- skipping head-to-head)")
        return None

    hyb = min(hybrid, key=lambda e: e["tuned"])      # hybrid ranks by tuned cost
    van = min(vanilla, key=lambda e: e["untuned"])   # vanilla ranks by untuned cost
    dh, dv = hyb["tuned"], van["untuned"]            # delivered cost

    print("\n  THREE-WAY HEAD-TO-HEAD (delivered cost, matched LLM budget)")
    print(f"    pure CMA-ES   {SEED_FIXED['name'][:34]:34s} -> cost = {fmt(pure_cost)}")
    print(f"    vanilla picks {van['name'][:34]:34s} -> cost = {fmt(dv)} (its guess)")
    print(f"    hybrid  picks {hyb['name'][:34]:34s} -> cost = {fmt(dh)} (tuned)")
    trio = {"hybrid": dh, "vanilla autoresearch": dv, "pure CMA-ES": pure_cost}
    winner = min(trio, key=trio.get)
    print(f"    => LOWEST delivered cost: {winner.upper()} (cost = {fmt(trio[winner])})")

    # De-aliasing isolation (H2): tune BOTH LLM picks, compare structure choice.
    print("  DE-ALIASING (tune both LLM picks; H2): "
          f"vanilla-pick-tuned cost={fmt(van['tuned'])}  "
          f"hybrid-pick-tuned cost={fmt(hyb['tuned'])}")
    if hyb["tuned"] < van["tuned"]:
        print("    => hybrid's RANKING chose a structurally better policy "
              "(de-aliasing, Prop. 1).")
    else:
        print("    => structure choices comparable once tuned; gain is the tuning itself.")

    # Regime: advantage ~ E[Delta]; Delta = untuned_cost - tuned_cost >= 0.
    finite = [e["untuned"] - e["tuned"] for e in (hybrid + vanilla)
              if np.isfinite(e["untuned"]) and np.isfinite(e["tuned"])]
    mean_delta = float(np.mean(finite)) if finite else float("nan")
    print(f"  REGIME: mean tuning gap E[Delta] = cost_untuned - cost_tuned = "
          f"{fmt(mean_delta)} (over {len(finite)} proposals)")

    return dict(regime=name, desc=bench.desc, pure_cost=pure_cost,
                van_pick=van["name"], hyb_pick=hyb["name"],
                van_delivered=dv, hyb_delivered=dh, pure_delivered=pure_cost,
                van_pick_tuned=van["tuned"], hyb_pick_tuned=hyb["tuned"],
                mean_delta=mean_delta, winner=winner)


def summary(rows):
    rows = [r for r in rows if r]
    if not rows:
        return
    print("\n" + "=" * 96)
    print("SUMMARY across regimes  (delivered cost; LOWER is better)")
    print("  vanilla = untuned LLM guess | pure = CMA-ES on fixed structure | "
          "hybrid = LLM structure + CMA-ES tuning")
    print("=" * 96)
    print(f"  {'regime':16s} {'E[Delta]':>10} {'vanilla':>10} {'pureCMA':>10} "
          f"{'hybrid':>10}  {'winner':>20}")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['regime']:16s} {fmt(r['mean_delta']):>10} "
              f"{fmt(r['van_delivered']):>10} {fmt(r['pure_delivered']):>10} "
              f"{fmt(r['hyb_delivered']):>10}  {r['winner']:>20}")
    print("  " + "-" * 92)
    wins = sum(1 for r in rows if r["winner"] == "hybrid")
    print(f"  hybrid delivers the lowest cost on {wins}/{len(rows)} regimes.")
    print("  (Eq. 4 prediction: hybrid advantage grows with E[Delta]; "
          "low-Delta regimes tie.)")
    print("=" * 96)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def make_client():
    if PROVIDER == "gemini":
        import google.genai as genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("No GEMINI_API_KEY / GOOGLE_API_KEY set (use --replay/--selftest).")
        return genai.Client(api_key=key)
    if PROVIDER == "openrouter":
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("No OPENROUTER_API_KEY set (use --replay/--selftest).")
        base_url = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
        return OpenAI(api_key=key, base_url=base_url)
    if PROVIDER == "yunwu":
        from openai import OpenAI
        key = os.environ.get("YUNWU_API_KEY")
        if not key:
            sys.exit("No YUNWU_API_KEY set (use --replay/--selftest).")
        base_url = os.environ.get("YUNWU_BASE_URL", "https://yunwu.ai/v1")
        return OpenAI(api_key=key, base_url=base_url)
    if PROVIDER == "openai":
        from openai import OpenAI
        key = os.environ.get("OPENAI_API_KEY")
        if not key:
            sys.exit("No OPENAI_API_KEY set (use --replay/--selftest).")
        base_url = os.environ.get("OPENAI_BASE_URL")
        return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    sys.exit(f"Unknown LLM_PROVIDER={PROVIDER!r} (use gemini | openrouter | openai).")


def live_run():
    client = make_client()
    print(f"Outer operator: {PROVIDER} '{MODEL}'  |  proposals/arm={N_PROPOSALS}  "
          f"|  inner CMA-ES B_in={B_IN}  |  regimes: {', '.join(REGIMES_RUN)}")
    print(f"  (logging to {LOG_PATH})")
    results = {}
    for name in REGIMES_RUN:
        bench = CantBeLateBenchmark(name)
        print(f"\n########## REGIME: {name} ({bench.desc}) "
              f"[ddl={bench.deadline_h}h, overhead={bench.overhead}h, "
              f"n_traces={len(bench.traces)}] ##########")
        pure_cost, pure_params, pure_ev = pure_cmaes(
            bench, SEED_FIXED, budget=PURE_BUDGET, seed=SEED)
        print(f"[pure CMA-ES] fixed structure tuned to cost={fmt(pure_cost)} "
              f"in {pure_ev} evals")
        hybrid = run_arm(client, bench, "tuned", N_PROPOSALS, seed_base=1000, label="HYB")
        vanilla = run_arm(client, bench, "untuned", N_PROPOSALS, seed_base=3000, label="VAN")
        results[name] = {"hybrid": hybrid, "vanilla": vanilla,
                         "pure_cost": pure_cost, "pure_params": pure_params}
        with open(LOG_PATH, "w") as f:
            json.dump(results, f, indent=2)
    print(f"\n(logged all regimes to {LOG_PATH})")
    return results


def replay_run():
    if not os.path.exists(LOG_PATH):
        sys.exit(f"--replay needs {LOG_PATH}; run live once first.")
    with open(LOG_PATH) as f:
        data = json.load(f)
    print(f"Replaying {LOG_PATH}: regimes {', '.join(data.keys())}")
    return data


def selftest_run():
    """No-API end-to-end: run the hand-written sketches as a stand-in for LLM
    proposals, exercising the full arm/report machinery offline."""
    print("SELF-TEST (no API): hand-written sketches stand in for LLM proposals.")
    results = {}
    for name in REGIMES_RUN:
        bench = CantBeLateBenchmark(name)
        print(f"\n########## REGIME: {name} ({bench.desc}) ##########")
        pure_cost, pure_params, _ = pure_cmaes(
            bench, SEED_FIXED, budget=B_IN, seed=SEED)
        entries = []
        for cand in SEEDS.values():
            man = cand["manifest"]
            untuned, _ = untuned_score(bench, cand)
            tuned, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
            entries.append(dict(name=cand["name"], code="<hand-written>",
                                manifest=man, rationale="", untuned=untuned,
                                tuned=tuned, tuned_params=tparams))
        results[name] = {"hybrid": entries, "vanilla": entries,
                         "pure_cost": pure_cost, "pure_params": pure_params}
    return results


def main():
    print("=" * 84)
    print("LLM-DRIVEN HYBRID NESTED SEARCH over 'Can't Be Late' scheduling regimes")
    print(f"  '{MODEL}' proposes scheduling-policy structures; CMA-ES tunes the")
    print("  manifests. Arms: hybrid vs vanilla autoresearch vs pure CMA-ES.")
    print("  Objective: total finishing cost in dollars (LOWER is better).")
    print("=" * 84)
    if "--selftest" in sys.argv:
        results = selftest_run()
    elif "--replay" in sys.argv:
        results = replay_run()
    else:
        results = live_run()

    rows = []
    for name in REGIMES_RUN:
        if name not in results:
            continue
        bench = CantBeLateBenchmark(name)
        r = results[name]
        rows.append(report_regime(name, r["hybrid"], r["vanilla"],
                                  r["pure_cost"], r["pure_params"], bench))
    summary(rows)


if __name__ == "__main__":
    main()
