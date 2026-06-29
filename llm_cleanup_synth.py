"""
LLM-driven Hybrid Nested Search on the (hidden-dynamics) Cleanup social dilemma.

This is the sequential-social-dilemma instantiation of llm_meta_optimizer.py,
realizing Testbed A of the method note (paper.pdf, Sec. 6.1-6.2). The artifact
under search is a TEAM POLICY whose STRUCTURE maps the global pollution level to
a number of cleaners; the LLM is the outer structural operator M of Algorithm 1
and CMA-ES is the inner Tune() over the structure's continuous holes.

Provider-agnostic outer operator (set LLM_PROVIDER), identical to the
meta-optimizer experiment:
  * gemini      -> google-genai SDK          (GEMINI_API_KEY / GOOGLE_API_KEY)
  * openrouter  -> any model via OpenAI API  (OPENROUTER_API_KEY) e.g. z-ai/glm-5.2
  * openai      -> OpenAI directly           (OPENAI_API_KEY)

THREE ARMS are compared (the user-requested set; paper Sec. 6.2):

  HYBRID (ours, Arm 3):   feedback = CMA-ES *tuned* welfare U_hat(tau).
      A tuner picks the constants; the LLM ranks structures by their parametric
      optimum, so it keeps high-ceiling structures (e.g. a waste-adaptive ladder)
      even if their textbook-default breakpoints are bad.

  VANILLA AUTORESEARCH (Arm 1): feedback = *untuned* welfare U at the LLM's
      guessed constants. No inner loop; the LLM ranks by its raw guess and is
      fooled by parametric aliasing (Prop. 1).

  PURE CMA-ES (vanilla numerical, Arm 2): CMA-ES over the parameters of a FIXED
      seed structure; no structural search. Capped at the seed's ceiling --
      isolates the value of structural discovery.

OBJECTIVE-BLINDNESS: the LLM is told it is a Cleanup public-goods gridworld and
gets the high-level mechanics, but the *dynamics constants* that set the optimal
cleaning effort (waste spawn rate, depletion threshold, apple respawn rate, team
size) are withheld, and reading them off the env is forbidden by the validator.
The only pollution observable is waste_ratio(env). This is what creates the
tuning gap Delta(tau) the hybrid arm de-aliases.

Run (Gemini, the original backend):
  uv run --isolated --with numpy --with cma --with google-genai \
      python3 llm_cleanup_synth.py

Run (OpenRouter, OpenAI API format -- e.g. z-ai/glm-5.2):
  LLM_PROVIDER=openrouter MODEL=z-ai/glm-5.2 \
  uv run --isolated --with numpy --with cma --with openai \
      python3 llm_cleanup_synth.py

  uv run --isolated --with numpy --with cma \
      python3 llm_cleanup_synth.py --replay     # reuse the log, no API
  uv run --isolated --with numpy --with cma \
      python3 llm_cleanup_synth.py --selftest    # no API: hand-written sketches
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
from cleanup_benchmark import (
    CleanupBenchmark, REGIMES, ORDER, pure_cmaes, SEED_FIXED_FRACTION, SEEDS,
)
from cleanup_helpers import policy_namespace

# --------------------------------------------------------------------------- #
# Provider / model selection (identical to llm_meta_optimizer.py).
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
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))

N_PROPOSALS = int(os.environ.get("N_PROPOSALS", "3"))
# regimes to run (default: ORDER); override with REGIMES_RUN="light,heavy"
REGIMES_RUN = [r.strip() for r in
               os.environ.get("REGIMES_RUN", ",".join(ORDER)).split(",")
               if r.strip()]
B_IN = int(os.environ.get("B_IN", "100"))     # inner CMA-ES budget per structure
PURE_BUDGET = int(os.environ.get("PURE_BUDGET", "120"))  # pure-CMA-ES budget
SEED = 1


def _default_log_path():
    if PROVIDER == "gemini" and MODEL.startswith("gemini-3.5"):
        return "llm_cleanup_log.json"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return f"llm_cleanup_log_{slug}.json"


LOG_PATH = os.environ.get("LOG_PATH") or _default_log_path()


# --------------------------------------------------------------------------- #
# Prompts -- the LLM knows the GAME but not the hidden dynamics constants; the
# only per-arm difference is who sets the constants (tuner vs the LLM itself).
# --------------------------------------------------------------------------- #
_HEAD = """You are an expert in multi-agent coordination proposing the STRUCTURE
(control flow + decision rule) of a TEAM POLICY for a public-goods gridworld --
a "Cleanup" sequential social dilemma with N cooperating agents.

The dilemma. A river slowly accumulates waste (pollution). Apples regrow in the
orchard ONLY while pollution is low enough; each apple collected is +1 shared
welfare. Cleaning a waste cell costs the cleaner a small reward but benefits the
whole team by letting apples regrow (a public good). Too few cleaners -> the
river saturates and apples stop -> welfare collapses. Too many cleaners -> nobody
harvests and cleaning costs pile up -> welfare also drops. The team objective is
the utilitarian efficiency U (collective reward per timestep; HIGHER is better).

You are BLIND to the exact dynamics. The waste spawn rate, the pollution level at
which apples stop regrowing, the apple regrowth rate, and the team size are all
HIDDEN, so you CANNOT compute the optimal number of cleaners analytically. You
must propose a STRUCTURE that maps the one pollution observable you are given to
a cleaning effort, exposing the cut-points / counts / gains as tunable constants.

Write exactly this function:
    def policy(env, agent_id, params):
        # return ONE integer action in [0, 8] for `agent_id` this timestep

You are given SKILLS so you only decide HOW MANY agents clean vs harvest -- you
do NOT write navigation:
    waste_ratio(env) -> float
        the global pollution fraction in [0, 1] (your ONLY pollution signal).
    assign_roles(env, n_cleaners) -> dict {agent_id: "clean" | "harvest"}
        deterministically assigns the n_cleaners agents nearest the pollution to
        clean and the rest to harvest (shared, identical for every agent).
    clean_action(env, agent_id) -> int
        the cleaner skill: navigate to and remove the nearest waste, return action.
    harvest_action(env, agent_id) -> int
        the harvester skill: navigate to and collect the nearest apple.
    env.n_agents          -> int   team size (read it; you may scale counts by it)
    env.agent_timeout[i]  -> int   >0 means agent i is frozen this step
    Also available: np, math.

Rules:
  - `params` is a dict mapping each hyperparameter name to a float. Read tunable
    constants ONLY from `params` (you MAY read env.n_agents, which is observable).
  - Use ONLY the skills above plus `np`/`math`. No import statements, no I/O, no
    while-loops, no names starting with an underscore, no dunder attributes.
  - Do NOT read environment dynamics constants (e.g. waste/apple spawn rates or
    thresholds); they are hidden on purpose.
  - Always handle the frozen case first: if int(env.agent_timeout[agent_id]) > 0:
    return 7  (STAND).

Canonical body to fill in (replace the STRUCTURE line):
    def policy(env, agent_id, params):
        if int(env.agent_timeout[agent_id]) > 0:
            return 7
        wr = waste_ratio(env)
        n_cleaners = ...            # YOUR STRUCTURE: map wr (and env.n_agents) to
                                    # a cleaner count, using constants from params
        roles = assign_roles(env, n_cleaners)
        if roles[agent_id] == "clean":
            return clean_action(env, agent_id)
        return harvest_action(env, agent_id)"""

_CLAUSE_TUNED = """A separate zero-order tuner (CMA-ES) will choose the exact
VALUES of your constants WITHIN the [low, high] ranges you declare. You choose
only the STRUCTURE and the ranges; set each `guess` to a sensible default. The
feedback you receive for each structure is its welfare U AFTER the tuner
optimized its constants."""

_CLAUSE_UNTUNED = """There is NO tuner. The constant VALUES you put in each
`guess` field are used AS-IS to run the policy -- you must choose both the
STRUCTURE AND good concrete constant values. The feedback you receive for each
structure is the welfare U of your proposed policy evaluated at exactly those
values."""

_TAIL = """Propose a structure DISTINCT from any already tried. Draw from
families such as: fixed-fraction roles (a constant share of the team cleans);
proportional controller (n_cleaners = gain * waste_ratio * n_agents); a
waste-adaptive threshold ladder (breakpoints in pollution -> per-band cleaner
counts); a piecewise-linear ramp with a deadzone and a saturation cap;
hysteresis-style two-threshold switching.

Cleaner counts may be fractional (they are rounded and clamped to [0, n_agents]).
Manifest 'scale': "log" for multiplicative gains, "cont" for additive or bounded
coefficients (thresholds in [0,1], counts, fractions)."""

JSON_INSTRUCTION = """Return ONLY a JSON object with keys:
  "name":      short structure name (string),
  "rationale": <=2 sentences (string),
  "code":      the full `def policy(env, agent_id, params)` source (string),
  "manifest":  array of objects {"param","low","high","guess","scale"}.
The "param" names in the manifest MUST exactly match the keys read from `params`."""


def system_for(mode):
    clause = _CLAUSE_TUNED if mode == "tuned" else _CLAUSE_UNTUNED
    return f"{_HEAD}\n\n{clause}\n\n{_TAIL}"


def build_user_prompt(entries, mode):
    if not entries:
        hist = "No structures tried yet. Propose your first team-policy structure."
    else:
        if mode == "tuned":
            desc = ("their CMA-ES *tuned* welfare U (higher is better; the score "
                    "AFTER a separate optimizer chose the best constants)")
            key = "tuned"
        else:
            desc = ("the welfare U of the policy at the DEFAULT constant values "
                    "you guessed (higher is better; no tuning was applied)")
            key = "untuned"
        lines = [f"  - {e['name']}: U = {fmt(e[key])}" for e in entries]
        hist = ("Structures already tried, with " + desc + ":\n" + "\n".join(lines)
                + "\n\nPropose a new, structurally DIFFERENT team policy that could "
                "achieve a HIGHER welfare U.")
    return hist + "\n\n" + JSON_INSTRUCTION


# --------------------------------------------------------------------------- #
# LLM call (constrained decoding) -- identical dispatch to the meta-optimizer
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
            "X-Title": "Hybrid Nested Search (Cleanup)",
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
# Attributes that would let a policy read the hidden objective and collapse the
# tuning gap; reading them is treated as a validation failure.
_FORBIDDEN_ATTRS = {
    "threshold_depletion", "threshold_restoration", "waste_spawn_prob",
    "apple_respawn_prob", "_current_apple_spawn_prob", "_compute_waste_density",
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
    if "policy" not in {n.name for n in ast.walk(tree)
                        if isinstance(n, ast.FunctionDef)}:
        return False, "missing policy function"
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, "imports are not allowed"
        if isinstance(node, ast.While):
            return False, "while-loops are not allowed"
        if isinstance(node, ast.Attribute):
            if node.attr.startswith("__"):
                return False, "dunder attribute access"
            if node.attr in _FORBIDDEN_ATTRS:
                return False, f"forbidden env attribute: {node.attr}"
        if isinstance(node, ast.Name) and node.id in _FORBIDDEN_NAMES:
            return False, f"forbidden name: {node.id}"
    return True, "ok"


def compile_candidate(name, code, manifest):
    ok, msg = validate_code(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    ns = {"__builtins__": _SAFE_BUILTINS}
    ns.update(policy_namespace())
    exec(compile(code, f"<llm:{name}>", "exec"), ns)
    fn = ns.get("policy")
    if not callable(fn):
        raise ValueError("policy is not callable")
    # Smoke test on a tiny live Cleanup env -- does NOT leak the regime's hidden
    # dynamics; must run a few steps without raising and return ints in [0, 8].
    from cleanup_env import make_cleanup
    env = make_cleanup(n_agents=3, small=True, max_steps=20, seed=0)
    env.reset(seed=0)
    test_params = {k: v["guess"] for k, v in manifest.items()}
    with np.errstate(all="ignore"):
        for _ in range(6):
            actions = {}
            for i in range(env.n_agents):
                a = fn(env, i, test_params)
                a = int(a)
                if a < 0 or a > 8:
                    raise ValueError(f"policy returned out-of-range action {a}")
                actions[i] = a
            env.step(actions)
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
    'untuned'). We always compute BOTH welfare scores for analysis, but only the
    mode-appropriate one is shown to the LLM."""
    print(f"\n----- {label} arm  (feedback = {mode} welfare U) "
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
                untuned_U = -untuned_score(bench, cand)[0]
                tuned_loss, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
                tuned_U = -tuned_loss
                entries.append(dict(
                    name=prop["name"], code=prop["code"], manifest=man,
                    rationale=prop.get("rationale", ""), untuned=untuned_U,
                    tuned=tuned_U, tuned_params=tparams))
                seen = untuned_U if mode == "untuned" else tuned_U
                print(f"[{label} {i+1}/{n}] {prop['name'][:32]:32s} "
                      f"LLM sees {mode} U={fmt(seen):>10}  "
                      f"(untuned={fmt(untuned_U)}, tuned={fmt(tuned_U)})")
                accepted = True
                break
            except Exception as e:
                print(f"[{label} {i+1}/{n}] rejected ({attempt+1}/3): "
                      f"{type(e).__name__}: {str(e)[:90]}")
        if not accepted:
            print(f"[{label} {i+1}/{n}] giving up on this slot.")
    return entries


# --------------------------------------------------------------------------- #
# Reporting (welfare U, higher is better)
# --------------------------------------------------------------------------- #
def _arm_table(title, entries, rank_key):
    print(f"\n  {title}  (the LLM ranks by '{rank_key}', higher U better)")
    print(f"    {'structure':30s} {'untuned U':>12} {'tuned U':>12}")
    print("    " + "-" * 58)
    best = max(entries, key=lambda x: x[rank_key])
    for e in sorted(entries, key=lambda e: -e[rank_key]):
        mark = " *" if e is best else "  "
        print(f"  {mark}{e['name'][:30]:30s} {fmt(e['untuned']):>12} "
              f"{fmt(e['tuned']):>12}")


def report_regime(name, hybrid, vanilla, pure_U, pure_params, bench):
    print("\n" + "=" * 84)
    print(f"RESULTS for regime '{name}'  ({bench.desc})")
    print(f"  n_agents={bench.n_agents}  steps={bench.max_steps}  seeds={bench.seeds}"
          f"  B_in={B_IN}")
    print("=" * 84)
    print(f"Pure CMA-ES (vanilla numerical): tunes a FIXED seed structure "
          f"({SEED_FIXED_FRACTION['name']})")
    print(f"  -> delivers U = {fmt(pure_U)}  at "
          f"{ {k: round(v, 4) for k, v in pure_params.items()} }  (no structural search)")

    if hybrid:
        _arm_table("HYBRID arm (tuned feedback)", hybrid, "tuned")
    if vanilla:
        _arm_table("VANILLA autoresearch arm (untuned feedback)", vanilla, "untuned")
    if not (hybrid and vanilla):
        print("\n(one LLM arm empty -- skipping head-to-head)")
        return None

    hyb = max(hybrid, key=lambda e: e["tuned"])      # hybrid ranks by tuned U
    van = max(vanilla, key=lambda e: e["untuned"])   # vanilla ranks by untuned U
    dh, dv = hyb["tuned"], van["untuned"]            # delivered welfare

    print("\n  THREE-WAY HEAD-TO-HEAD (delivered welfare U, matched LLM budget)")
    print(f"    pure CMA-ES   {SEED_FIXED_FRACTION['name'][:34]:34s} -> U = {fmt(pure_U)}")
    print(f"    vanilla picks {van['name'][:34]:34s} -> U = {fmt(dv)} (its guess)")
    print(f"    hybrid  picks {hyb['name'][:34]:34s} -> U = {fmt(dh)} (tuned)")
    trio = {"hybrid": dh, "vanilla autoresearch": dv, "pure CMA-ES": pure_U}
    winner = max(trio, key=trio.get)
    print(f"    => HIGHEST delivered welfare: {winner.upper()} (U = {fmt(trio[winner])})")

    # De-aliasing isolation (H2): tune BOTH LLM picks, compare structure choice.
    print("  DE-ALIASING (tune both LLM picks; H2): "
          f"vanilla-pick-tuned U={fmt(van['tuned'])}  "
          f"hybrid-pick-tuned U={fmt(hyb['tuned'])}")
    if hyb["tuned"] > van["tuned"]:
        print("    => hybrid's RANKING chose a structurally better policy "
              "(de-aliasing, Prop. 1).")
    else:
        print("    => structure choices comparable once tuned; gain is the tuning itself.")

    # Regime: advantage ~ E[Delta]
    finite = [e["tuned"] - e["untuned"] for e in (hybrid + vanilla)
              if np.isfinite(e["untuned"]) and np.isfinite(e["tuned"])]
    mean_delta = float(np.mean(finite)) if finite else float("nan")
    print(f"  REGIME: mean tuning gap E[Delta] = U_tuned - U_untuned = {fmt(mean_delta)} "
          f"(over {len(finite)} proposals)")

    return dict(regime=name, desc=bench.desc, pure_U=pure_U,
                van_pick=van["name"], hyb_pick=hyb["name"],
                van_delivered=dv, hyb_delivered=dh, pure_delivered=pure_U,
                van_pick_tuned=van["tuned"], hyb_pick_tuned=hyb["tuned"],
                mean_delta=mean_delta, winner=winner)


def summary(rows):
    rows = [r for r in rows if r]
    if not rows:
        return
    print("\n" + "=" * 96)
    print("SUMMARY across regimes  (delivered welfare U; higher is better)")
    print("  vanilla = untuned LLM guess | pure = CMA-ES on fixed structure | "
          "hybrid = LLM structure + CMA-ES tuning")
    print("=" * 96)
    print(f"  {'regime':10s} {'E[Delta]':>9} {'vanilla':>9} {'pureCMA':>9} "
          f"{'hybrid':>9}  {'winner':>20}")
    print("  " + "-" * 92)
    for r in rows:
        print(f"  {r['regime']:10s} {fmt(r['mean_delta']):>9} "
              f"{fmt(r['van_delivered']):>9} {fmt(r['pure_delivered']):>9} "
              f"{fmt(r['hyb_delivered']):>9}  {r['winner']:>20}")
    print("  " + "-" * 92)
    wins = sum(1 for r in rows if r["winner"] == "hybrid")
    print(f"  hybrid delivers the highest welfare on {wins}/{len(rows)} regimes.")
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
        bench = CleanupBenchmark(name)
        print(f"\n########## REGIME: {name} ({bench.desc}) "
              f"[n_agents={bench.n_agents}, steps={bench.max_steps}] ##########")
        pure_U, pure_params, pure_ev = pure_cmaes(
            bench, SEED_FIXED_FRACTION, budget=PURE_BUDGET, seed=SEED)
        print(f"[pure CMA-ES] fixed structure tuned to U={fmt(pure_U)} "
              f"in {pure_ev} evals")
        hybrid = run_arm(client, bench, "tuned", N_PROPOSALS, seed_base=1000, label="HYB")
        vanilla = run_arm(client, bench, "untuned", N_PROPOSALS, seed_base=3000, label="VAN")
        results[name] = {"hybrid": hybrid, "vanilla": vanilla,
                         "pure_U": pure_U, "pure_params": pure_params}
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
    """No-API end-to-end: run the three hand-written sketches as a stand-in for
    LLM proposals, so the full arm/report machinery is exercised offline."""
    print("SELF-TEST (no API): hand-written sketches stand in for LLM proposals.")
    results = {}
    for name in REGIMES_RUN:
        bench = CleanupBenchmark(name)
        print(f"\n########## REGIME: {name} ({bench.desc}) ##########")
        pure_U, pure_params, _ = pure_cmaes(
            bench, SEED_FIXED_FRACTION, budget=B_IN, seed=SEED)
        entries = []
        for cand in SEEDS.values():
            man = cand["manifest"]
            untuned_U = -untuned_score(bench, cand)[0]
            tuned_loss, tparams, _ = tune(bench, cand, budget=B_IN, seed=SEED)
            entries.append(dict(name=cand["name"], code="<hand-written>",
                                manifest=man, rationale="", untuned=untuned_U,
                                tuned=-tuned_loss, tuned_params=tparams))
        results[name] = {"hybrid": entries, "vanilla": entries,
                         "pure_U": pure_U, "pure_params": pure_params}
    return results


def main():
    print("=" * 84)
    print("LLM-DRIVEN HYBRID NESTED SEARCH over HIDDEN-DYNAMICS Cleanup regimes")
    print(f"  '{MODEL}' proposes TEAM-POLICY structures (waste -> cleaner count);")
    print("  CMA-ES tunes the manifests. Arms: hybrid vs vanilla autoresearch vs")
    print("  pure CMA-ES.  Objective: utilitarian efficiency U (higher better).")
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
        bench = CleanupBenchmark(name)
        r = results[name]
        rows.append(report_regime(name, r["hybrid"], r["vanilla"],
                                  r["pure_U"], r["pure_params"], bench))
    summary(rows)


if __name__ == "__main__":
    main()
