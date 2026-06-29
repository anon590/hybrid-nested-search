"""
LLM-driven Hybrid Nested Search on the (hidden) Rosenbrock meta-optimizer.

Replaces the two hand-written candidates of synth_meta_optimizer.py with
optimizer STRUCTURES proposed on-the-fly by an LLM. The LLM is the outer
structural operator M of Algorithm 1; CMA-ES is the inner Tune().

The outer operator is provider-agnostic (set LLM_PROVIDER):
  * gemini      -> google-genai SDK          (GEMINI_API_KEY / GOOGLE_API_KEY)
  * openrouter  -> any model via OpenAI API  (OPENROUTER_API_KEY) e.g. z-ai/glm-5.2
  * openai      -> OpenAI directly           (OPENAI_API_KEY)

We run TWO independent Gemini autoresearch loops (the paper's arms), identical
except for the FEEDBACK SIGNAL the LLM sees when choosing what to propose next:

  HYBRID arm  (Arm 3, ours):  feedback = CMA-ES *tuned* loss F_hat(tau).
      A tuner picks the constants; the LLM ranks structures by their parametric
      optimum, so it keeps high-ceiling structures even if their defaults are bad.

  VANILLA arm (Arm 1/4):      feedback = *untuned* loss g_van(tau)=f(tau, guess).
      No tuner in the loop; the constants the LLM guesses are used as-is, and it
      ranks structures by that untuned score -> it is fooled by parametric
      aliasing and steers away from structures that diverge / look bad at their
      textbook defaults.

CRITICAL -- the objective (Rosenbrock) is NEVER revealed to either LLM loop:
  * prompts describe only a generic, possibly ill-conditioned smooth 2-D
    function and a black-box gradient oracle grad_fn(x,y)->(dx,dy);
  * the start point, gradient formula, and the name "Rosenbrock" are withheld;
  * the only feedback is a scalar loss (tuned or untuned) -- never the function.

The comparison reproduces Prop. 1 / H1-H2 with a real LLM as the outer operator:
the hybrid arm reaches a far lower delivered loss, and even with identical
tuning applied to both arms' picks, the hybrid arm's *ranking* selects a better
structure -- the gain is de-aliasing the ranking, not merely tuning.

Run (Gemini, the original backend):
  uv run --isolated --with numpy --with cma --with google-genai \
      python3 llm_meta_optimizer.py             # live: calls Gemini (both arms)

Run (OpenRouter, OpenAI API format -- e.g. z-ai/glm-5.2):
  LLM_PROVIDER=openrouter MODEL=z-ai/glm-5.2 \
  uv run --isolated --with numpy --with cma --with openai \
      python3 llm_meta_optimizer.py             # live: calls OpenRouter (both arms)

  uv run --isolated --with numpy --with cma \
      python3 llm_meta_optimizer.py --replay    # reuse the log, no API
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
from benchmarks import Benchmark, PROBLEMS, ORDER, direct_cmaes

# --------------------------------------------------------------------------- #
# Provider / model selection.  LLM_PROVIDER in {gemini, openrouter, openai};
# override the model with MODEL=... (e.g. MODEL=z-ai/glm-5.2).
# --------------------------------------------------------------------------- #
PROVIDER = os.environ.get("LLM_PROVIDER", "gemini").lower()
_DEFAULT_MODEL = {
    "gemini": "gemini-3.5-flash",
    "openrouter": "z-ai/glm-5.2",
    "openai": "gpt-4o-mini",
}
MODEL = (os.environ.get("MODEL")
         or os.environ.get("GEMINI_MODEL")            # back-compat
         or _DEFAULT_MODEL.get(PROVIDER, "gemini-3.5-flash"))
# generous cap: reasoning models (e.g. GLM) spend tokens before the JSON answer
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "16384"))

N_PROPOSALS = int(os.environ.get("N_PROPOSALS", "3"))
# functions to run (default: all); override with FUNCS="rosenbrock,ackley"
FUNCS = [f.strip() for f in os.environ.get("FUNCS", ",".join(ORDER)).split(",") if f.strip()]
B_IN = 100
SEED = 1


def _default_log_path():
    """Per-model log path so different backends don't clobber each other's runs.
    The original Gemini run keeps its historical filename."""
    if PROVIDER == "gemini" and MODEL.startswith("gemini-3.5"):
        return "llm_multifn_log.json"
    slug = re.sub(r"[^A-Za-z0-9.]+", "-", MODEL).strip("-")
    return f"llm_multifn_log_{slug}.json"


LOG_PATH = os.environ.get("LOG_PATH") or _default_log_path()


# --------------------------------------------------------------------------- #
# Prompts -- objective-blind; the only per-arm difference is who sets constants
# --------------------------------------------------------------------------- #
_HEAD = """You are an expert in numerical optimization proposing the STRUCTURE
(control flow + update rule) of a first-order iterative optimizer for an UNKNOWN
2-D objective. The objective may be badly ill-conditioned, multimodal (many
local minima), deceptive, or have large flat regions -- you do not know which.

You are BLIND to the objective. You only have a black-box gradient oracle
grad_fn(x, y) -> (dx, dy). Do not assume any particular function.

Write exactly this function:
    def custom_optimizer(grad_fn, x, y, steps, params):
        # run `steps` iterations starting from (x, y); return the final (x, y)
Rules:
  - `params` is a dict mapping each hyperparameter name to a float. Read
    constants ONLY from `params`.
  - Use ONLY `math` and numpy as `np`. No import statements, no I/O, no
    while-loops, no names starting with an underscore. Iterate with
    `for _ in range(steps):`."""

_CLAUSE_TUNED = """A separate zero-order tuner (CMA-ES) will choose the exact
VALUES of your constants WITHIN the [low, high] ranges you declare. You choose
only the STRUCTURE and the ranges; set each `guess` to a sensible default. The
feedback you receive for each structure is its loss AFTER the tuner optimized
its constants."""

_CLAUSE_UNTUNED = """There is NO tuner. The constant VALUES you put in each
`guess` field are used AS-IS to run the optimizer -- you must choose both the
structure AND good concrete constant values. The feedback you receive for each
structure is the loss of your proposed artifact evaluated at exactly those
values."""

_TAIL = """Propose a structure DISTINCT from any already tried. Draw from
families such as: gradient descent, heavy-ball momentum, Nesterov accelerated
gradient, RMSProp, Adagrad, Adam, gradient clipping, decaying/warmup
learning-rate schedules, sign-SGD.

Manifest 'scale': "log" for multiplicative gains (learning rates, epsilons),
"cont" for additive or bounded coefficients (momentum/decay in [0,1))."""

JSON_INSTRUCTION = """Return ONLY a JSON object with keys:
  "name":      short structure name (string),
  "rationale": <=2 sentences (string),
  "code":      the full `def custom_optimizer(...)` source (string),
  "manifest":  array of objects {"param","low","high","guess","scale"}.
The "param" names in the manifest MUST exactly match the keys read from `params`."""


def system_for(mode):
    clause = _CLAUSE_TUNED if mode == "tuned" else _CLAUSE_UNTUNED
    return f"{_HEAD}\n\n{clause}\n\n{_TAIL}"


def build_user_prompt(entries, mode):
    if not entries:
        hist = "No structures tried yet. Propose your first optimizer structure."
    else:
        if mode == "tuned":
            desc = ("their CMA-ES *tuned* loss (lower is better; the score AFTER a "
                    "separate optimizer chose the best constants)")
            key = "tuned"
        else:
            desc = ("the loss of the artifact at the DEFAULT constant values you "
                    "guessed (lower is better; no tuning was applied)")
            key = "untuned"
        lines = [f"  - {e['name']}: loss = {fmt(e[key])}" for e in entries]
        hist = ("Structures already tried, with " + desc + ":\n" + "\n".join(lines)
                + "\n\nPropose a new, structurally DIFFERENT optimizer that could "
                "achieve a lower loss.")
    return hist + "\n\n" + JSON_INSTRUCTION


# --------------------------------------------------------------------------- #
# Gemini call (constrained decoding)
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
    """Dispatch to the configured backend; returns a proposal dict."""
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
    """OpenAI chat-completions format -- used for OpenRouter and OpenAI.

    Requests a JSON object; not every OpenRouter model accepts response_format /
    seed, so we retry once without them and rely on the prompt + _extract_json."""
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
            "X-Title": "Hybrid Nested Search",
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
        "zip", "sum", "pow", "list", "tuple", "dict", "bool", "round", "map",
        "filter", "sorted", "reversed",
    ]
}
_SAFE_BUILTINS.update({"True": True, "False": False, "None": None})


def validate_code(code):
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return False, f"syntax error: {e}"
    if "custom_optimizer" not in {n.name for n in ast.walk(tree)
                                  if isinstance(n, ast.FunctionDef)}:
        return False, "missing custom_optimizer"
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


def compile_candidate(name, code, manifest):
    ok, msg = validate_code(code)
    if not ok:
        raise ValueError(f"VALIDATE failed: {msg}")
    ns = {"__builtins__": _SAFE_BUILTINS, "math": math, "np": np, "numpy": np}
    exec(compile(code, f"<llm:{name}>", "exec"), ns)
    fn = ns.get("custom_optimizer")
    if not callable(fn):
        raise ValueError("custom_optimizer is not callable")
    # Smoke test on a GENERIC quadratic (min at (3,-2)) -- does NOT leak the real
    # objective; must run without raising and return two numbers.
    test_grad = lambda x, y: (2.0 * (x - 3.0), 2.0 * (y + 2.0))
    test_params = {k: v["guess"] for k, v in manifest.items()}
    with np.errstate(all="ignore"):
        out = fn(test_grad, 0.0, 0.0, 30, test_params)
    if not (isinstance(out, (tuple, list)) and len(out) == 2):
        raise ValueError("custom_optimizer must return (x, y)")
    float(out[0]); float(out[1])
    return fn


def _first(d, keys, default=None):
    """First present, non-null value among `keys` in dict `d`."""
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return default


def normalize_manifest(manifest):
    """Coerce an LLM-proposed manifest into {param: {bounds, guess, scale}}.

    Tolerant of formatting drift across providers (the Gemini path pins a pydantic
    schema, but OpenAI/OpenRouter JSON mode does not): accepts either a list of
    items or a {param: spec} dict, and common key aliases (param/name, low/min,
    high/max, guess/default, scale/type)."""
    if isinstance(manifest, dict):                       # {"lr": {"low":..}, ...}
        items = [{"param": k, **(v if isinstance(v, dict) else {})}
                 for k, v in manifest.items()]
    else:
        items = list(manifest)

    out = {}
    for item in items:
        if not isinstance(item, dict):
            continue
        name = _first(item, ("param", "name", "parameter", "key"))
        if name is None:                                 # nameless hole -> unusable
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
    """Run one Gemini loop. `mode` selects the feedback signal ('tuned' or
    'untuned'). We always compute BOTH scores for analysis, but only the
    mode-appropriate one is shown to the LLM -- so the proposal trajectory is
    driven purely by that signal (faithful to the arm)."""
    print(f"\n----- {label} arm  (feedback = {mode} loss) "
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
                print(f"[{label} {i+1}/{n}] {prop['name']:32s} "
                      f"LLM sees {mode}={fmt(seen):>12}  "
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
# Reporting
# --------------------------------------------------------------------------- #
def _arm_table(title, entries, rank_key):
    print(f"\n  {title}  (the LLM ranks by '{rank_key}')")
    print(f"    {'structure':30s} {'untuned guess':>14} {'tuned (F_hat)':>14}")
    print("    " + "-" * 60)
    for e in sorted(entries, key=lambda e: e[rank_key]):
        mark = " *" if e is min(entries, key=lambda x: x[rank_key]) else "  "
        print(f"  {mark}{e['name'][:30]:30s} {fmt(e['untuned']):>14} "
              f"{fmt(e['tuned']):>14}")


def _ratio(a, b):
    return a / b if b > 0 else float("inf")


def report_function(name, hybrid, vanilla, bench):
    """Per-function report. Returns a summary-row dict for the final table."""
    print("\n" + "=" * 80)
    print(f"RESULTS for '{name}'  ({bench.desc})")
    print("=" * 80)
    ref, ref_ev = direct_cmaes(bench)
    print(f"Reference: direct CMA-ES on f reaches {fmt(ref)} in {ref_ev} evals "
          "(needs the objective itself; not a synthesis method).")

    if hybrid:
        _arm_table("HYBRID arm (tuned feedback)", hybrid, "tuned")
    if vanilla:
        _arm_table("VANILLA arm (untuned feedback)", vanilla, "untuned")
    if not (hybrid and vanilla):
        print("\n(one arm empty -- skipping comparison)")
        return None

    hyb = min(hybrid, key=lambda e: e["tuned"])     # hybrid ranks by tuned
    van = min(vanilla, key=lambda e: e["untuned"])  # vanilla ranks by untuned
    dh, dv = hyb["tuned"], van["untuned"]           # delivered losses

    print("\n  HEAD-TO-HEAD (delivered artifact, matched LLM budget)")
    print(f"    vanilla picks {van['name'][:34]:34s} -> delivers {fmt(dv)} (its guess)")
    print(f"    hybrid  picks {hyb['name'][:34]:34s} -> delivers {fmt(dh)} (tuned)")
    if dh < dv:
        print(f"    => HYBRID WINS (H1): {_ratio(dv, dh):.2e}x lower delivered loss.")
        verdict = "hybrid"
    elif dv < dh:
        print(f"    => vanilla lower this run by {_ratio(dh, dv):.2e}x (see regime).")
        verdict = "vanilla"
    else:
        print("    => tie on delivered loss.")
        verdict = "tie"

    # H2: de-aliasing isolation -- tune BOTH picks, compare the structure choice
    print("  DE-ALIASING (tune both picks; H2): "
          f"vanilla-pick-tuned={fmt(van['tuned'])}  hybrid-pick-tuned={fmt(hyb['tuned'])}")
    if hyb["tuned"] < van["tuned"]:
        print("    => hybrid's RANKING chose a structurally better optimizer "
              "(de-aliasing, Prop. 1).")
    else:
        print("    => structure choices comparable once tuned; gain is the tuning itself.")

    # regime: advantage ~ E[Delta]
    finite = [e["untuned"] - e["tuned"] for e in (hybrid + vanilla)
              if np.isfinite(e["untuned"])]
    mean_delta = float(np.mean(finite)) if finite else float("inf")
    n_div = sum(1 for e in (hybrid + vanilla) if not np.isfinite(e["untuned"]))
    print(f"  REGIME: mean finite tuning gap E[Delta]={fmt(mean_delta)}; "
          f"{n_div}/{len(hybrid+vanilla)} proposals diverged at their defaults.")

    return dict(function=name, desc=bench.desc, ref=ref,
                van_pick=van["name"], hyb_pick=hyb["name"],
                van_delivered=dv, hyb_delivered=dh,
                van_pick_tuned=van["tuned"], hyb_pick_tuned=hyb["tuned"],
                mean_delta=mean_delta, n_diverged=n_div, verdict=verdict)


def summary(rows):
    rows = [r for r in rows if r]
    if not rows:
        return
    print("\n" + "=" * 92)
    print("SUMMARY across functions  (delivered loss: vanilla=untuned guess, "
          "hybrid=CMA-ES tuned)")
    print("=" * 92)
    print(f"  {'function':11s} {'regime':24s} {'E[Δ]':>10} {'vanilla':>11} "
          f"{'hybrid':>11} {'advantage':>11}")
    print("  " + "-" * 88)
    for r in rows:
        adv = _ratio(r["van_delivered"], r["hyb_delivered"])
        adv_s = "tie" if r["verdict"] == "tie" else (
            f"{adv:.1e}x" if r["verdict"] == "hybrid" else
            f"van {_ratio(r['hyb_delivered'], r['van_delivered']):.1e}x")
        print(f"  {r['function']:11s} {r['desc'][:24]:24s} {fmt(r['mean_delta']):>10} "
              f"{fmt(r['van_delivered']):>11} {fmt(r['hyb_delivered']):>11} {adv_s:>11}")
    print("  " + "-" * 88)
    wins = sum(1 for r in rows if r["verdict"] == "hybrid")
    print(f"  hybrid delivers lower loss on {wins}/{len(rows)} functions.")
    print("=" * 92)


# --------------------------------------------------------------------------- #
# Drivers
# --------------------------------------------------------------------------- #
def make_client():
    """Build the provider-appropriate client (env-driven)."""
    if PROVIDER == "gemini":
        import google.genai as genai
        key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            sys.exit("No GEMINI_API_KEY / GOOGLE_API_KEY set (use --replay instead).")
        return genai.Client(api_key=key)
    if PROVIDER == "openrouter":
        from openai import OpenAI
        key = os.environ.get("OPENROUTER_API_KEY")
        if not key:
            sys.exit("No OPENROUTER_API_KEY set (use --replay instead).")
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
            sys.exit("No OPENAI_API_KEY set (use --replay instead).")
        base_url = os.environ.get("OPENAI_BASE_URL")
        return OpenAI(api_key=key, base_url=base_url) if base_url else OpenAI(api_key=key)
    sys.exit(f"Unknown LLM_PROVIDER={PROVIDER!r} (use gemini | openrouter | openai).")


def live_run():
    client = make_client()
    print(f"Outer operator: {PROVIDER} '{MODEL}'  |  proposals/arm={N_PROPOSALS}  "
          f"|  inner CMA-ES budget B_in={B_IN}  |  functions: {', '.join(FUNCS)}")
    print(f"  (logging to {LOG_PATH})")
    results = {}
    for name in FUNCS:
        bench = Benchmark(name, steps=1000)
        print(f"\n########## FUNCTION: {name} ({bench.desc}) "
              f"[starts={len(bench.starts)}] ##########")
        hybrid = run_arm(client, bench, "tuned", N_PROPOSALS, seed_base=1000, label="HYB")
        vanilla = run_arm(client, bench, "untuned", N_PROPOSALS, seed_base=3000, label="VAN")
        results[name] = {"hybrid": hybrid, "vanilla": vanilla}
        with open(LOG_PATH, "w") as f:       # checkpoint after each function
            json.dump(results, f, indent=2)
    print(f"\n(logged all functions to {LOG_PATH})")
    return results


def replay_run():
    if not os.path.exists(LOG_PATH):
        sys.exit(f"--replay needs {LOG_PATH}; run live once first.")
    with open(LOG_PATH) as f:
        data = json.load(f)
    print(f"Replaying {LOG_PATH}: functions {', '.join(data.keys())}")
    return data


def main():
    print("=" * 80)
    print("LLM-DRIVEN HYBRID NESTED SEARCH over a suite of HIDDEN objectives")
    print(f"  '{MODEL}' proposes optimizer structures from a black-box gradient")
    print("  only; it never sees any objective. CMA-ES tunes the manifests.")
    print("=" * 80)
    results = replay_run() if "--replay" in sys.argv else live_run()
    rows = []
    for name in FUNCS:
        if name not in results:
            continue
        bench = Benchmark(name, steps=1000)
        rows.append(report_function(name, results[name]["hybrid"],
                                    results[name]["vanilla"], bench))
    summary(rows)


if __name__ == "__main__":
    main()
