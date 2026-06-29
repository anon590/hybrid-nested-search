# A Hybrid Nested Harness for Decoupling Structure and Parameters in LLM-Driven Optimization

Code for the paper *"A Hybrid Nested Harness for Decoupling Structure and Parameters
in LLM-Driven Optimization"* (under double-blind review).

In LLM-driven evolutionary search, the model acts as a single operator that
simultaneously edits **structure** (control flow, update rules) and **continuous
parameters** (learning rates, thresholds). LLMs are good at the former but
inefficient at the latter. We formalize a **hybrid nested search**: an outer loop
in which the LLM proposes a structural *sketch* with numeric holes, and an inner
numerical optimizer tunes those holes. Evaluating each structure on its *tuned*
value (rather than the LLM's raw guess) removes the **parametric aliasing** that
makes joint search discard good structures whose default constants look bad.

This repository is organized by the experiment families in the paper:

1. **Meta-optimizers** — LLM-written optimization algorithms on hidden 2-D objectives (below).
2. **Executable policies** — code-based policies for cloud-systems benchmarks and a
   social-dilemma gridworld (below).
3. **Approximate Bayesian inference** — *to follow.*

## Meta-optimizers

The LLM writes the *optimization algorithm* (the structure: momentum, adaptive
gradients, restarts, schedules, ...) for a **hidden** 2-D objective, while CMA-ES
tunes that algorithm's hyperparameters. The LLM is **blind to the objective**: it
sees only a black-box gradient oracle `grad_fn(x, y) -> (dx, dy)` and a starting
point — never the function, its formula, or its name.

Two arms are run under an identical budget; they differ *only* in the feedback
signal the LLM sees when choosing what to propose next:

- **Hybrid (ours):** feedback is the CMA-ES-*tuned* loss `F_hat(tau)`. The LLM
  ranks structures by their parametric optimum, so it keeps high-ceiling
  structures even when their textbook defaults diverge.
- **Vanilla:** feedback is the *untuned* loss `f(tau, guess)`. The LLM ranks
  structures by the score of its own guessed constants, so parametric aliasing
  steers it away from structures that look bad at their defaults.

### Files

| File | Role |
|------|------|
| `llm_meta_optimizer.py` | Main experiment. A real LLM is the outer structural operator; runs the hybrid and vanilla arms over a suite of hidden objectives. |
| `synth_meta_optimizer.py` | Self-contained synthetic version (no LLM): two fixed candidate structures (GD vs. heavy-ball momentum) on Rosenbrock, isolating the de-aliasing claim. Also provides the inner `tune()` (CMA-ES), `untuned_score()`, and formatting helpers used by the main experiment. |
| `benchmarks.py` | The five hidden test functions (`rosenbrock`, `ellipsoid`, `rastrigin`, `ackley`, `schwefel`) with analytic gradient oracles, the objective-blind `Benchmark` harness, and a `direct_cmaes` reference. |

### Quick start (no API key)

The synthetic experiment reproduces the core de-aliasing result with no LLM and no
API key — CMA-ES plays every role on two fixed structures:

```bash
uv run --with numpy --with cma python3 synth_meta_optimizer.py
```

It shows that the *untuned* guess for the better structure (heavy-ball momentum)
diverges, so vanilla joint search ranks it below plain GD (aliasing); the inner
CMA-ES de-aliases the ranking and recovers momentum as the superior structure.

### Running the full LLM experiment

The outer operator is provider-agnostic; select it with `LLM_PROVIDER` and the
model with `MODEL`. An API key for the chosen provider is required.

```bash
# Gemini (default provider)
GEMINI_API_KEY=...  uv run --isolated --with numpy --with cma --with google-genai \
    python3 llm_meta_optimizer.py

# Any model via OpenRouter (OpenAI API format), e.g. GLM
LLM_PROVIDER=openrouter MODEL=z-ai/glm-5.2 OPENROUTER_API_KEY=... \
    uv run --isolated --with numpy --with cma --with openai \
    python3 llm_meta_optimizer.py

# OpenAI directly
LLM_PROVIDER=openai MODEL=gpt-4o-mini OPENAI_API_KEY=... \
    uv run --isolated --with numpy --with cma --with openai \
    python3 llm_meta_optimizer.py
```

Each live run writes a per-model proposal log (`llm_multifn_log*.json`). Passing
`--replay` re-scores an existing log without any API calls:

```bash
uv run --isolated --with numpy --with cma python3 llm_meta_optimizer.py --replay
```

> Proposal logs and figures are **not** included in this repository; `--replay`
> reuses a log produced by a prior live run.

### Useful environment variables

| Variable | Default | Meaning |
|----------|---------|---------|
| `LLM_PROVIDER` | `gemini` | `gemini`, `openrouter`, or `openai`. |
| `MODEL` | per-provider | Outer-operator model (e.g. `z-ai/glm-5.2`). |
| `N_PROPOSALS` | `3` | Structures proposed per arm. |
| `FUNCS` | all five | Comma-separated subset, e.g. `FUNCS=rosenbrock,ackley`. |
| `MAX_TOKENS` | `16384` | Generation cap for the proposer. |
| `LOG_PATH` | per-model | Override the proposal log location. |

### Safety of executed code

LLM-proposed optimizers are run through an AST sandbox before execution
(`validate_code` / `compile_candidate` in `llm_meta_optimizer.py`): no imports, no
`while`-loops, no dunder access, a restricted builtins set, and a smoke test on a
generic quadratic that does not leak the real objective.

## Executable policies

Here the artifact is a *code-based policy* evaluated by a simulator, so the
objective is no longer closed-form. The LLM proposes the policy structure (control
flow mapping observations to actions or roles) with numeric holes; CMA-ES tunes the
holes (buffers, thresholds, weights). We compare the three strategies from the
paper — **vanilla** (LLM joint search), **numerical-only** (CMA-ES on a fixed
hand-designed structure), and **hybrid** — and additionally re-run every task with
**GEPA** (reflective prompt evolution) as the outer operator in place of the (1+1)
loop, showing the inner tuner is orthogonal to the choice of structural optimizer.

Three tasks:

- **Can't Be Late** — a spot/on-demand cloud scheduler; the policy maps state
  (remaining work, slack to deadline, restart overhead) to an action, minimizing
  dollar cost under a hard deadline. Regimes vary the restart overhead.
- **Cloudcast** — a multi-cloud broadcast router; the policy proposes the routing
  topology (relay trees, hub routing, *k*-shortest multipath) and CMA-ES tunes its
  weights and tolerances. Regimes: intra- vs. inter-cloud egress cost.
- **Cleanup** — a team policy for the Cleanup public-goods gridworld (a sequential
  social dilemma): map environment state (e.g. pollution level) to each agent's
  role, maximizing utilitarian welfare under self-play. Seeds vary the difficulty.

### Files

| File | Task | Role |
|------|------|------|
| `cantbelate_benchmark.py` | Can't Be Late | Simulator harness (regimes, traces, scoring). |
| `llm_cantbelate_synth.py` | Can't Be Late | LLM outer operator: hybrid vs. vanilla arms. |
| `gepa_hybrid_cantbelate.py` | Can't Be Late | GEPA as the outer operator (orthogonality test). |
| `cloudcast_benchmark.py` | Cloudcast | Simulator harness over the multi-cloud graph. |
| `llm_cloudcast_synth.py` | Cloudcast | LLM outer operator: hybrid vs. vanilla arms. |
| `gepa_hybrid_cloudcast.py` | Cloudcast | GEPA as the outer operator. |
| `cleanup_benchmark.py` | Cleanup | Self-play welfare harness over the gridworld. |
| `llm_cleanup_synth.py` | Cleanup | LLM outer operator: hybrid vs. vanilla arms. |
| `gepa_hybrid_cleanup.py` | Cleanup | GEPA as the outer operator. |
| `cleanup_env.py`, `gathering_env.py`, `cleanup_helpers.py` | Cleanup | Gridworld environment and policy helpers. |

All of these reuse the inner `tune()` / `untuned_score()` from `synth_meta_optimizer.py`.

### External dependencies (Can't Be Late + Cloudcast)

The two cloud benchmarks call simulators that ship with the **GEPA** repository's
ADRS examples (`examples/adrs/can_be_late`, `examples/adrs/cloudcast`). Point
`GEPA_ROOT` at a local checkout (it defaults to `~/gepa`):

```bash
git clone https://github.com/gepa-ai/gepa ~/gepa
export GEPA_ROOT=~/gepa
```

`cantbelate_benchmark.py` additionally expects the spot-availability traces shipped
under that checkout; Cloudcast additionally needs `networkx` and `pandas`. The
Cleanup task is self-contained (no `GEPA_ROOT`), with optional `Pillow` only for
rendering frames.

### Running

Each `llm_*_synth.py` mirrors the meta-optimizer driver: a live run calls the
configured provider, `--selftest` validates the wiring with no API, and `--replay`
re-scores an existing proposal log.

```bash
# Can't Be Late (Gemini)
GEMINI_API_KEY=... GEPA_ROOT=~/gepa  uv run --isolated \
    --with numpy --with cma --with google-genai \
    python3 llm_cantbelate_synth.py

# Cloudcast (Gemini) — extra graph deps
GEMINI_API_KEY=... GEPA_ROOT=~/gepa  uv run --isolated \
    --with numpy --with cma --with networkx --with pandas --with google-genai \
    python3 llm_cloudcast_synth.py

# Cleanup (Gemini) — self-contained
GEMINI_API_KEY=...  uv run --isolated --with numpy --with cma --with google-genai \
    python3 llm_cleanup_synth.py

# No API key: validate the harness wiring for any task
uv run --isolated --with numpy --with cma python3 llm_cleanup_synth.py --selftest
```

The GEPA-outer variants additionally install the local `gepa` package and `litellm`:

```bash
GEMINI_API_KEY=... GEPA_ROOT=~/gepa  uv run --isolated \
    --with "$GEPA_ROOT" --with litellm --with numpy --with cma --with google-genai \
    python3 gepa_hybrid_cantbelate.py
```

`LLM_PROVIDER` / `MODEL` select the proposer (Gemini, OpenRouter, OpenAI) exactly as
in the meta-optimizer experiment.

## Requirements

Python 3.10+ with `numpy` and `cma`; LLM runs additionally need the client SDK for
the chosen provider (`google-genai` for Gemini, `openai` for OpenRouter/OpenAI). The
commands above use [`uv`](https://github.com/astral-sh/uv) to provision these in an
isolated environment; a plain `pip install numpy cma google-genai openai` works too.
The executable-policy experiments have additional dependencies described in their
section above (a GEPA checkout for the two cloud tasks; `networkx`/`pandas` for
Cloudcast; `gepa`/`litellm` for the GEPA-outer variants).
