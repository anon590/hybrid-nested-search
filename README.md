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

This repository is organized by the experiment families in the paper. **This first
drop contains the meta-optimizer experiments** (Section "Meta-optimizers"); the
other families will follow.

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

## Requirements

Python 3.10+ with `numpy` and `cma`; the LLM runs additionally need the client SDK
for the chosen provider (`google-genai` for Gemini, `openai` for
OpenRouter/OpenAI). The commands above use [`uv`](https://github.com/astral-sh/uv)
to provision these in an isolated environment; a plain `pip install numpy cma
google-genai openai` works too.
