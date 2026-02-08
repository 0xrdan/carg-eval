# CARG Eval Harness

Evaluation harness for the [CARG](https://danmonteiro.com) (Context-Augmented Retrieval and Generation) chatbot system. Runs curated test queries, scores responses across 5 quality dimensions, and generates comparison reports.

Built to answer: *"Did this prompt change make things better or worse — and where?"*

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env  # optionally add API keys

# Run with fixture responses (offline, no API needed)
python eval.py run --fixtures --tag baseline

# View results
python eval.py show baseline

# Make a change, run again, compare
python eval.py run --fixtures --tag after-change
python eval.py compare baseline after-change
```

## Scoring Methodology

Responses are scored across 5 quality dimensions, drawn from the CARG evaluation methodology:

| Dimension | Method | What It Measures |
|-----------|--------|------------------|
| **Accuracy** | LLM-as-judge / keyword | Factual correctness given context |
| **Relevance** | LLM-as-judge / overlap | Query-response alignment |
| **Groundedness** | LLM-as-judge / attribution | Claims traceable to context |
| **Efficiency** | Rule-based | Word count vs target, latency vs route budget |
| **Format Compliance** | Rule-based | Markdown structure, no preamble, length bounds |

### Hybrid Scoring

When `ANTHROPIC_API_KEY` is set, accuracy/relevance/groundedness use **Claude Haiku as a judge** — an LLM evaluates the response quality on a 0-10 scale. This captures subjective dimensions that rules can't.

When no API key is set, all dimensions fall back to **rule-based heuristics** — keyword matching, term overlap, attribution pattern detection. Less accurate, but free, deterministic, and works offline.

### Route-Specific Targets

Each route has latency budgets from the production evaluation methodology:

| Route | P50 Target | P95 Target |
|-------|-----------|-----------|
| Fast | <2s | <4s |
| Standard | <5s | <8s |
| Deep | <10s | <15s |
| Research | <8s | <12s |

## Dataset

30 curated test cases in `dataset.json`:

- **Fast** (fast-01 to fast-05): Simple factual queries
- **Standard** (standard-01 to standard-05): Synthesis and exploration
- **Deep** (deep-01 to deep-05): Analysis, comparison, multi-step reasoning
- **Creative** (creative-01 to creative-05): Open-ended ideation
- **Research** (research-01 to research-05): Article-specific academic queries
- **Adversarial** (adversarial-01 to adversarial-05): Robustness testing — prompt injection, ambiguous multi-topic, out-of-scope (political), multilingual, overlong demand

Each test case includes a hand-crafted fixture response for offline evaluation.

## CLI Commands

```bash
python eval.py run --tag <name>              # Run against live CARG API
python eval.py run --fixtures --tag <name>    # Run with fixture responses
python eval.py compare <run_a> <run_b>        # Diff two runs
python eval.py show <run>                     # Display a saved run
python eval.py list                           # List all saved runs
```

**Options:**
- `--fixtures` — Use hand-crafted responses instead of hitting the API
- `--delay <seconds>` — Pause between live API calls (default: 0.5s)
- Run names support prefix matching: `baseline` matches `baseline_20260207_172117`

## Output

Each run saves to `runs/<tag>_<timestamp>/`:
- `results.json` — Full scores, raw responses, and metadata
- `summary.md` — Human-readable markdown report

## Architecture

```
eval.py       CLI entrypoint, data classes, run execution, persistence
scorers.py    Hybrid scoring engine (rule-based + LLM-as-judge)
report.py     Terminal (ANSI) and markdown report generation
dataset.json  30 curated test cases with fixture responses (25 standard + 5 adversarial)
```

Three files of logic. No config system, no plugin architecture. The dataset is JSON, the scoring logic is functions, and the reports are strings. Swap `dataset.json` and adjust `scorers.py` to evaluate a different chatbot.

## Environment Variables

| Variable | Purpose | Required |
|----------|---------|----------|
| `CARG_API_BASE` | CARG API endpoint for live evaluation | For live mode |
| `ANTHROPIC_API_KEY` | Enables LLM-as-judge scoring | No (falls back to rules) |
