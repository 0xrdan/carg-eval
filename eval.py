#!/usr/bin/env python3
"""CARG Evaluation Harness — CLI entrypoint.

Run curated test queries against the CARG chatbot (live or fixtures),
score responses across 5 quality dimensions, and generate reports.

Usage:
    python eval.py run --tag baseline           # Run all 30 queries
    python eval.py run --fixtures --tag test     # Run with fixture responses
    python eval.py compare <run_a> <run_b>       # Diff two runs
    python eval.py show <run>                    # Display a saved run
"""

import json
import os
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import click
from dotenv import load_dotenv

from report import print_report, write_summary_markdown, print_comparison
from scorers import score_result, has_llm_judge

load_dotenv()

DATASET_PATH = Path(__file__).parent / "dataset.json"
RUNS_DIR = Path(__file__).parent / "runs"


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TestCase:
    id: str
    query: str
    route: str
    category: str
    expected: dict
    fixture_response: str


@dataclass
class Scores:
    accuracy: float = 0.0
    relevance: float = 0.0
    groundedness: float = 0.0
    efficiency: float = 0.0
    format_compliance: float = 0.0

    def as_dict(self) -> dict:
        return asdict(self)

    @property
    def average(self) -> float:
        vals = [self.accuracy, self.relevance, self.groundedness,
                self.efficiency, self.format_compliance]
        return sum(vals) / len(vals)


@dataclass
class Result:
    test_case: TestCase
    response: str
    scores: Scores
    latency_ms: float = 0.0
    source: str = "fixture"  # "fixture" | "live"
    scorer_mode: str = "rule"  # "rule" | "hybrid"


@dataclass
class RunResult:
    tag: str
    timestamp: str
    results: list[Result] = field(default_factory=list)
    mode: str = "fixture"  # "fixture" | "live"
    scorer_mode: str = "rule"

    @property
    def run_dir_name(self) -> str:
        return f"{self.tag}_{self.timestamp}"


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_dataset() -> list[TestCase]:
    """Load test cases from dataset.json."""
    with open(DATASET_PATH) as f:
        raw = json.load(f)
    return [TestCase(**item) for item in raw]


# ---------------------------------------------------------------------------
# Response fetchers
# ---------------------------------------------------------------------------

def fetch_fixture_response(test_case: TestCase) -> tuple[str, float]:
    """Return the hand-crafted fixture response with simulated latency."""
    # Simulate route-appropriate latency (ms)
    latency_map = {
        "fast": 800,
        "standard": 3000,
        "deep": 7000,
        "creative": 4000,
        "research": 5000,
    }
    simulated_latency = latency_map.get(test_case.route, 3000)
    return test_case.fixture_response, float(simulated_latency)


def fetch_live_response(test_case: TestCase, api_base: str, delay: float) -> tuple[str, float]:
    """Hit the CARG API and return the response + actual latency."""
    import requests

    url = f"{api_base.rstrip('/')}/chat"
    payload = {"message": test_case.query}

    if delay > 0:
        time.sleep(delay)

    start = time.perf_counter()
    resp = requests.post(url, json=payload, timeout=30)
    elapsed_ms = (time.perf_counter() - start) * 1000

    resp.raise_for_status()
    data = resp.json()

    # Extract response text — adapt to actual API shape
    response_text = data.get("response") or data.get("message") or data.get("text", "")
    return response_text, elapsed_ms


# ---------------------------------------------------------------------------
# Run execution
# ---------------------------------------------------------------------------

def execute_run(
    tag: str,
    use_fixtures: bool = True,
    delay: float = 0.0,
) -> RunResult:
    """Execute all test cases and return scored results."""
    dataset = load_dataset()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    api_base = os.getenv("CARG_API_BASE", "")
    use_llm = has_llm_judge()

    run = RunResult(
        tag=tag,
        timestamp=timestamp,
        mode="fixture" if use_fixtures else "live",
        scorer_mode="hybrid" if use_llm else "rule",
    )

    total = len(dataset)
    for i, tc in enumerate(dataset, 1):
        click.echo(f"  [{i}/{total}] {tc.id}: {tc.query[:50]}...")

        if use_fixtures:
            response, latency = fetch_fixture_response(tc)
        else:
            if not api_base:
                raise click.ClickException(
                    "CARG_API_BASE not set. Use --fixtures for offline mode."
                )
            response, latency = fetch_live_response(tc, api_base, delay)

        scores = score_result(tc, response, latency)

        result = Result(
            test_case=tc,
            response=response,
            scores=scores,
            latency_ms=latency,
            source="fixture" if use_fixtures else "live",
            scorer_mode="hybrid" if use_llm else "rule",
        )
        run.results.append(result)

    return run


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(run: RunResult) -> Path:
    """Save run results to runs/<tag>_<timestamp>/."""
    run_dir = RUNS_DIR / run.run_dir_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    results_data = {
        "tag": run.tag,
        "timestamp": run.timestamp,
        "mode": run.mode,
        "scorer_mode": run.scorer_mode,
        "results": [
            {
                "id": r.test_case.id,
                "query": r.test_case.query,
                "route": r.test_case.route,
                "category": r.test_case.category,
                "response": r.response,
                "latency_ms": r.latency_ms,
                "source": r.source,
                "scorer_mode": r.scorer_mode,
                "scores": r.scores.as_dict(),
            }
            for r in run.results
        ],
    }

    results_path = run_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump(results_data, f, indent=2)

    # Save markdown summary
    summary_path = run_dir / "summary.md"
    write_summary_markdown(run, summary_path)

    return run_dir


def load_run(run_name: str) -> RunResult:
    """Load a saved run from disk."""
    run_dir = RUNS_DIR / run_name
    results_path = run_dir / "results.json"

    if not results_path.exists():
        raise click.ClickException(f"Run not found: {run_name}")

    with open(results_path) as f:
        data = json.load(f)

    run = RunResult(
        tag=data["tag"],
        timestamp=data["timestamp"],
        mode=data.get("mode", "unknown"),
        scorer_mode=data.get("scorer_mode", "rule"),
    )

    for r in data["results"]:
        tc = TestCase(
            id=r["id"],
            query=r["query"],
            route=r["route"],
            category=r["category"],
            expected={},  # Not persisted in results
            fixture_response="",
        )
        scores = Scores(**r["scores"])
        result = Result(
            test_case=tc,
            response=r["response"],
            scores=scores,
            latency_ms=r["latency_ms"],
            source=r.get("source", "unknown"),
            scorer_mode=r.get("scorer_mode", "rule"),
        )
        run.results.append(result)

    return run


def list_runs() -> list[str]:
    """List available run directories."""
    if not RUNS_DIR.exists():
        return []
    return sorted(
        [d.name for d in RUNS_DIR.iterdir() if d.is_dir()],
        reverse=True,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

@click.group()
def cli():
    """CARG Evaluation Harness — measure chatbot quality across 5 dimensions."""
    pass


@cli.command()
@click.option("--tag", required=True, help="Label for this run (e.g. 'baseline', 'after-change').")
@click.option("--fixtures", is_flag=True, help="Use fixture responses instead of live API.")
@click.option("--delay", default=0.5, type=float, help="Delay between live API calls (seconds).")
def run(tag: str, fixtures: bool, delay: float):
    """Execute evaluation run against all test cases."""
    mode_label = "fixtures" if fixtures else "live API"
    scorer_label = "hybrid (rule + LLM)" if has_llm_judge() else "rule-based only"

    click.echo(f"\n  CARG Eval — running {mode_label}, scoring: {scorer_label}\n")

    result = execute_run(tag, use_fixtures=fixtures, delay=delay)
    run_dir = save_run(result)

    click.echo("")
    print_report(result)
    click.echo(f"\n  Results saved to: {run_dir}")
    click.echo(f"  Summary: {run_dir / 'summary.md'}\n")


@cli.command()
@click.argument("run_a")
@click.argument("run_b")
def compare(run_a: str, run_b: str):
    """Compare two runs and show score deltas."""
    # Allow partial matching for convenience
    all_runs = list_runs()
    run_a = _resolve_run_name(run_a, all_runs)
    run_b = _resolve_run_name(run_b, all_runs)

    click.echo(f"\n  Comparing: {run_a} vs {run_b}\n")
    a = load_run(run_a)
    b = load_run(run_b)
    print_comparison(a, b)


@cli.command()
@click.argument("run_name")
def show(run_name: str):
    """Display a saved run's report."""
    all_runs = list_runs()
    run_name = _resolve_run_name(run_name, all_runs)

    result = load_run(run_name)
    click.echo("")
    print_report(result)


@cli.command(name="list")
def list_cmd():
    """List all saved runs."""
    runs = list_runs()
    if not runs:
        click.echo("  No saved runs found.")
        return

    click.echo("\n  Saved runs:")
    for r in runs:
        click.echo(f"    {r}")
    click.echo("")


def _resolve_run_name(partial: str, all_runs: list[str]) -> str:
    """Resolve a partial run name to a full directory name."""
    # Exact match
    if partial in all_runs:
        return partial

    # Prefix match
    matches = [r for r in all_runs if r.startswith(partial)]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise click.ClickException(
            f"Ambiguous run name '{partial}'. Matches: {', '.join(matches)}"
        )

    raise click.ClickException(
        f"Run not found: '{partial}'. Available: {', '.join(all_runs) or 'none'}"
    )


if __name__ == "__main__":
    cli()
