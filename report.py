"""Report generation for CARG evaluation runs.

Produces two output formats:
- Terminal: ANSI-colored table output via Click
- Markdown: Persistent summary saved alongside results.json
"""

from collections import defaultdict
from pathlib import Path

import click


# ---------------------------------------------------------------------------
# ANSI color helpers
# ---------------------------------------------------------------------------

def _score_color(score: float) -> str:
    """Return ANSI color code based on score threshold."""
    if score >= 0.8:
        return "green"
    elif score >= 0.6:
        return "yellow"
    else:
        return "red"


def _score_bar(score: float, width: int = 15) -> str:
    """Render a score as a filled bar: [████████░░░░░░░] 0.82"""
    filled = int(score * width)
    empty = width - filled
    bar = "\u2588" * filled + "\u2591" * empty
    return f"[{bar}]"


def _delta_str(delta: float) -> str:
    """Format a delta as +0.05 or -0.03 with color."""
    sign = "+" if delta >= 0 else ""
    color = "green" if delta >= 0 else "red"
    return click.style(f"{sign}{delta:.3f}", fg=color)


# ---------------------------------------------------------------------------
# Aggregate computations
# ---------------------------------------------------------------------------

def _compute_dimension_averages(results) -> dict:
    """Compute average score per dimension across all results."""
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]
    totals = {d: 0.0 for d in dims}
    count = len(results)

    for r in results:
        for d in dims:
            totals[d] += getattr(r.scores, d)

    return {d: round(totals[d] / count, 3) if count > 0 else 0.0 for d in dims}


def _compute_route_breakdown(results) -> dict:
    """Compute average scores grouped by route."""
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]
    by_route = defaultdict(lambda: {"count": 0, **{d: 0.0 for d in dims}})

    for r in results:
        route = r.test_case.route
        by_route[route]["count"] += 1
        for d in dims:
            by_route[route][d] += getattr(r.scores, d)

    # Average
    for route, data in by_route.items():
        count = data["count"]
        for d in dims:
            data[d] = round(data[d] / count, 3) if count > 0 else 0.0

    return dict(by_route)


def _find_flags(results, threshold: float = 0.5) -> list:
    """Find results scoring below threshold on any dimension."""
    flags = []
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]

    for r in results:
        low_dims = []
        for d in dims:
            score = getattr(r.scores, d)
            if score < threshold:
                low_dims.append((d, score))
        if low_dims:
            flags.append((r.test_case.id, r.test_case.query, low_dims))

    return flags


# ---------------------------------------------------------------------------
# Terminal report
# ---------------------------------------------------------------------------

def print_report(run):
    """Print a formatted terminal report for a run."""
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]
    dim_labels = {
        "accuracy": "Accuracy",
        "relevance": "Relevance",
        "groundedness": "Grounded",
        "efficiency": "Efficiency",
        "format_compliance": "Format",
    }

    # Header
    click.echo(click.style("=" * 65, fg="cyan"))
    click.echo(click.style(f"  CARG Eval Report: {run.tag}", fg="cyan", bold=True))
    click.echo(click.style(f"  {run.timestamp} | {run.mode} mode | {run.scorer_mode} scoring", dim=True))
    click.echo(click.style("=" * 65, fg="cyan"))

    # Overall averages
    avgs = _compute_dimension_averages(run.results)
    overall = sum(avgs.values()) / len(avgs)

    click.echo(click.style("\n  Overall Score", bold=True))
    click.echo(f"  {_score_bar(overall)} {click.style(f'{overall:.3f}', fg=_score_color(overall), bold=True)}")

    click.echo(click.style("\n  Dimension Averages", bold=True))
    for d in dims:
        label = dim_labels[d].ljust(12)
        score = avgs[d]
        bar = _score_bar(score)
        colored = click.style(f"{score:.3f}", fg=_score_color(score))
        click.echo(f"    {label} {bar} {colored}")

    # Per-route breakdown
    route_data = _compute_route_breakdown(run.results)
    route_order = ["fast", "standard", "deep", "creative", "research"]

    click.echo(click.style("\n  Per-Route Breakdown", bold=True))
    header = "    Route       " + "".join(dim_labels[d][:6].ljust(10) for d in dims) + "Avg"
    click.echo(click.style(header, dim=True))
    click.echo(click.style("    " + "-" * 75, dim=True))

    for route in route_order:
        if route not in route_data:
            continue
        data = route_data[route]
        line = f"    {route.ljust(12)}"
        scores = []
        for d in dims:
            s = data[d]
            scores.append(s)
            line += click.style(f"{s:.3f}", fg=_score_color(s)).ljust(19)  # Account for ANSI codes
        avg = sum(scores) / len(scores)
        line += click.style(f"{avg:.3f}", fg=_score_color(avg))
        click.echo(line)

    # Individual results
    click.echo(click.style("\n  Individual Results", bold=True))
    click.echo(click.style(f"    {'ID':<14} {'Route':<10} {'Avg':>6}  Query", dim=True))
    click.echo(click.style("    " + "-" * 70, dim=True))

    for r in run.results:
        avg = r.scores.average
        color = _score_color(avg)
        query_short = r.test_case.query[:42] + "..." if len(r.test_case.query) > 45 else r.test_case.query
        click.echo(
            f"    {r.test_case.id:<14} {r.test_case.route:<10} "
            f"{click.style(f'{avg:.3f}', fg=color):>15}  {query_short}"
        )

    # Flagged regressions
    flags = _find_flags(run.results)
    if flags:
        click.echo(click.style(f"\n  Flagged ({len(flags)} issues)", fg="red", bold=True))
        for test_id, query, low_dims in flags:
            dims_str = ", ".join(
                f"{d}={click.style(f'{s:.2f}', fg='red')}"
                for d, s in low_dims
            )
            click.echo(f"    {test_id}: {dims_str}")
            click.echo(click.style(f"      \"{query}\"", dim=True))

    click.echo("")


# ---------------------------------------------------------------------------
# Comparison report
# ---------------------------------------------------------------------------

def print_comparison(run_a, run_b):
    """Print a comparison between two runs."""
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]
    dim_labels = {
        "accuracy": "Accuracy",
        "relevance": "Relevance",
        "groundedness": "Grounded",
        "efficiency": "Efficiency",
        "format_compliance": "Format",
    }

    avgs_a = _compute_dimension_averages(run_a.results)
    avgs_b = _compute_dimension_averages(run_b.results)

    click.echo(click.style("=" * 65, fg="cyan"))
    click.echo(click.style("  CARG Eval — Run Comparison", fg="cyan", bold=True))
    click.echo(click.style("=" * 65, fg="cyan"))
    click.echo(f"  A: {run_a.tag} ({run_a.timestamp})")
    click.echo(f"  B: {run_b.tag} ({run_b.timestamp})")

    # Dimension deltas
    click.echo(click.style("\n  Dimension Deltas (B - A)", bold=True))
    for d in dims:
        label = dim_labels[d].ljust(12)
        a_val = avgs_a[d]
        b_val = avgs_b[d]
        delta = b_val - a_val
        click.echo(
            f"    {label} {a_val:.3f} → {b_val:.3f}  {_delta_str(delta)}"
        )

    overall_a = sum(avgs_a.values()) / len(avgs_a)
    overall_b = sum(avgs_b.values()) / len(avgs_b)
    overall_delta = overall_b - overall_a

    click.echo(click.style(f"\n    {'Overall'.ljust(12)} {overall_a:.3f} → {overall_b:.3f}  {_delta_str(overall_delta)}", bold=True))

    # Per-route deltas
    route_a = _compute_route_breakdown(run_a.results)
    route_b = _compute_route_breakdown(run_b.results)
    route_order = ["fast", "standard", "deep", "creative", "research"]

    click.echo(click.style("\n  Per-Route Deltas", bold=True))
    for route in route_order:
        if route not in route_a or route not in route_b:
            continue
        a_avg = sum(route_a[route][d] for d in dims) / len(dims)
        b_avg = sum(route_b[route][d] for d in dims) / len(dims)
        delta = b_avg - a_avg
        click.echo(f"    {route.ljust(12)} {a_avg:.3f} → {b_avg:.3f}  {_delta_str(delta)}")

    # Per-test deltas (only show regressions)
    results_a = {r.test_case.id: r for r in run_a.results}
    results_b = {r.test_case.id: r for r in run_b.results}

    regressions = []
    improvements = []

    for test_id in results_a:
        if test_id not in results_b:
            continue
        a_avg = results_a[test_id].scores.average
        b_avg = results_b[test_id].scores.average
        delta = b_avg - a_avg

        if delta < -0.05:
            regressions.append((test_id, a_avg, b_avg, delta))
        elif delta > 0.05:
            improvements.append((test_id, a_avg, b_avg, delta))

    if regressions:
        click.echo(click.style(f"\n  Regressions ({len(regressions)})", fg="red", bold=True))
        for test_id, a_avg, b_avg, delta in sorted(regressions, key=lambda x: x[3]):
            click.echo(
                f"    {test_id:<14} {a_avg:.3f} → {b_avg:.3f}  {_delta_str(delta)}"
            )

    if improvements:
        click.echo(click.style(f"\n  Improvements ({len(improvements)})", fg="green", bold=True))
        for test_id, a_avg, b_avg, delta in sorted(improvements, key=lambda x: -x[3]):
            click.echo(
                f"    {test_id:<14} {a_avg:.3f} → {b_avg:.3f}  {_delta_str(delta)}"
            )

    if not regressions and not improvements:
        click.echo(click.style("\n  No significant per-test changes (threshold: ±0.05)", dim=True))

    click.echo("")


# ---------------------------------------------------------------------------
# Markdown summary
# ---------------------------------------------------------------------------

def write_summary_markdown(run, path: Path):
    """Write a markdown summary report to disk."""
    dims = ["accuracy", "relevance", "groundedness", "efficiency", "format_compliance"]
    dim_labels = {
        "accuracy": "Accuracy",
        "relevance": "Relevance",
        "groundedness": "Groundedness",
        "efficiency": "Efficiency",
        "format_compliance": "Format Compliance",
    }

    avgs = _compute_dimension_averages(run.results)
    overall = sum(avgs.values()) / len(avgs)
    route_data = _compute_route_breakdown(run.results)

    lines = []
    lines.append(f"# CARG Eval Report: {run.tag}")
    lines.append("")
    lines.append(f"**Timestamp:** {run.timestamp}  ")
    lines.append(f"**Mode:** {run.mode}  ")
    lines.append(f"**Scoring:** {run.scorer_mode}  ")
    lines.append(f"**Test cases:** {len(run.results)}  ")
    lines.append(f"**Overall Score:** {overall:.3f}")
    lines.append("")

    # Dimension averages
    lines.append("## Dimension Averages")
    lines.append("")
    lines.append("| Dimension | Score |")
    lines.append("|-----------|-------|")
    for d in dims:
        lines.append(f"| {dim_labels[d]} | {avgs[d]:.3f} |")
    lines.append(f"| **Overall** | **{overall:.3f}** |")
    lines.append("")

    # Per-route breakdown
    lines.append("## Per-Route Breakdown")
    lines.append("")
    header = "| Route | " + " | ".join(dim_labels[d] for d in dims) + " | Avg |"
    separator = "|-------|" + "|".join("------" for _ in dims) + "|-----|"
    lines.append(header)
    lines.append(separator)

    for route in ["fast", "standard", "deep", "creative", "research"]:
        if route not in route_data:
            continue
        data = route_data[route]
        scores = [data[d] for d in dims]
        avg = sum(scores) / len(scores)
        row = f"| {route} | " + " | ".join(f"{data[d]:.3f}" for d in dims) + f" | {avg:.3f} |"
        lines.append(row)
    lines.append("")

    # Individual results
    lines.append("## Individual Results")
    lines.append("")
    lines.append("| ID | Route | Query | Avg | Acc | Rel | Gnd | Eff | Fmt |")
    lines.append("|----|-------|-------|-----|-----|-----|-----|-----|-----|")

    for r in run.results:
        s = r.scores
        avg = s.average
        query = r.test_case.query[:40] + "..." if len(r.test_case.query) > 40 else r.test_case.query
        lines.append(
            f"| {r.test_case.id} | {r.test_case.route} | {query} | "
            f"{avg:.3f} | {s.accuracy:.3f} | {s.relevance:.3f} | "
            f"{s.groundedness:.3f} | {s.efficiency:.3f} | {s.format_compliance:.3f} |"
        )
    lines.append("")

    # Flags
    flags = _find_flags(run.results)
    if flags:
        lines.append("## Flagged Issues")
        lines.append("")
        for test_id, query, low_dims in flags:
            dims_str = ", ".join(f"{d}={s:.2f}" for d, s in low_dims)
            lines.append(f"- **{test_id}**: {dims_str}")
            lines.append(f"  - Query: \"{query}\"")
        lines.append("")

    lines.append("---")
    lines.append(f"*Generated by CARG Eval Harness*")

    path.write_text("\n".join(lines))
