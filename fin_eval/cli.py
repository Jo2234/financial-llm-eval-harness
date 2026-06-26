from __future__ import annotations

import json
from pathlib import Path

import typer

from .runner import compare_runs, load_cases, run_suite, validate_cases

app = typer.Typer(help="Financial LLM evaluation harness")


def _split_csv(value: str | None) -> list[str] | None:
    if not value:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


@app.command()
def run(
    suite: str = typer.Option(..., "--suite", "-s", help="YAML or JSON eval suite path."),
    target: str = typer.Option("mock", "--target", "-t", help="Target adapter: mock or copilot-api."),
    base_url: str | None = typer.Option(None, "--base-url", help="Base URL for copilot-api target."),
    endpoint: str = typer.Option("/research/chat", "--endpoint", help="API endpoint for copilot-api target."),
    timeout_s: float = typer.Option(20.0, "--timeout", help="Per-case target API timeout in seconds."),
    out: str = typer.Option("runs/latest", "--out", "-o", help="Directory for run artifacts."),
    fixture: str | None = typer.Option(None, "--fixture", help="Mock response fixture JSON/YAML."),
    case_ids: str | None = typer.Option(None, "--case-id", help="Comma-separated case IDs to run."),
    categories: str | None = typer.Option(None, "--category", help="Comma-separated categories to run."),
    tags: str | None = typer.Option(None, "--tag", help="Comma-separated tags to run."),
    limit: int | None = typer.Option(None, "--limit", help="Limit number of selected cases."),
    threshold_overall: float | None = typer.Option(None, "--threshold-overall", help="Minimum overall score."),
    threshold_citation_precision: float | None = typer.Option(None, "--threshold-citation-precision", help="Minimum citation precision."),
    threshold_citation_recall: float | None = typer.Option(None, "--threshold-citation-recall", help="Minimum citation recall."),
    threshold_refusal_accuracy: float | None = typer.Option(None, "--threshold-refusal-accuracy", help="Minimum refusal accuracy."),
    max_error_rate: float | None = typer.Option(None, "--max-error-rate", help="Maximum error rate."),
    max_latency_ms: int | None = typer.Option(None, "--max-median-latency-ms", help="Maximum median latency in milliseconds."),
) -> None:
    thresholds = {
        key: value
        for key, value in {
            "overall_score": threshold_overall,
            "citation_precision": threshold_citation_precision,
            "citation_recall": threshold_citation_recall,
            "refusal_accuracy": threshold_refusal_accuracy,
            "error_rate": max_error_rate,
            "max_median_latency_ms": max_latency_ms,
        }.items()
        if value is not None
    }
    result = run_suite(
        suite=suite,
        target=target,
        out=out,
        base_url=base_url,
        endpoint=endpoint,
        timeout_s=timeout_s,
        fixture=fixture,
        case_ids=_split_csv(case_ids),
        categories=_split_csv(categories),
        tags=_split_csv(tags),
        limit=limit,
        thresholds=thresholds,
    )
    typer.echo(json.dumps({"passed": result["passed"], "out": out, **result["summary"]}, indent=2))
    raise typer.Exit(code=0 if result["passed"] else 1)


@app.command("validate-suite")
def validate_suite(suite: str = typer.Option(..., "--suite", "-s", help="YAML or JSON eval suite path.")) -> None:
    cases = load_cases(suite)
    stats = validate_cases(cases)
    typer.echo(json.dumps({"valid": True, **stats}, indent=2))


@app.command()
def report(
    run: str = typer.Option(..., "--run", "-r", help="Run artifact directory."),
    format: str = typer.Option("markdown", "--format", "-f", help="markdown, html, or json."),
) -> None:
    if format == "html":
        path = Path(run) / "report.html"
        typer.echo(path.read_text())
        return
    if format == "json":
        path = Path(run) / "results.json"
        typer.echo(path.read_text())
        return
    if format != "markdown":
        raise typer.BadParameter("format must be markdown, html, or json")
    typer.echo((Path(run) / "summary.md").read_text())


@app.command("list-failures")
def list_failures(
    run: str = typer.Option(..., "--run", "-r", help="Run artifact directory."),
    json_output: bool = typer.Option(False, "--json", help="Emit machine-readable JSON."),
) -> None:
    payload = json.loads((Path(run) / "results.json").read_text())
    failures = [row for row in payload["results"] if not row["passed"]]
    if json_output:
        typer.echo(json.dumps(failures, indent=2))
        return
    for row in payload["results"]:
        if not row["passed"]:
            typer.echo(f"{row['case_id']}: {row['overall_score']:.3f} {row.get('error') or ''}")


@app.command()
def compare(
    baseline: str = typer.Option(..., "--baseline", "-b", help="Baseline run artifact directory."),
    candidate: str = typer.Option(..., "--candidate", "-c", help="Candidate run artifact directory."),
    gate: bool = typer.Option(False, "--gate", help="Exit 1 when regression thresholds fail."),
    overall_drop: float = typer.Option(0.03, "--max-overall-drop", help="Allowed overall score drop."),
    citation_precision_drop: float = typer.Option(0.05, "--max-citation-precision-drop", help="Allowed citation precision drop."),
    cost_increase_pct: float = typer.Option(0.25, "--max-cost-increase-pct", help="Allowed cost per case increase."),
) -> None:
    result = compare_runs(
        baseline=baseline,
        candidate=candidate,
        thresholds={
            "overall_score_drop": overall_drop,
            "citation_precision_drop": citation_precision_drop,
            "cost_per_case_increase_pct": cost_increase_pct,
        },
    )
    typer.echo(json.dumps(result, indent=2))
    if gate and not result["regression_pass"]:
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
