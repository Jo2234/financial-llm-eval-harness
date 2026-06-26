from __future__ import annotations

import csv
import json
import time
from datetime import datetime, timezone
from html import escape
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from .adapters import CopilotApiAdapter, MockAdapter, load_fixture_responses
from .models import EvalCase, TargetAdapter, TargetResponse
from .scoring import aggregate, score_case


DEFAULT_THRESHOLDS = {
    "overall_score": 0.80,
    "answer_point_recall": 0.80,
    "citation_precision": 0.80,
    "citation_recall": 0.75,
    "refusal_accuracy": 0.90,
    "error_rate": 0.05,
    "max_severe_hallucination_count": 0,
    "max_median_latency_ms": 8000,
}

THRESHOLDS = DEFAULT_THRESHOLDS

REGRESSION_THRESHOLDS = {
    "overall_score_drop": 0.03,
    "citation_precision_drop": 0.05,
    "cost_per_case_increase_pct": 0.25,
}


def load_cases(path: str | Path) -> list[EvalCase]:
    path = Path(path)
    raw = path.read_text()
    payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    rows = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    if not isinstance(rows, list):
        raise ValueError("Eval suite must be a list of cases or an object with a 'cases' list")

    return [EvalCase(**row) for row in rows]


def validate_cases(cases: list[EvalCase]) -> dict[str, Any]:
    seen: set[str] = set()
    duplicates: list[str] = []
    for case in cases:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)

    if duplicates:
        raise ValueError(f"Duplicate case IDs: {', '.join(sorted(set(duplicates)))}")

    return {
        "case_count": len(cases),
        "categories": sorted({case.category for case in cases}),
        "difficulties": sorted({case.difficulty for case in cases}),
        "refusal_cases": sum(1 for case in cases if case.refusal_expected),
    }


def adapter_for(
    target: str,
    base_url: str | None = None,
    endpoint: str = "/research/chat",
    timeout_s: float = 20.0,
    fixture: str | None = None,
) -> TargetAdapter:
    if target == "mock":
        return MockAdapter(load_fixture_responses(fixture) if fixture else None)
    if target == "copilot-api":
        if not base_url:
            raise ValueError("base_url is required for copilot-api")
        return CopilotApiAdapter(base_url=base_url, endpoint=endpoint, timeout_s=timeout_s)
    raise ValueError(f"Unknown target: {target}")


def _model_dump(model: Any) -> dict[str, Any]:
    if hasattr(model, "model_dump"):
        return model.model_dump()
    return model.dict()


def _filter_cases(
    cases: list[EvalCase],
    case_ids: list[str] | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
) -> list[EvalCase]:
    selected = cases
    if case_ids:
        wanted = set(case_ids)
        selected = [case for case in selected if case.id in wanted]
    if categories:
        wanted = set(categories)
        selected = [case for case in selected if case.category in wanted]
    if tags:
        wanted = set(tags)
        selected = [case for case in selected if wanted.intersection(case.tags)]
    if limit is not None:
        selected = selected[:limit]
    return selected


def _case_result(case: EvalCase, response: TargetResponse) -> dict[str, Any]:
    result = score_case(case, response)
    result.update(
        {
            "question": case.question,
            "difficulty": case.difficulty,
            "tags": case.tags,
            "expected_answer_points": case.expected_answer_points,
            "must_not_include": case.must_not_include,
            "answer": response.answer,
            "citations": [_model_dump(citation) for citation in response.citations],
            "raw_response": response.raw_response,
            "model": response.model,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
        }
    )
    return result


def _category_breakdown(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_category: dict[str, list[dict[str, Any]]] = {}
    for result in results:
        by_category.setdefault(result["category"], []).append(result)

    return {category: aggregate(rows) for category, rows in sorted(by_category.items())}


def summarize_results(results: list[dict[str, Any]]) -> dict[str, Any]:
    summary = aggregate(results)
    total_cases = summary["total_cases"]
    summary["cost_per_case_usd"] = (
        summary["total_estimated_cost_usd"] / total_cases if total_cases else 0.0
    )
    return summary


def gate_summary(
    summary: dict[str, Any],
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    thresholds = {**DEFAULT_THRESHOLDS, **(thresholds or {})}
    checks = [
        ("overall_score", ">=", thresholds["overall_score"], summary.get("overall_score", 0.0)),
        ("answer_point_recall", ">=", thresholds["answer_point_recall"], summary.get("answer_point_recall", 0.0)),
        ("citation_precision", ">=", thresholds["citation_precision"], summary.get("citation_precision", 0.0)),
        ("citation_recall", ">=", thresholds["citation_recall"], summary.get("citation_recall", 0.0)),
        ("refusal_accuracy", ">=", thresholds["refusal_accuracy"], summary.get("refusal_accuracy", 0.0)),
        ("error_rate", "<=", thresholds["error_rate"], summary.get("error_rate", 1.0)),
        (
            "severe_hallucination_count",
            "<=",
            thresholds["max_severe_hallucination_count"],
            summary.get("severe_hallucination_count", 0),
        ),
        (
            "median_latency_ms",
            "<=",
            thresholds["max_median_latency_ms"],
            summary.get("median_latency_ms", 0),
        ),
    ]
    violations = []
    for metric, operator, threshold, value in checks:
        ok = value >= threshold if operator == ">=" else value <= threshold
        if not ok:
            violations.append(
                {
                    "metric": metric,
                    "operator": operator,
                    "threshold": threshold,
                    "actual": value,
                }
            )

    return {"passed": not violations, "thresholds": thresholds, "violations": violations}


def run_suite(
    suite: str | Path,
    target: str = "mock",
    out: str | Path = "runs/latest",
    base_url: str | None = None,
    endpoint: str = "/research/chat",
    timeout_s: float = 20.0,
    fixture: str | None = None,
    case_ids: list[str] | None = None,
    categories: list[str] | None = None,
    tags: list[str] | None = None,
    limit: int | None = None,
    metadata: dict[str, Any] | None = None,
    thresholds: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    cases = load_cases(suite)
    validate_cases(cases)
    cases = _filter_cases(cases, case_ids=case_ids, categories=categories, tags=tags, limit=limit)
    if endpoint == "/research/chat" and timeout_s == 20.0 and fixture is None:
        adapter = adapter_for(target, base_url=base_url)
    else:
        adapter = adapter_for(target, base_url=base_url, endpoint=endpoint, timeout_s=timeout_s, fixture=fixture)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    started = time.perf_counter()

    for case in cases:
        response = adapter.answer(case)
        details.append(_case_result(case, response))

    summary = summarize_results(details)
    gate = gate_summary(summary, thresholds=thresholds)
    run_metadata = {
        "suite": str(suite),
        "target": target,
        "base_url": base_url,
        "endpoint": endpoint if target == "copilot-api" else None,
        "case_count": len(details),
        "started_at": datetime.now(timezone.utc).isoformat(),
        "duration_ms": int((time.perf_counter() - started) * 1000),
        **(metadata or {}),
    }
    payload = {
        "metadata": run_metadata,
        "summary": summary,
        "category_breakdown": _category_breakdown(details),
        "passed": gate["passed"],
        "gate": gate,
        "results": details,
        "target": target,
    }
    (out_path / "results.json").write_text(json.dumps(payload, indent=2))
    (out_path / "summary.md").write_text(markdown_report(payload))
    (out_path / "report.html").write_text(html_report(payload))
    (out_path / "failures.csv").write_text(failures_csv(details))
    (out_path / "config.json").write_text(
        json.dumps(
            {
                "suite": str(suite),
                "target": target,
                "base_url": base_url,
                "endpoint": endpoint,
                "fixture": fixture,
                "filters": {
                    "case_ids": case_ids,
                    "categories": categories,
                    "tags": tags,
                    "limit": limit,
                },
                "thresholds": gate["thresholds"],
            },
            indent=2,
        )
    )
    return payload


def failures_csv(results: list[dict[str, Any]]) -> str:
    output = StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["case_id", "category", "difficulty", "overall_score", "error", "answer"],
        extrasaction="ignore",
    )
    writer.writeheader()
    for result in results:
        if not result["passed"]:
            writer.writerow(result)
    return output.getvalue()


def _recommendations(payload: dict[str, Any]) -> list[str]:
    summary = payload["summary"]
    recommendations = []
    if summary["severe_hallucination_count"]:
        recommendations.append("Block release until severe hallucination cases are reviewed.")
    if summary["citation_precision"] < DEFAULT_THRESHOLDS["citation_precision"]:
        recommendations.append("Inspect bad or missing citations before tuning answer prompts.")
    if summary["answer_point_recall"] < DEFAULT_THRESHOLDS["answer_point_recall"]:
        recommendations.append("Review missing expected points and retrieval coverage.")
    if summary["error_rate"] > DEFAULT_THRESHOLDS["error_rate"]:
        recommendations.append("Fix target API errors or timeouts before comparing model quality.")
    if not recommendations:
        recommendations.append("No blocking recommendations from deterministic gates.")
    return recommendations


def markdown_report(payload: dict[str, Any]) -> str:
    s = payload["summary"]
    lines = [
        "# Financial QA Evaluation Report",
        "",
        f"Target: `{payload['target']}`",
        f"Pass: `{payload['passed']}`",
        "",
        "## Run Metadata",
        f"- `target`: {payload['target']}",
        f"- `suite`: {payload.get('metadata', {}).get('suite')}",
        f"- `started_at`: {payload.get('metadata', {}).get('started_at')}",
        f"- `duration_ms`: {payload.get('metadata', {}).get('duration_ms')}",
        f"- `pass`: {payload['passed']}",
        "",
        "## Aggregate Metrics",
    ]
    for key, value in s.items():
        lines.append(f"- `{key}`: {value}")

    if payload.get("gate", {}).get("violations"):
        lines += ["", "## Gate Violations", "| Metric | Rule | Actual |", "|---|---:|---:|"]
        for violation in payload["gate"]["violations"]:
            lines.append(
                f"| {violation['metric']} | {violation['operator']} {violation['threshold']} | {violation['actual']} |"
            )

    lines += ["", "## Category Metrics", "| Category | Cases | Overall | Passed | Error Rate |", "|---|---:|---:|---:|---:|"]
    for category, metrics in payload.get("category_breakdown", {}).items():
        lines.append(
            f"| {category} | {metrics['total_cases']} | {metrics['overall_score']:.3f} | {metrics['passed_cases']} | {metrics['error_rate']:.3f} |"
        )

    failed = [r for r in payload["results"] if not r["passed"]]
    lines += ["", "## Failures", "| Case | Category | Score | Error |", "|---|---:|---:|---|"]
    for r in failed[:50]:
        lines.append(f"| {r['case_id']} | {r['category']} | {r['overall_score']:.3f} | {r.get('error') or ''} |")

    severe = [r for r in payload["results"] if r.get("severe_hallucination")]
    lines += ["", "## Severe Hallucinations"]
    lines += [f"- `{r['case_id']}`: unsupported_claim_count={r['unsupported_claim_count']}" for r in severe[:25]] or ["None"]

    slowest = sorted(payload["results"], key=lambda row: row.get("latency_ms") or 0, reverse=True)[:10]
    lines += ["", "## Slowest Cases", "| Case | Latency ms |", "|---|---:|"]
    for row in slowest:
        lines.append(f"| {row['case_id']} | {row.get('latency_ms') or 0} |")

    expensive = sorted(payload["results"], key=lambda row: row.get("estimated_cost_usd") or 0.0, reverse=True)[:10]
    lines += ["", "## Most Expensive Cases", "| Case | Cost USD |", "|---|---:|"]
    for row in expensive:
        lines.append(f"| {row['case_id']} | {row.get('estimated_cost_usd') or 0.0:.6f} |")

    lines += ["", "## Recommendations"]
    lines += [f"- {item}" for item in _recommendations(payload)]
    return "\n".join(lines) + "\n"


def html_report(payload: dict[str, Any]) -> str:
    rows = "".join(
        "<tr>"
        f"<td>{escape(r['case_id'])}</td>"
        f"<td>{escape(r['category'])}</td>"
        f"<td>{r['overall_score']:.3f}</td>"
        f"<td class=\"{'pass' if r['passed'] else 'fail'}\">{'pass' if r['passed'] else 'fail'}</td>"
        f"<td>{escape(r.get('error') or '')}</td>"
        "</tr>"
        for r in payload["results"]
    )
    category_rows = "".join(
        "<tr>"
        f"<td>{escape(category)}</td>"
        f"<td>{metrics['total_cases']}</td>"
        f"<td>{metrics['overall_score']:.3f}</td>"
        f"<td>{metrics['passed_cases']}</td>"
        "</tr>"
        for category, metrics in payload.get("category_breakdown", {}).items()
    )
    recommendations = "".join(f"<li>{escape(item)}</li>" for item in _recommendations(payload))
    return f"""<!doctype html><html><head><meta charset='utf-8'><title>Financial QA Eval Report</title><style>body{{font-family:Arial;margin:32px;background:#f7f8fa;color:#1f2933}}table{{border-collapse:collapse;width:100%;background:white;margin:16px 0}}td,th{{border:1px solid #d8dee9;padding:8px;text-align:left}}.pass{{color:#087f5b;font-weight:700}}.fail{{color:#c92a2a;font-weight:700}}pre{{background:white;border:1px solid #d8dee9;padding:16px;overflow:auto}}</style></head><body><h1>Financial QA Eval Report</h1><p>Target: {escape(payload['target'])} | Pass: <strong>{payload['passed']}</strong></p><h2>Aggregate Metrics</h2><pre>{escape(json.dumps(payload['summary'], indent=2))}</pre><h2>Category Metrics</h2><table><thead><tr><th>Category</th><th>Cases</th><th>Overall</th><th>Passed</th></tr></thead><tbody>{category_rows}</tbody></table><h2>Case Results</h2><table><thead><tr><th>Case</th><th>Category</th><th>Score</th><th>Status</th><th>Error</th></tr></thead><tbody>{rows}</tbody></table><h2>Recommendations</h2><ul>{recommendations}</ul></body></html>"""


def load_run(path: str | Path) -> dict[str, Any]:
    return json.loads((Path(path) / "results.json").read_text())


def compare_runs(
    baseline: str | Path,
    candidate: str | Path,
    thresholds: dict[str, float] | None = None,
) -> dict[str, Any]:
    thresholds = {**REGRESSION_THRESHOLDS, **(thresholds or {})}
    baseline_payload = load_run(baseline)
    candidate_payload = load_run(candidate)
    baseline_summary = baseline_payload["summary"]
    candidate_summary = candidate_payload["summary"]

    metric_keys = sorted(set(baseline_summary) & set(candidate_summary))
    deltas = {
        key: candidate_summary[key] - baseline_summary[key]
        for key in metric_keys
        if isinstance(candidate_summary[key], (int, float)) and isinstance(baseline_summary[key], (int, float))
    }

    baseline_failed = {row["case_id"] for row in baseline_payload["results"] if not row["passed"]}
    candidate_failed = {row["case_id"] for row in candidate_payload["results"] if not row["passed"]}

    violations = []
    if deltas.get("overall_score", 0.0) < -thresholds["overall_score_drop"]:
        violations.append(
            {
                "metric": "overall_score",
                "rule": f"drop <= {thresholds['overall_score_drop']}",
                "delta": deltas.get("overall_score", 0.0),
            }
        )
    if deltas.get("citation_precision", 0.0) < -thresholds["citation_precision_drop"]:
        violations.append(
            {
                "metric": "citation_precision",
                "rule": f"drop <= {thresholds['citation_precision_drop']}",
                "delta": deltas.get("citation_precision", 0.0),
            }
        )
    if deltas.get("severe_hallucination_count", 0) > 0:
        violations.append(
            {
                "metric": "severe_hallucination_count",
                "rule": "must not increase",
                "delta": deltas.get("severe_hallucination_count", 0),
            }
        )

    baseline_cost = baseline_summary.get("cost_per_case_usd") or 0.0
    candidate_cost = candidate_summary.get("cost_per_case_usd") or 0.0
    if baseline_cost > 0 and candidate_cost > baseline_cost * (1 + thresholds["cost_per_case_increase_pct"]):
        violations.append(
            {
                "metric": "cost_per_case_usd",
                "rule": f"increase <= {thresholds['cost_per_case_increase_pct']:.0%}",
                "delta": candidate_cost - baseline_cost,
            }
        )

    return {
        "baseline": {"path": str(baseline), "summary": baseline_summary},
        "candidate": {"path": str(candidate), "summary": candidate_summary},
        "delta": deltas,
        "new_failures": sorted(candidate_failed - baseline_failed),
        "fixed_failures": sorted(baseline_failed - candidate_failed),
        "regression_pass": not violations,
        "violations": violations,
        "thresholds": thresholds,
    }
