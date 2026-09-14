from __future__ import annotations

import csv
import hashlib
import json
import platform
import re
import time
from datetime import datetime, timezone
from io import StringIO
from pathlib import Path
from typing import Any

import yaml

from .adapters import TARGET_REQUEST_FIELDS, CopilotApiAdapter, MockAdapter, load_fixture_responses
from .document_map import DocumentMap, apply_document_map, load_document_map, validate_document_map
from .models import EvalCase, TargetAdapter, TargetResponse
from .reporting import render_report
from .schema import SCHEMA_VERSION, EvalSuite, RunArtifact
from .scoring import DETERMINISTIC_CHECKS, SCORER_VERSION, aggregate, score_case

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


def load_suite(path: str | Path) -> EvalSuite:
    path = Path(path)
    raw = path.read_text()
    payload = json.loads(raw) if path.suffix.lower() == ".json" else yaml.safe_load(raw)
    if isinstance(payload, dict) and "cases" in payload:
        return EvalSuite(**payload)
    rows = payload
    if not isinstance(rows, list):
        raise ValueError("Eval suite must be a list of cases or an object with a 'cases' list")

    return EvalSuite(cases=rows)


def load_cases(path: str | Path) -> list[EvalCase]:
    return load_suite(path).cases


def validate_cases(cases: list[EvalCase], *, quality: bool = False) -> dict[str, Any]:
    seen: set[str] = set()
    duplicates: list[str] = []
    invalid_cases: list[str] = []
    for case in cases:
        if case.id in seen:
            duplicates.append(case.id)
        seen.add(case.id)
        if not case.question.strip():
            invalid_cases.append(f"{case.id}: question is required")
        if not case.refusal_expected and not case.expected_answer_points:
            invalid_cases.append(f"{case.id}: expected_answer_points are required unless refusal_expected=true")
        for pattern in case.required_answer_patterns:
            try:
                if not pattern.strip():
                    raise ValueError("empty pattern")
                re.compile(pattern, re.IGNORECASE)
            except (re.error, ValueError) as exc:
                invalid_cases.append(f"{case.id}: invalid required answer pattern: {exc}")
        if quality:
            if case.refusal_expected and case.expected_answer_points:
                invalid_cases.append(f"{case.id}: refusal cases must put behavioral requirements in judge_rubric, not answer points")
            if any(point.lower().startswith(("the answer should", "the response should", "the summary should")) for point in case.expected_answer_points):
                invalid_cases.append(f"{case.id}: expected answer points must be factual, not rubric instructions")
            evidence = case.source_evidence
            evidence_docs = {item.get("document_id") for item in evidence if isinstance(item, dict) and item.get("url") and item.get("excerpt")}
            if not case.refusal_expected:
                required_docs = {rule.get("document_id") for rule in case.required_citation_rules}
                if not case.documents or not set(case.documents) <= evidence_docs:
                    invalid_cases.append(f"{case.id}: every answer source needs an evidence excerpt and URL")
                if not set(case.documents) <= required_docs:
                    invalid_cases.append(f"{case.id}: citation rules must require every answer source")

    if duplicates:
        raise ValueError(f"Duplicate case IDs: {', '.join(sorted(set(duplicates)))}")
    if invalid_cases:
        raise ValueError("Invalid eval cases:\n" + "\n".join(invalid_cases))

    return {
        "case_count": len(cases),
        "categories": sorted({case.category for case in cases}),
        "difficulties": sorted({case.difficulty for case in cases}),
        "refusal_cases": sum(1 for case in cases if case.refusal_expected),
        "schema": SCHEMA_VERSION,
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


def case_fingerprint(case: EvalCase) -> str:
    encoded = json.dumps(_model_dump(case), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _case_result(case: EvalCase, response: TargetResponse, document_map: DocumentMap | None = None) -> dict[str, Any]:
    """Score one response. ``raw_response`` is preserved; only scored citations are canonicalized."""
    response = apply_document_map(response, document_map)
    result = score_case(case, response)
    result.update(
        {
            "case_definition": _model_dump(case),
            "case_fingerprint": case_fingerprint(case),
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
    document_map: str | Path | DocumentMap | None = None,
) -> dict[str, Any]:
    suite_document = load_suite(suite)
    all_cases = suite_document.cases
    quality = suite_document.suite_kind == "quality"
    validate_cases(all_cases, quality=quality)
    cases = _filter_cases(all_cases, case_ids=case_ids, categories=categories, tags=tags, limit=limit)
    if not cases:
        raise ValueError("No cases selected; an empty run cannot establish evaluation quality")
    # Validate identity mapping before any target request can be made.
    doc_map = document_map if isinstance(document_map, DocumentMap) or document_map is None else load_document_map(document_map)
    map_provenance = validate_document_map(doc_map, all_cases, cases) if doc_map else None
    if quality and target == "mock":
        if fixture is None:
            raise ValueError("Quality suites require an explicit --fixture for mock replay; use evals/plumbing.yaml for generated mock smoke tests")
        fixture_rows = load_fixture_responses(fixture)
        missing = sorted(case.id for case in cases if case.id not in fixture_rows)
        if missing:
            raise ValueError("Quality fixture is missing selected cases: " + ", ".join(missing))
        adapter = MockAdapter(fixture_rows)
    elif endpoint == "/research/chat" and timeout_s == 20.0 and fixture is None:
        adapter = adapter_for(target, base_url=base_url)
    else:
        adapter = adapter_for(target, base_url=base_url, endpoint=endpoint, timeout_s=timeout_s, fixture=fixture)
    out_path = Path(out)
    out_path.mkdir(parents=True, exist_ok=True)
    details: list[dict[str, Any]] = []
    started_at = datetime.now(timezone.utc).isoformat()
    started = time.perf_counter()

    for case in cases:
        response = adapter.answer(case)
        details.append(_case_result(case, response, doc_map))

    summary = summarize_results(details)
    gate = gate_summary(summary, thresholds=thresholds)
    run_metadata = {
        "suite": str(suite),
        "target": target,
        "base_url": base_url,
        "endpoint": endpoint if target == "copilot-api" else None,
        "target_request_fields": list(TARGET_REQUEST_FIELDS) if target == "copilot-api" else None,
        "case_count": len(details),
        "suite_kind": suite_document.suite_kind,
        "suite_version": suite_document.schema_version,
        "evaluation_mode": "fixture-replay" if target == "mock" and fixture else target,
        "rubric_evaluation": "not_performed",
        "rubric_review_required_cases": sum(bool(case.judge_rubric) for case in cases),
        "semantic_review": "not_performed",
        "deterministic_checks": DETERMINISTIC_CHECKS,
        "document_map": map_provenance,
        "started_at": started_at,
        "scorer_version": SCORER_VERSION,
        "duration_ms": int((time.perf_counter() - started) * 1000),
        "python_version": platform.python_version(),
        **(metadata or {}),
    }
    run_metadata["scorer_version"] = SCORER_VERSION
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
    (out_path / "metrics.csv").write_text(metrics_csv(payload))
    (out_path / "cases.jsonl").write_text(cases_jsonl(details))
    (out_path / "config.json").write_text(
        json.dumps(
            {
                "suite": str(suite),
                "target": target,
                "base_url": base_url,
                "endpoint": endpoint,
                "fixture": fixture,
                "document_map": map_provenance,
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
    manifest = artifact_manifest(out_path)
    (out_path / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
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


def metrics_csv(payload: dict[str, Any]) -> str:
    output = StringIO()
    writer = csv.DictWriter(output, fieldnames=["scope", "name", "metric", "value"])
    writer.writeheader()
    for metric, value in sorted(payload.get("summary", {}).items()):
        if isinstance(value, int | float):
            writer.writerow({"scope": "run", "name": "all", "metric": metric, "value": value})
    for category, metrics in sorted(payload.get("category_breakdown", {}).items()):
        for metric, value in sorted(metrics.items()):
            if isinstance(value, int | float):
                writer.writerow({"scope": "category", "name": category, "metric": metric, "value": value})
    return output.getvalue()


def cases_jsonl(results: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(row, sort_keys=True) + "\n" for row in results)


def artifact_manifest(out_path: Path) -> dict[str, Any]:
    artifacts = []
    for path in sorted(out_path.iterdir()):
        if path.is_file() and path.name != "manifest.json":
            artifacts.append({"name": path.name, "bytes": path.stat().st_size})
    return {"schema": "financial-eval-run-manifest/v1", "artifacts": artifacts}


def _recommendations(payload: dict[str, Any]) -> list[str]:
    summary = payload["summary"]
    recommendations = []
    if summary["severe_hallucination_count"]:
        recommendations.append("Review deterministic severe flags before release; prohibited-term, phrase-level assertion, direction-conflict or missing-refusal matches are not independent proof of hallucination.")
    if summary["citation_precision"] < DEFAULT_THRESHOLDS["citation_precision"]:
        recommendations.append("Inspect bad or missing citations before tuning answer prompts.")
    if summary["answer_point_recall"] < DEFAULT_THRESHOLDS["answer_point_recall"]:
        recommendations.append("Review missing expected points and retrieval coverage.")
    if summary["error_rate"] > DEFAULT_THRESHOLDS["error_rate"]:
        recommendations.append("Fix target API errors or timeouts before comparing model quality.")
    if not recommendations:
        recommendations.append("No blocking recommendations from deterministic gates.")
    return recommendations


def _flag_kinds(row: dict[str, Any]) -> str:
    kinds = sorted({item.get("kind", "assertion") for item in row.get("unsupported_assertions") or []})
    if row.get("contradicted_points"):
        kinds.append("contradicted_point")
    if row.get("must_not_include_hits"):
        kinds.append("must_not_include")
    return f" ({', '.join(kinds)})" if kinds else ""


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
        f"- `scorer_version`: {payload.get('metadata', {}).get('scorer_version', 'not recorded')}",
        f"- `suite_version`: {payload.get('metadata', {}).get('suite_version', 'not recorded')}",
        f"- `rubric_evaluation`: {payload.get('metadata', {}).get('rubric_evaluation', 'not performed')}",
        "",
        "The pass gate covers deterministic checks only; contextual judge rubrics are not automatically evaluated and semantic review is a separate step. Behavior metrics apply only to nonempty, error-free responses. Error and empty responses remain failed cases; refusal accuracy excludes them.",
        "",
        "## Aggregate Metrics",
    ]
    metadata = payload.get("metadata", {})
    if metadata.get("rescored_at"):
        lines[2:2] = [
            "Historical response rescore: saved answers were rescored offline; no new target requests were made.",
            f"Captured: `{metadata.get('started_at', 'not recorded')}`; rescored: `{metadata['rescored_at']}`.",
            f"Original artifact: `{metadata.get('original_artifact', 'not recorded')}`.",
            "",
        ]
    lines.insert(lines.index("## Aggregate Metrics"), f"Evaluation mode: `{metadata.get('evaluation_mode', 'historical-capture-rescore' if metadata.get('rescored_at') else payload['target'])}`.")
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
    lines += ["", "## Deterministic Severe Flags"]
    lines += [f"- `{r['case_id']}`: unsupported_claim_count={r['unsupported_claim_count']}{_flag_kinds(r)}" for r in severe[:25]] or ["None"]

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
    return render_report(payload, _recommendations(payload))


def load_run(path: str | Path) -> dict[str, Any]:
    payload = json.loads((Path(path) / "results.json").read_text())
    RunArtifact(**payload)
    return payload


def compare_markdown(result: dict[str, Any]) -> str:
    lines = [
        "# Financial QA Eval Run Comparison",
        "",
        f"Regression pass: `{result['regression_pass']}`",
        "",
        "## Metric Deltas",
        "| Metric | Baseline | Candidate | Delta |",
        "|---|---:|---:|---:|",
    ]
    baseline_summary = result["baseline"]["summary"]
    candidate_summary = result["candidate"]["summary"]
    for metric in sorted(result["delta"]):
        lines.append(
            f"| {metric} | {baseline_summary.get(metric)} | {candidate_summary.get(metric)} | {result['delta'][metric]} |"
        )
    lines += ["", "## Case Changes"]
    lines.append("- Comparable: " + str(result.get("comparable", False)))
    for name in ("added_cases", "removed_cases", "changed_cases"):
        lines.append("- " + name.replace("_", " ").capitalize() + ": " + (", ".join(result.get(name, [])) or "none"))
    lines.append("- New failures: " + (", ".join(result["new_failures"]) if result["new_failures"] else "none"))
    lines.append("- Fixed failures: " + (", ".join(result["fixed_failures"]) if result["fixed_failures"] else "none"))
    lines += ["", "## Regression Violations"]
    if result["violations"]:
        lines += ["| Metric | Rule | Delta |", "|---|---|---:|"]
        for violation in result["violations"]:
            lines.append(f"| {violation['metric']} | {violation['rule']} | {violation['delta']} |")
    else:
        lines.append("None")
    return "\n".join(lines) + "\n"


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
    baseline_rows = {row["case_id"]: row for row in baseline_payload["results"]}
    candidate_rows = {row["case_id"]: row for row in candidate_payload["results"]}
    common = baseline_rows.keys() & candidate_rows.keys()
    added = sorted(candidate_rows.keys() - baseline_rows.keys())
    removed = sorted(baseline_rows.keys() - candidate_rows.keys())
    incompatibilities = []
    if not baseline_rows or not candidate_rows:
        incompatibilities.append("empty case cohort")
    if len(baseline_rows) != len(baseline_payload["results"]) or len(candidate_rows) != len(candidate_payload["results"]):
        incompatibilities.append("duplicate case IDs")
    if added or removed:
        incompatibilities.append("case cohorts differ")
    versions = [p.get("metadata", {}).get("scorer_version") for p in (baseline_payload, candidate_payload)]
    compatible_versions = all(versions) and versions[0] == versions[1]
    if not compatible_versions:
        incompatibilities.append("missing or incompatible scorer versions")
    changed = sorted(case_id for case_id in common if not baseline_rows[case_id].get("case_fingerprint")
                     or baseline_rows[case_id].get("case_fingerprint") != candidate_rows[case_id].get("case_fingerprint"))
    if changed:
        incompatibilities.append("missing or incompatible case fingerprints")
    comparable = not incompatibilities
    comparable_cases = common - set(changed) if compatible_versions else set()

    metric_keys = sorted(set(baseline_summary) & set(candidate_summary))
    deltas = {
        key: candidate_summary[key] - baseline_summary[key]
        for key in metric_keys
        if comparable and isinstance(candidate_summary[key], int | float) and isinstance(baseline_summary[key], int | float)
    }

    baseline_failed = {row["case_id"] for row in baseline_payload["results"] if not row["passed"]}
    candidate_failed = {row["case_id"] for row in candidate_payload["results"] if not row["passed"]}

    violations = [{"metric": "comparability", "rule": reason, "delta": None} for reason in incompatibilities]
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
    if comparable and baseline_cost > 0 and candidate_cost > baseline_cost * (1 + thresholds["cost_per_case_increase_pct"]):
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
        "comparable": comparable,
        "incompatibilities": incompatibilities,
        "added_cases": added,
        "removed_cases": removed,
        "changed_cases": changed,
        "new_failures": sorted((candidate_failed - baseline_failed) & comparable_cases),
        "fixed_failures": sorted((baseline_failed - candidate_failed) & comparable_cases),
        "regression_pass": not violations,
        "violations": violations,
        "thresholds": thresholds,
    }
