"""Create a public, allowlisted export of a capture made by run.py without changing any score.

Only reviewed metadata, curated evaluator definitions, measured scores, identifiers and
hashes are exported. All target free text (answers, key points, limitations, debug
passages, labels, titles, section titles, excerpts, error messages and unknown fields)
is dropped or replaced by a SHA-256 fingerprint. Unclassified scored-result fields stop
the export. The original capture is verified against its inventory and never modified.
Redacted public records cannot independently rescore answer text; the original local
capture or a new reproduction can.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.common import (  # noqa: E402
    COPILOT_MODEL,
    COPILOT_PROVIDER,
    DOCUMENT_TYPES,
    IDENTIFIER_RE,
    BenchmarkError,
    checked_fields,
    checked_settings,
    find_local_paths,
    is_count,
    is_date,
    is_hex_digest,
    is_loopback_url,
    is_public_source_url,
    is_timestamp,
    sha256_bytes,
    sha256_json,
    write_json,
)
from benchmark.run import BASELINE_CONFIG, CONFIG_TEXT, SOURCE_ORIGIN, TARGETS, public_ingestion  # noqa: E402
from fin_eval.adapters import DEFAULT_TOP_K  # noqa: E402
from fin_eval.document_map import mapping_sha256, parse_document_map  # noqa: E402
from fin_eval.runner import (  # noqa: E402
    _category_breakdown,
    artifact_manifest,
    cases_jsonl,
    failures_csv,
    gate_summary,
    html_report,
    markdown_report,
    metrics_csv,
    summarize_results,
)
from fin_eval.scoring import (  # noqa: E402
    ASSERTION_PATTERNS,
    DEFAULT_MAX_LATENCY_MS,
    DETERMINISTIC_CHECKS,
    QUANTITY_RE,
    SCORER_VERSION,
    rule_label,
)

NOTE = "[Response text omitted from public export; original fingerprint in answer_sha256]"
REDACTED = "[redacted]"
SAFE_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,127}")
ERROR_TYPE_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]{0,80}):")
CONFIDENCE_VALUES = {"low", "medium", "high"}

CITATION_FIELDS = (
    "document_id", "source_document_id", "document_map_status", "chunk_id", "company_id", "company_ticker",
    "page", "page_start", "page_end", "score", "overlap",
)
USAGE_FIELDS = ("provider", "model", "latency_ms", "input_tokens", "output_tokens", "estimated_cost_usd")
RAW_ID_FIELDS = ("model", "message_id", "conversation_id")
REQUEST_FIELDS = ("question", "company_ids", "top_k")
CAPTURE_FIELDS = {"case_id", "target", "request", "captured_at", "latency_ms", "raw_response", "response_sha256"}
CONTRADICTION_FIELDS = ("point", "quantity", "expected_direction", "observed_direction")
SOURCE_FIELDS = (
    "document_id", "document_type", "url", "period_end", "sha256", "normalized_text_sha256",
    "normalized_characters", "verified_contexts", "origin",
)
CONFIG_FIELDS = {
    "capture_started_at", "suite_sha256", "evidence_manifest_sha256", "runner_sha256", "harness", "copilot",
    "copilot_backend_tree", "scorer_version", "python", "api_base_url", "api_provider", "api_model",
    "provider_selection", "api_request_fields", "api_top_k", "baseline", "source_documents", "source_scope",
    "source_retrieved_on", "source_provenance_sha256", "document_map_sha256", "ingestion_manifest_sha256",
    "api_settings", "api_chunk_count", "ordering", "latency", "rubric_review",
}
PAYLOAD_FIELDS = {"metadata", "summary", "category_breakdown", "passed", "gate", "results", "target"}
METADATA_FIELDS = {
    "suite", "suite_version", "suite_kind", "target", "case_count", "evaluation_mode", "started_at", "duration_ms",
    "scorer_version", "deterministic_checks", "rubric_evaluation", "semantic_review", "rubric_review_required_cases",
    "document_map", "configuration_sha256",
}
REVISION_FIELDS = {"head", "clean", "changed_paths"}
DIAGNOSTIC_FIELDS = {"missing_citation_issue", "latency_budget_ms", "cost_budget_usd"}


def _numeric_object(value: Any, label: str) -> dict[str, Any]:
    """Metric values cannot carry free text or nested unclassified containers."""
    if not isinstance(value, dict) or any(not isinstance(item, bool | int | float) and item is not None for item in value.values()):
        raise BenchmarkError(f"{label} must contain numeric/boolean metrics only")
    if any(not SAFE_TOKEN_RE.fullmatch(key) for key in value):
        raise BenchmarkError(f"{label} metric names must be identifiers")
    return copy.deepcopy(value)


def _fixed(expected: Any):
    def check(value: Any) -> bool:
        if type(value) is not type(expected):
            return False
        if isinstance(expected, dict):
            return value.keys() == expected.keys() and all(_fixed(item)(value[key]) for key, item in expected.items())
        if isinstance(expected, list):
            return len(value) == len(expected) and all(_fixed(item)(actual) for item, actual in zip(expected, value, strict=True))
        return value == expected
    return check


def _revision(value: Any) -> bool:
    try:
        revision = checked_fields(value, REVISION_FIELDS, "revision")
    except BenchmarkError:
        return False
    return (
        set(revision) == REVISION_FIELDS
        and is_hex_digest(revision["head"], (40, 64))
        and type(revision["clean"]) is bool
        and is_count(revision["changed_paths"])
    )


def _settings(value: Any) -> bool:
    try:
        checked_settings(value)
    except BenchmarkError:
        return False
    return True


# Every exported config field has a typed/value validator; there is no generic scalar fallback.
CONFIG_VALIDATORS = {
    "capture_started_at": is_timestamp,
    "suite_sha256": is_hex_digest,
    "evidence_manifest_sha256": is_hex_digest,
    "runner_sha256": is_hex_digest,
    "source_provenance_sha256": is_hex_digest,
    "document_map_sha256": is_hex_digest,
    "ingestion_manifest_sha256": is_hex_digest,
    "copilot_backend_tree": lambda value: is_hex_digest(value, (40, 64)),
    "harness": _revision,
    "copilot": _revision,
    "scorer_version": _fixed(SCORER_VERSION),
    "python": lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9]+\.[0-9]+\.[0-9]+(?:(?:a|b|rc)[0-9]+)?", value)),
    "api_base_url": is_loopback_url,
    "api_provider": _fixed(COPILOT_PROVIDER),
    "api_model": _fixed(COPILOT_MODEL),
    "api_request_fields": _fixed(list(REQUEST_FIELDS)),
    "api_top_k": _fixed(DEFAULT_TOP_K),
    "baseline": _fixed(BASELINE_CONFIG),
    "source_documents": is_count,
    "source_retrieved_on": is_date,
    "api_settings": _settings,
    "api_chunk_count": is_count,
    **{key: _fixed(text) for key, text in CONFIG_TEXT.items()},
}
if set(CONFIG_VALIDATORS) != CONFIG_FIELDS:  # import-time completeness, also covered by tests
    raise RuntimeError("Every exported config field needs a validator")


def _validated(record: Any, validators: dict[str, Any], label: str) -> dict[str, Any]:
    result = checked_fields(record, set(validators), label)
    missing = sorted(set(validators) - set(result))
    if missing:
        raise BenchmarkError(f"Missing {label} fields: " + ", ".join(missing))
    invalid = []
    for key, check in validators.items():
        try:
            valid = check(result[key])
        except (TypeError, ValueError, KeyError):
            valid = False
        if not valid:
            invalid.append(key)
    invalid.sort()
    if invalid:
        raise BenchmarkError(f"Invalid {label} values: " + ", ".join(invalid))
    return copy.deepcopy(result)


def public_config(config: dict[str, Any]) -> dict[str, Any]:
    return _validated(config, CONFIG_VALIDATORS, "config")


def public_metadata(metadata: dict[str, Any], *, target: str, config: dict[str, Any], config_sha256: str, case_count: int) -> dict[str, Any]:
    """Result metadata, every value typed and linked to the capture configuration."""
    validators = {
        "suite": lambda value: value == "evals/core.yaml" or (isinstance(value, str) and bool(re.fullmatch(r"[A-Za-z0-9._-]+\.(?:json|ya?ml)", value))),
        "suite_version": lambda value: value is None or (isinstance(value, str) and bool(re.fullmatch(r"financial-eval-suite/v\d+", value))),
        "suite_kind": _fixed("quality"),
        "target": _fixed(target),
        "case_count": _fixed(case_count),
        "evaluation_mode": _fixed("fresh execution; deterministic no-LLM comparison"),
        "started_at": lambda value: is_timestamp(value) and value == config["capture_started_at"],
        "duration_ms": is_count,
        "scorer_version": lambda value: value == config["scorer_version"],
        "deterministic_checks": _fixed(list(DETERMINISTIC_CHECKS)),
        "rubric_evaluation": _fixed("not_performed"),
        "semantic_review": _fixed("not_performed"),
        "rubric_review_required_cases": _fixed(case_count),
        "document_map": _fixed({"mapping_sha256": config["document_map_sha256"]} if target == "copilot-local" else None),
        "configuration_sha256": _fixed(config_sha256),
    }
    return _validated(metadata, validators, f"{target} metadata")


def public_sources(sources: Any, ingestion: dict[str, Any], document_map: dict[str, str]) -> list[dict[str, Any]]:
    """Source provenance rows, typed and linked to the verified upload record."""
    if not isinstance(sources, list):
        raise BenchmarkError("Source provenance must be a list")
    uploads = {row["canonical_id"]: row for row in ingestion["documents"]}
    rows = []
    for source in sources:
        doc_id = source.get("document_id") if isinstance(source, dict) and isinstance(source.get("document_id"), str) else None
        upload = uploads.get(doc_id, {})
        metadata = upload.get("metadata", {})
        validators = {
            "document_id": lambda value: value in document_map.values() and value in uploads,
            "document_type": lambda value, m=metadata: isinstance(value, str) and value in DOCUMENT_TYPES and DOCUMENT_TYPES[value] == m.get("document_type", DOCUMENT_TYPES[value]),
            "url": lambda value, m=metadata: is_public_source_url(value) and value == m.get("source_url", value),
            "period_end": lambda value, m=metadata: is_date(value) and value == m.get("period_end_date", value),
            "sha256": lambda value, u=upload: is_hex_digest(value) and value == u.get("raw_sha256"),
            "normalized_text_sha256": is_hex_digest,
            "normalized_characters": is_count,
            "verified_contexts": is_count,
            "origin": _fixed(SOURCE_ORIGIN),
        }
        rows.append(_validated(source, validators, f"source {doc_id!r}"))
    return rows


def _case_fingerprint(definition: dict[str, Any]) -> str:
    return sha256_bytes(json.dumps(definition, sort_keys=True, separators=(",", ":")).encode())


def _score(value: Any) -> bool:
    return _nonnegative_number(value) and value <= 1.0


def _nonnegative_number(value: Any) -> bool:
    return type(value) in (int, float) and math.isfinite(value) and value >= 0


def _optional_bool(value: Any) -> bool:
    return value is None or type(value) is bool


def _subset(allowed: list[Any]):
    return lambda value: isinstance(value, list) and all(item in allowed for item in value)


def check_result_values(row: dict[str, Any]) -> None:
    """Non-gold result values must be typed and linked to the trusted case definition.

    ``case_definition`` is the curated gold boundary; copied gold fields must equal it.
    """
    definition = row.get("case_definition")
    if not isinstance(definition, dict) or row.get("case_fingerprint") != _case_fingerprint(definition):
        raise BenchmarkError("Result case definition does not match its fingerprint")
    rules = definition.get("required_citation_rules") or [{"document_id": doc} for doc in definition.get("documents", [])]
    gold = {
        "case_id": definition.get("id"), "category": definition.get("category"), "difficulty": definition.get("difficulty"),
        "tags": definition.get("tags"), "question": definition.get("question"),
        "expected_answer_points": definition.get("expected_answer_points"), "must_not_include": definition.get("must_not_include"),
    }
    validators = {
        **{key: _fixed(value) for key, value in gold.items()},
        "case_definition": lambda value: True,
        "case_fingerprint": is_hex_digest,
        "covered_points": _subset(gold["expected_answer_points"] or []),
        "missing_points": _subset(gold["expected_answer_points"] or []),
        "missing_answer_patterns": _subset(definition.get("required_answer_patterns") or []),
        "must_not_include_hits": _subset(gold["must_not_include"] or []),
        "missing_required_citations": _subset([rule_label(rule) for rule in rules]),
        "passed": lambda value: type(value) is bool,
        "behavior_evaluated": lambda value: type(value) is bool,
        "severe_hallucination": lambda value: type(value) is bool,
        "refused": _optional_bool,
        "refusal_correct": _optional_bool,
        "execution_status": lambda value: value in {"answered", "error", "empty"},
        **{key: _score for key in ("overall_score", "answer_point_recall", "citation_precision", "citation_recall", "format_score", "latency_score", "cost_score")},
        "unsupported_claim_count": is_count,
        "latency_ms": is_count,
        "total_tokens": is_count,
        "input_tokens": lambda value: value is None or is_count(value),
        "output_tokens": lambda value: value is None or is_count(value),
        "estimated_cost_usd": _nonnegative_number,
        "diagnostics": lambda value: True,  # typed by public_result_row
    }
    _validated({key: row[key] for key in validators if key in row}, validators, f"result {gold['case_id']!r}")


def public_gate(gate: dict[str, Any]) -> dict[str, Any]:
    result = checked_fields(gate, {"passed", "thresholds", "violations"}, "gate")
    result["thresholds"] = _numeric_object(result["thresholds"], "gate thresholds")
    if type(result["passed"]) is not bool or not isinstance(result["violations"], list):
        raise BenchmarkError("Invalid gate values")
    for item in result["violations"]:
        checked_fields(item, {"metric", "operator", "threshold", "actual"}, "gate violation")
        if not isinstance(item.get("metric"), str) or not SAFE_TOKEN_RE.fullmatch(item["metric"]) or item.get("operator") not in {">=", "<=", "="}:
            raise BenchmarkError("Invalid gate violation identity")
        if any(not isinstance(item.get(key), int | float) for key in ("threshold", "actual")):
            raise BenchmarkError("Invalid gate violation metrics")
    return result
# Scored-result fields derived from the curated suite or the scorer, safe to publish as-is.
EVALUATOR_RESULT_FIELDS = {
    "case_id", "category", "passed", "overall_score", "answer_point_recall", "covered_points", "missing_points",
    "missing_answer_patterns", "citation_precision", "citation_recall", "missing_required_citations",
    "behavior_evaluated", "execution_status", "refusal_correct", "refused", "format_score", "latency_score",
    "cost_score", "unsupported_claim_count", "must_not_include_hits", "severe_hallucination", "latency_ms",
    "input_tokens", "output_tokens", "total_tokens", "estimated_cost_usd", "diagnostics", "case_definition",
    "case_fingerprint", "question", "difficulty", "tags", "expected_answer_points", "must_not_include",
}
# Fields carrying target output; each has an explicit public transformation below.
TARGET_RESULT_FIELDS = {
    "answer", "citations", "raw_response", "error", "model", "unsupported_assertions", "bad_citations", "contradicted_points",
}


def digest_text(value: str) -> str:
    return sha256_bytes(value.encode())


def public_error(value: Any) -> dict[str, Any]:
    if not isinstance(value, str):
        return {"type": "error"}
    match = ERROR_TYPE_RE.match(value)
    return {"type": match.group(1) if match else "error", "sha256": digest_text(value), "characters": len(value)}


def public_citation(citation: Any) -> dict[str, Any]:
    if not isinstance(citation, dict):
        return {"omitted": True}
    result = {}
    for key in CITATION_FIELDS:
        if key not in citation:
            continue
        value = citation[key]
        if key in {"page", "page_start", "page_end", "overlap"}:
            result[key] = value if value is None or is_count(value) else REDACTED
        elif key == "score":
            result[key] = value if value is None or _nonnegative_number(value) else REDACTED
        elif key == "document_map_status":
            result[key] = value if isinstance(value, str) and value in {"mapped", "unmapped", "missing_document_id"} else REDACTED
        else:
            result[key] = public_identifier(value)
    result["omitted_field_count"] = sum(1 for key, value in citation.items() if key not in CITATION_FIELDS and value is not None)
    return result


def public_identifier(value: Any) -> Any:
    return value if value is None or (isinstance(value, str) and IDENTIFIER_RE.fullmatch(value)) else REDACTED


def public_raw_response(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {"omitted": True}
    result: dict[str, Any] = {}
    handled = {"answer", "key_points", "limitations", "citations", "confidence", "usage", "error", *RAW_ID_FIELDS}
    if isinstance(raw.get("answer"), str):
        result["answer_sha256"] = digest_text(raw["answer"])
        result["answer_characters"] = len(raw["answer"])
    for key in ("key_points", "limitations"):
        if isinstance(raw.get(key), list):
            result[f"{key}_count"] = len(raw[key])
    if isinstance(raw.get("citations"), list):
        result["citations"] = [public_citation(citation) for citation in raw["citations"]]
    if isinstance(raw.get("confidence"), str) and raw["confidence"] in CONFIDENCE_VALUES:
        result["confidence"] = raw["confidence"]
    if isinstance(raw.get("usage"), dict):
        result["usage"] = {}
        for key in USAGE_FIELDS:
            if key not in raw["usage"]:
                continue
            value = raw["usage"][key]
            if key in {"provider", "model"}:
                reviewed = {COPILOT_PROVIDER, COPILOT_MODEL, "local-baseline", "none"}
                result["usage"][key] = value if isinstance(value, str) and value in reviewed else REDACTED
            else:
                check = _nonnegative_number if key == "estimated_cost_usd" else is_count
                result["usage"][key] = value if value is None or check(value) else REDACTED
    for key in RAW_ID_FIELDS:
        if key in raw:
            result[key] = public_model(raw[key]) if key == "model" else public_identifier(raw[key])
    if "error" in raw:
        result["error"] = public_error(raw["error"])
    result["omitted_field_count"] = sum(1 for key in raw if key not in handled)
    return result


ASSERTION_KINDS = {kind for kind, _ in ASSERTION_PATTERNS} | {"unsupported_numeric_forecast"}
BAD_CITATION_REASONS = {
    "citation is empty or unstructured",
    "document_id is not in the verified document map",
    "citation has no document_id to verify against the document map",
    "does not match required documents, chunks, or citation rules",
}
DIRECTIONS = {"up", "down", "amount"}
OBSERVED_DIRECTIONS = {"up", "down", "not up", "not down", "negated amount"}


def _enum(value: Any, allowed: set[str], label: str) -> Any:
    if not isinstance(value, str) or value not in allowed:
        raise BenchmarkError(f"Unreviewed {label} value")
    return value


def public_model(value: Any) -> Any:
    reviewed = {COPILOT_MODEL, "none", "lexical-overlap-extractive-v1-no-llm"}
    return value if value is None or (isinstance(value, str) and value in reviewed) else REDACTED


def public_result_row(row: dict[str, Any]) -> dict[str, Any]:
    unknown = sorted(set(row) - EVALUATOR_RESULT_FIELDS - TARGET_RESULT_FIELDS)
    if unknown:
        raise BenchmarkError("Unclassified result fields; review and update the export allowlist: " + ", ".join(unknown))
    if set(row) != EVALUATOR_RESULT_FIELDS | TARGET_RESULT_FIELDS:
        raise BenchmarkError("Missing reviewed result fields")
    check_result_values(row)
    result = {key: copy.deepcopy(row[key]) for key in sorted(EVALUATOR_RESULT_FIELDS) if key in row}
    if "diagnostics" in result:
        definition = row["case_definition"]
        result["diagnostics"] = _validated(result["diagnostics"], {
            "missing_citation_issue": _fixed(bool(row["missing_required_citations"])),
            "latency_budget_ms": _fixed(definition.get("max_latency_ms") or DEFAULT_MAX_LATENCY_MS),
            "cost_budget_usd": _fixed(definition.get("max_estimated_cost_usd")),
        }, "scorer diagnostics")
    answer = row.get("answer") if isinstance(row.get("answer"), str) else ""
    result.update(
        answer=NOTE,
        answer_sha256=digest_text(answer),
        answer_characters=len(answer),
        citations=[public_citation(citation) for citation in row.get("citations") or []],
        raw_response=public_raw_response(row.get("raw_response")),
        error=public_error(row["error"])["type"] if row.get("error") is not None else None,
        model=public_model(row.get("model")),
        unsupported_assertions=[
            {"kind": _enum(item.get("kind"), ASSERTION_KINDS, "assertion kind"), "text_sha256": digest_text(str(item.get("text", "")))}
            for item in row.get("unsupported_assertions") or []
        ],
        bad_citations=[
            {
                "index": item.get("index") if is_count(item.get("index")) else _enum(None, set(), "citation index"),
                "document_id": public_identifier(item.get("document_id")),
                "chunk_id": public_identifier(item.get("chunk_id")),
                "reason": _enum(item.get("reason"), BAD_CITATION_REASONS, "citation reason"),
            }
            for item in row.get("bad_citations") or []
        ],
        contradicted_points=[
            {
                "point": _enum(item.get("point"), set(row.get("expected_answer_points") or []), "contradicted point"),
                "quantity": item.get("quantity") if isinstance(item.get("quantity"), str) and QUANTITY_RE.fullmatch(item["quantity"]) else _enum(None, set(), "quantity"),
                "expected_direction": _enum(item.get("expected_direction"), DIRECTIONS, "expected direction"),
                "observed_direction": _enum(item.get("observed_direction"), OBSERVED_DIRECTIONS, "observed direction"),
            }
            for item in row.get("contradicted_points") or []
        ],
    )
    return result


def public_capture(row: dict[str, Any], definitions: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """A capture trace: typed values, request linked to the case definition, target text omitted."""
    row = checked_fields(row, CAPTURE_FIELDS, "capture")
    definition = definitions.get(row.get("case_id")) if isinstance(row.get("case_id"), str) else None
    if definition is None or row.get("target") not in TARGETS:
        raise BenchmarkError("Capture row does not identify a scored case and target")
    expected_request = {"question": definition["question"], "company_ids": definition["company_ids"], "top_k": DEFAULT_TOP_K}
    if row.get("request") != expected_request:
        raise BenchmarkError(f"Capture request for {row['case_id']!r} is not the reviewed target envelope")
    if not (is_timestamp(row.get("captured_at")) and is_count(row.get("latency_ms")) and is_hex_digest(row.get("response_sha256"))):
        raise BenchmarkError(f"Invalid capture values for {row['case_id']!r}")
    if row["response_sha256"] != sha256_json(row.get("raw_response")):
        raise BenchmarkError(f"Capture response hash differs for {row['case_id']!r}")
    raw = public_raw_response(row.get("raw_response"))
    return {
        "case_id": row["case_id"],
        "target": row["target"],
        "request": expected_request,
        "captured_at": row["captured_at"],
        "latency_ms": row["latency_ms"],
        "response_sha256": row["response_sha256"],
        "raw_response": raw,
        "public_response_sha256": sha256_json(raw),
    }


def _load_json(text: str | bytes) -> Any:
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise BenchmarkError("Duplicate JSON field in capture input")
            result[key] = value
        return result

    try:
        return json.loads(text, object_pairs_hook=object_pairs)
    except (ValueError, UnicodeDecodeError) as exc:
        raise BenchmarkError("Invalid JSON capture input") from exc


def _case_definitions(originals: dict[str, Any]) -> dict[str, dict[str, Any]]:
    definitions = None
    for name, original in originals.items():
        checked_fields(original, PAYLOAD_FIELDS, "result payload")
        cases = original.get("results")
        if not isinstance(cases, list) or not cases:
            raise BenchmarkError(f"{name}: result cases must be a nonempty list")
        current = {}
        for row in cases:
            if not isinstance(row, dict):
                raise BenchmarkError(f"{name}: invalid result case")
            check_result_values(row)
            case_id = row.get("case_id")
            if not isinstance(case_id, str) or case_id in current:
                raise BenchmarkError(f"{name}: missing or duplicate case identity")
            current[case_id] = row["case_definition"]
        if definitions is not None and not _fixed(definitions)(current):
            raise BenchmarkError("Targets have different curated case definitions")
        definitions = current
    return definitions


def _crosscheck_records(config, ingestion, sources, document_map, originals, captures):
    """Check retained identities and measurements, without rescoring any response."""
    canonical = set(document_map.values())
    if any(not IDENTIFIER_RE.fullmatch(key) or not IDENTIFIER_RE.fullmatch(value) for key, value in document_map.items()):
        raise BenchmarkError("Public document map IDs must be identifiers")
    uploads = ingestion["documents"]
    if len({row["canonical_id"] for row in uploads}) != len(uploads) or {row["canonical_id"] for row in uploads} != canonical:
        raise BenchmarkError("Ingestion documents must uniquely cover the document map")
    if len({row["document_id"] for row in sources}) != len(sources) or {row["document_id"] for row in sources} != canonical:
        raise BenchmarkError("Source provenance must uniquely cover the document map")
    linked = {"endpoint": "api_base_url", "provider": "api_provider", "model": "api_model", "equity_base_sha": None, "manifest_sha256": "evidence_manifest_sha256", "document_map_sha256": "document_map_sha256"}
    for key, config_key in linked.items():
        expected = config[config_key] if config_key else config["copilot"]["head"]
        if key in ingestion and not _fixed(expected)(ingestion[key]):
            raise BenchmarkError(f"Ingestion {key} differs from capture configuration")
    if "health" in ingestion and (ingestion["health"].get("documents") != len(uploads) or ingestion["health"].get("chunks") != config["api_chunk_count"]):
        raise BenchmarkError("Ingestion health differs from recorded corpus counts")
    if "companies" in ingestion:
        if any(row.get("company_uuid") != ingestion["companies"].get(row["canonical_id"].split("_")[0].upper()) for row in uploads):
            raise BenchmarkError("Ingestion companies differ from document company IDs")
        if "health" in ingestion and ingestion["health"].get("companies") != len(ingestion["companies"]):
            raise BenchmarkError("Ingestion company health differs from recorded companies")
    records = {}
    initial_metadata = originals[TARGETS[0]]["metadata"]
    for name, original in originals.items():
        if any(not _fixed(initial_metadata[key])(original["metadata"][key]) for key in ("suite", "suite_version")):
            raise BenchmarkError("Target suite identities differ")
        for row in original["results"]:
            records[name, row["case_id"]] = row
    seen = set()
    durations = dict.fromkeys(TARGETS, 0)
    for trace in captures:
        key = trace["target"], trace["case_id"]
        if key in seen or key not in records:
            raise BenchmarkError("Capture rows must uniquely cover scored cases and targets")
        seen.add(key)
        row = records[key]
        if not _fixed(row["raw_response"])(trace.get("raw_response")) or not _fixed(row["latency_ms"])(trace["latency_ms"]):
            raise BenchmarkError("Capture response or timing differs from the scored record")
        durations[key[0]] += trace["latency_ms"]
    if seen != set(records):
        raise BenchmarkError("Capture rows do not cover every scored case and target")
    for name, duration in durations.items():
        if originals[name]["metadata"]["duration_ms"] != duration:
            raise BenchmarkError("Target duration differs from captured call timings")


def verify_capture_inventory(captured: Path) -> dict[str, str]:
    inventory_path = captured / "SHA256SUMS.json"
    if not inventory_path.exists():
        raise BenchmarkError("Captured run has no SHA256SUMS.json inventory")
    inventory = _load_json(inventory_path.read_bytes())
    expected_files = {"config.json", "sources.json", "ingestion.json", "document-map.json", "captures.jsonl"}
    expected_files.update(f"{name}/{file}" for name in TARGETS for file in ("results.json", "summary.md", "report.html", "failures.csv", "metrics.csv", "cases.jsonl", "manifest.json"))
    if not isinstance(inventory, dict) or set(inventory) != expected_files or any(not is_hex_digest(value) for value in inventory.values()):
        raise BenchmarkError("Capture inventory must contain only the reviewed workflow artifacts and SHA-256 values")
    present = {str(path.relative_to(captured)) for path in captured.rglob("*") if path.is_file()} - {"SHA256SUMS.json"}
    if present != set(inventory):
        raise BenchmarkError("Captured files differ from their inventory")
    for name, expected in inventory.items():
        if sha256_bytes((captured / name).read_bytes()) != expected:
            raise BenchmarkError(f"Captured file changed after inventory: {name}")
    return inventory


def export(captured: Path, out: Path) -> Path:
    captured = captured.resolve()
    if out.exists():
        raise BenchmarkError("Choose a new public export directory; never overwrite an export or capture")
    verify_capture_inventory(captured)
    out.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{out.name}.", dir=out.parent))
    try:
        _write_export(captured, staging)
        hits = []
        for path in staging.rglob("*"):
            if path.is_file():
                hits += [f"{path.relative_to(staging)}: {hit}" for hit in find_local_paths(path.read_text(), [str(captured)])]
        if hits:
            raise BenchmarkError("Local filesystem paths would be published:\n" + "\n".join(hits))
        staging.rename(out)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return out


def _write_export(captured: Path, out: Path) -> None:
    # Load and validate every input, with cross-file links, before writing anything.
    config_bytes = (captured / "config.json").read_bytes()
    config = public_config(_load_json(config_bytes))
    document_map = parse_document_map((captured / "document-map.json").read_text()).mapping
    if mapping_sha256(document_map) != config["document_map_sha256"]:
        raise BenchmarkError("Document map does not match the capture configuration")
    ingestion = public_ingestion(_load_json((captured / "ingestion.json").read_bytes()))
    if ingestion.get("document_uuid_to_canonical", document_map) != document_map:
        raise BenchmarkError("Ingestion upload map differs from the capture document map")
    if any(document_map.get(row.get("uuid")) != row["canonical_id"] for row in ingestion["documents"]):
        raise BenchmarkError("Ingestion document UUIDs differ from the document map")
    if ingestion.get("settings", config["api_settings"]) != config["api_settings"]:
        raise BenchmarkError("Ingestion settings differ from the capture configuration")
    if sum(row.get("chunk_count", 0) for row in ingestion["documents"]) != config["api_chunk_count"]:
        raise BenchmarkError("Configured chunk count differs from the ingestion record")
    original_sources = _load_json((captured / "sources.json").read_bytes())
    if sha256_json(original_sources) != config["source_provenance_sha256"]:
        raise BenchmarkError("Source provenance does not match the capture configuration")
    sources = public_sources(original_sources, ingestion, document_map)
    if len(sources) != config["source_documents"] or len(ingestion["documents"]) != config["source_documents"]:
        raise BenchmarkError("Configured source count differs from the source and ingestion records")
    originals = {name: _load_json((captured / name / "results.json").read_bytes()) for name in TARGETS}
    definitions = _case_definitions(originals)
    rows = [_load_json(line) for line in (captured / "captures.jsonl").read_text().splitlines() if line.strip()]
    public_rows = [public_capture(row, definitions) for row in rows]

    payloads = {}
    for name in TARGETS:
        source = captured / name / "results.json"
        original = originals[name]
        checked_fields(original, PAYLOAD_FIELDS, "result payload")
        if original["target"] != name or type(original["passed"]) is not bool:
            raise BenchmarkError(f"{name}: invalid result target or pass value")
        payload = {key: copy.deepcopy(value) for key, value in original.items() if key != "results"}
        payload["metadata"] = public_metadata(
            original["metadata"], target=name, config=config, config_sha256=sha256_bytes(config_bytes), case_count=len(original["results"])
        )
        payload["results"] = [public_result_row(row) for row in original["results"]]
        payload["summary"] = _numeric_object(original["summary"], "summary")
        if not _fixed(summarize_results(original["results"]))(payload["summary"]):
            raise BenchmarkError(f"{name}: summary differs from the recorded case metrics")
        if not isinstance(original["category_breakdown"], dict):
            raise BenchmarkError("Category breakdown must be an object")
        payload["category_breakdown"] = {key: _numeric_object(value, "category metrics") for key, value in original["category_breakdown"].items()}
        if not _fixed(_category_breakdown(original["results"]))(payload["category_breakdown"]):
            raise BenchmarkError(f"{name}: category metrics differ from the recorded cases")
        payload["gate"] = public_gate(original["gate"])
        if not _fixed(gate_summary(payload["summary"]))(payload["gate"]) or original["passed"] != payload["gate"]["passed"]:
            raise BenchmarkError(f"{name}: gate differs from the reviewed deterministic thresholds")
        payload["metadata"]["public_export"] = {
            "redacted": True,
            "original_results_sha256": sha256_bytes(source.read_bytes()),
            "note": "Scores unchanged. Target free text omitted; original or newly reproduced captures are required to rescore text.",
        }
        for key in ("summary", "gate", "category_breakdown", "passed"):
            if payload[key] != original[key]:
                raise BenchmarkError(f"{name}: public {key} differs from the measured run")
        for public, measured in zip(payload["results"], original["results"], strict=True):
            if (public["case_id"], public["passed"], public["overall_score"]) != (measured["case_id"], measured["passed"], measured["overall_score"]):
                raise BenchmarkError(f"{name}: public score differs for {measured['case_id']}")
        payloads[name] = payload

    _crosscheck_records(config, ingestion, sources, document_map, originals, rows)
    write_json(out / "config.json", config)
    write_json(out / "sources.json", sources)
    write_json(out / "ingestion.json", ingestion)
    write_json(out / "document-map.json", document_map)
    shutil.copyfile(captured / "SHA256SUMS.json", out / "ORIGINAL_SHA256SUMS.json")
    (out / "captures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in public_rows))

    for name, payload in payloads.items():
        target_dir = out / name
        target_dir.mkdir()
        write_json(target_dir / "results.json", payload)
        (target_dir / "summary.md").write_text("> Public redacted export: target text omitted; scores unchanged.\n\n" + markdown_report(payload))
        (target_dir / "report.html").write_text(html_report(payload))
        (target_dir / "failures.csv").write_text(failures_csv(payload["results"]))
        (target_dir / "metrics.csv").write_text(metrics_csv(payload))
        (target_dir / "cases.jsonl").write_text(cases_jsonl(payload["results"]))
        write_json(target_dir / "manifest.json", artifact_manifest(target_dir))

    write_json(
        out / "PUBLIC_EXPORT.json",
        {
            "policy": "Allowlisted export. Target answers, key points, limitations, debug passages, labels, titles, section titles, excerpts, error messages and unknown fields are omitted or fingerprinted.",
            "original_capture_file_sha256": sha256_bytes((captured / "captures.jsonl").read_bytes()),
            "original_inventory_sha256": sha256_bytes((captured / "SHA256SUMS.json").read_bytes()),
            "export_script_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "original_hash_file": "ORIGINAL_SHA256SUMS.json",
            "public_hash_file": "SHA256SUMS.json",
            "scores_changed": False,
            "rescore_limitation": "Public redacted records cannot independently rescore answer text. Original local or newly reproduced captures are required.",
            "retained": "Questions, curated case targets/rubrics, short source locators, citation identifiers and numeric metadata, measured metrics/diagnostics, execution/config/source provenance, original response/answer hashes.",
        },
    )
    inventory = {str(path.relative_to(out)): sha256_bytes(path.read_bytes()) for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "SHA256SUMS.json", inventory)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--captured", type=Path, required=True, help="Capture directory written by run.py")
    parser.add_argument("--out", type=Path, required=True, help="New public export directory")
    args = parser.parse_args(argv)
    export(args.captured, args.out)
    print("Created allowlisted public export at", args.out)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
