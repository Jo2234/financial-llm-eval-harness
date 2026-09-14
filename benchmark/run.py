"""Fresh Equity Copilot API and lexical-baseline captures, scored by this checkout's harness.

Targets receive only ``question``, ``company_ids`` and ``top_k``; case IDs, selected
documents, answer points, evidence facts, rubrics and reference responses stay in the
evaluator. Local sources must match the pinned manifest before any target call. API
UUID citations are canonicalized for scoring through the verified upload map; raw
responses are captured unchanged. Every output directory must be new.
"""
from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.common import (  # noqa: E402
    COMPANY_NAMES,
    COPILOT_MODEL,
    COPILOT_PROVIDER,
    DEFAULT_API,
    DEFAULT_MANIFEST,
    DEFAULT_SUITE,
    DOCUMENT_TYPES,
    EXTRACTION_METHODS,
    IDENTIFIER_RE,
    ROOT,
    BenchmarkError,
    checked_fields,
    checked_settings,
    document_ticker,
    git_revision,
    is_count,
    is_date,
    is_hex_digest,
    is_loopback_url,
    is_public_source_url,
    is_timestamp,
    load_manifest,
    require_new_directory,
    sha256_bytes,
    sha256_json,
    write_json,
)
from fin_eval.adapters import TARGET_REQUEST_FIELDS, normalize_target_response, target_request  # noqa: E402
from fin_eval.document_map import DocumentMap, parse_document_map, validate_document_map  # noqa: E402
from fin_eval.runner import (  # noqa: E402
    _case_result,
    _category_breakdown,
    artifact_manifest,
    cases_jsonl,
    failures_csv,
    gate_summary,
    html_report,
    load_suite,
    markdown_report,
    metrics_csv,
    summarize_results,
    validate_cases,
)
from fin_eval.scoring import DETERMINISTIC_CHECKS, SCORER_VERSION  # noqa: E402

STOPWORDS = set(
    "a an and are as at be by did do does for from how in is it of on or that the their this to was were what which with".split()
)
BASELINE_CONFIG = {
    "algorithm": "unique lowercase alphanumeric query-token overlap, stopwords removed",
    "chunk_characters": 1200,
    "stride_characters": 1200,
    "top_k": 3,
    "answer_characters_per_chunk": 400,
    "ties": "document ID then character offset ascending",
    "scope": "all documents for requested company tickers, no case-specific document filter",
    "refusal": "only when every chunk has zero query-token overlap",
    "model": "none",
}
TARGETS = ("copilot-local", "lexical-baseline")
SOURCE_ORIGIN = "local primary-source snapshot; raw, normalized and context hashes verified for this run"
CONFIG_TEXT = {
    "provider_selection": (
        "AIERC_LLM_PROVIDER=local set explicitly by benchmark/serve.py (deterministic extractive provider). "
        "Every API response is checked to report this provider and model; no LLM or paid API is used."
    ),
    "source_scope": "complete primary documents via production HTML/PDF extraction, not selected answer passages",
    "ordering": "suite order; sequential API call followed by baseline for each case; no retries or warmup",
    "latency": "wall time of each answer call; API includes loopback HTTP and persistence, baseline is in-process; excludes corpus ingestion and baseline indexing",
    "rubric_review": "not performed by scorer; contextual rubrics and semantic correctness need separate review",
}
PUBLIC_INGESTION_FIELDS = {
    "timestamp_utc", "equity_base_sha", "endpoint", "provider", "model", "seed", "settings", "manifest_sha256",
    "companies", "document_uuid_to_canonical", "document_map_sha256",
}
PUBLIC_INGESTION_DOCUMENT_FIELDS = {
    "canonical_id", "uuid", "company_uuid", "raw_sha256", "upload_sha256", "extraction_method",
    "derived_text_sha256", "byte_count", "metadata", "chunk_count",
}
PUBLIC_DOCUMENT_METADATA_FIELDS = {"title", "document_type", "fiscal_year", "source_url", "period_end_date", "fiscal_quarter"}
PUBLIC_HEALTH_FIELDS = {"status", "companies", "documents", "chunks"}


# Local-only ingestion fields are dropped; anything else unclassified stops the export.
LOCAL_INGESTION_FIELDS = {"source_dir"}
LOCAL_INGESTION_DOCUMENT_FIELDS = {"source_path", "upload_path", "derived_text_path"}
DATE_RE = re.compile(r"\d{4}-\d{2}-\d{2}")
TIMESTAMP_RE = re.compile(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})")


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise BenchmarkError(message)


def _public_ingestion_document(row: Any) -> dict[str, Any]:
    allowed = PUBLIC_INGESTION_DOCUMENT_FIELDS | LOCAL_INGESTION_DOCUMENT_FIELDS
    entry = {key: value for key, value in checked_fields(row, allowed, "ingestion document").items() if key in PUBLIC_INGESTION_DOCUMENT_FIELDS}
    canonical = entry.get("canonical_id")
    _require(isinstance(canonical, str) and bool(IDENTIFIER_RE.fullmatch(canonical)), "Ingestion canonical_id must be an identifier")
    for key in ("uuid", "company_uuid"):
        _require(key not in entry or (isinstance(entry[key], str) and bool(IDENTIFIER_RE.fullmatch(entry[key]))), f"{canonical}: {key} must be an identifier")
    for key in ("raw_sha256", "upload_sha256", "derived_text_sha256"):
        _require(key not in entry or is_hex_digest(entry[key]), f"{canonical}: {key} must be a SHA-256")
    for key in ("byte_count", "chunk_count"):
        _require(key not in entry or (type(entry[key]) is int and entry[key] >= 0), f"{canonical}: {key} must be a count")
    _require(entry.get("extraction_method", EXTRACTION_METHODS[".html"]) in EXTRACTION_METHODS.values(), f"{canonical}: unreviewed extraction method")
    if "metadata" in entry:
        metadata = checked_fields(entry["metadata"], PUBLIC_DOCUMENT_METADATA_FIELDS, "ingestion document metadata")
        fiscal_year = canonical.split("_")[1] if "_" in canonical else None
        checks = {
            "title": lambda value: value == canonical,
            "document_type": lambda value: value in DOCUMENT_TYPES.values(),
            "fiscal_year": lambda value: isinstance(value, str) and bool(re.fullmatch(r"[0-9]{4}", value)) and value == fiscal_year,
            "source_url": is_public_source_url,
            "period_end_date": is_date,
            "fiscal_quarter": lambda value: isinstance(value, str) and value in {"1", "2", "3", "4"},
        }
        for key, value in metadata.items():
            _require(checks[key](value), f"{canonical}: invalid public metadata {key}")
        entry["metadata"] = metadata
    return entry


def public_ingestion(ingestion: dict[str, Any]) -> dict[str, Any]:
    """Strict public ingestion record: every exported value is validated; local paths and data_dir are dropped."""
    allowed = PUBLIC_INGESTION_FIELDS | LOCAL_INGESTION_FIELDS | {"health", "documents"}
    source = checked_fields(ingestion, allowed, "ingestion")
    result = {key: source[key] for key in sorted(PUBLIC_INGESTION_FIELDS) if key in source}
    _require("timestamp_utc" not in result or is_timestamp(result["timestamp_utc"]), "Invalid ingestion timestamp")
    _require("equity_base_sha" not in result or is_hex_digest(result["equity_base_sha"], (40, 64)), "Invalid Copilot revision")
    for key in ("manifest_sha256", "document_map_sha256"):
        _require(key not in result or is_hex_digest(result[key]), f"Invalid {key}")
    _require("endpoint" not in result or is_loopback_url(result["endpoint"]), "Ingestion endpoint must be an unauthenticated loopback URL")
    _require(result.get("provider", COPILOT_PROVIDER) == COPILOT_PROVIDER and result.get("model", COPILOT_MODEL) == COPILOT_MODEL, "Unreviewed ingestion provider/model")
    _require(result.get("seed", False) is False, "Public benchmark ingestion must be unseeded")
    if "settings" in result:
        result["settings"] = checked_settings(result["settings"])
    if "companies" in result:
        companies = checked_fields(result["companies"], set(COMPANY_NAMES), "companies")
        if any(not isinstance(value, str) or not IDENTIFIER_RE.fullmatch(value) for value in companies.values()):
            raise BenchmarkError("Company IDs must be identifier strings")
        result["companies"] = companies
    if "document_uuid_to_canonical" in result:
        result["document_uuid_to_canonical"] = parse_document_map(json.dumps(result["document_uuid_to_canonical"])).mapping
        _require(all(IDENTIFIER_RE.fullmatch(key) and IDENTIFIER_RE.fullmatch(value) for key, value in result["document_uuid_to_canonical"].items()), "Upload map IDs must be identifiers")
    if "health" in source:
        health = {key: value for key, value in checked_fields(source["health"], PUBLIC_HEALTH_FIELDS | {"data_dir"}, "health").items() if key != "data_dir"}
        if any(not is_count(value) for key, value in health.items() if key != "status") or not isinstance(health.get("status"), str) or health["status"] not in {"ok", "healthy"}:
            raise BenchmarkError("Invalid public health values")
        result["health"] = health
    documents = source.get("documents", [])
    _require(isinstance(documents, list), "Ingestion documents must be a list")
    result["documents"] = [_public_ingestion_document(row) for row in documents]
    return result


def verified_sources(source_dir: Path, manifest: dict[str, Any]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Reject raw, normalized or evidence-context drift before any target can be called. Facts are not returned."""
    documents, provenance = [], []
    for doc in manifest["documents"]:
        doc_id = doc["document_id"]
        raw = next((path for path in (source_dir / f"{doc_id}.html", source_dir / f"{doc_id}.pdf") if path.exists()), None)
        if raw is None or sha256_bytes(raw.read_bytes()) != doc["sha256"]:
            raise BenchmarkError(f"Missing or changed raw source: {doc_id}")
        text_path = source_dir / f"{doc_id}.txt"
        if not text_path.exists():
            raise BenchmarkError(f"Missing normalized source text: {doc_id}")
        text = text_path.read_text()
        evidence = [row for row in manifest["evidence"] if row["document_id"] == doc_id]
        if not evidence:
            raise BenchmarkError(f"No source provenance for {doc_id}")
        for row in evidence:
            if sha256_bytes(text.encode()) != row["normalized_text_sha256"]:
                raise BenchmarkError(f"Changed normalized source: {doc_id}")
            start = row["context_start_character"]
            if sha256_bytes(text[start:start + 1200].encode()) != row["context_sha256"]:
                raise BenchmarkError(f"Changed pinned context: {row['evidence_id']}")
        documents.append({"document_id": doc_id, "company": document_ticker(doc_id)})
        provenance.append(
            {
                **doc,
                "normalized_text_sha256": sha256_bytes(text.encode()),
                "normalized_characters": len(text),
                "verified_contexts": len(evidence),
                "origin": SOURCE_ORIGIN,
            }
        )
    return documents, provenance


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower())) - STOPWORDS


class LexicalBaseline:
    """Fixed extractive baseline: no model, synthesis, period filter or financial refusal policy."""

    def __init__(self, documents: list[dict[str, Any]]):
        self.chunks = []
        for doc in sorted(documents, key=lambda row: row["document_id"]):
            for offset in range(0, len(doc["text"]), BASELINE_CONFIG["stride_characters"]):
                text = doc["text"][offset:offset + BASELINE_CONFIG["chunk_characters"]]
                self.chunks.append((doc["company"], doc["document_id"], offset, text, tokens(text)))

    def answer(self, question: str, company_ids: list[str]) -> dict[str, Any]:
        query = tokens(question)
        scope = {company.upper() for company in company_ids}
        ranked = [(len(query & terms), doc, offset, text) for company, doc, offset, text, terms in self.chunks if company in scope]
        ranked.sort(key=lambda row: (-row[0], row[1], row[2]))
        selected = [row for row in ranked[: BASELINE_CONFIG["top_k"]] if row[0] > 0]
        width = BASELINE_CONFIG["answer_characters_per_chunk"]
        return {
            "answer": "\n\n".join(row[3][:width] for row in selected) if selected else "I do not have enough cited context to answer that.",
            "citations": [
                {"document_id": doc, "chunk_id": f"{doc}:{offset}", "excerpt": text[:width], "label": f"{doc}, character {offset}", "overlap": overlap}
                for overlap, doc, offset, text in selected
            ],
            "model": "lexical-overlap-extractive-v1-no-llm",
            "usage": {"provider": "local-baseline", "model": "none", "estimated_cost_usd": 0.0},
        }


def load_ingestion(path: Path, manifest: dict[str, Any], provenance: list[dict[str, Any]], documents: list[dict[str, Any]], suite_cases: list) -> tuple[dict[str, Any], DocumentMap]:
    """Strictly validate the upload map and production-extracted text against the verified corpus."""
    # Validate the original JSON, including duplicate keys, before an ordinary
    # parse can silently discard an ambiguous UUID assignment.
    text = path.read_text()
    document_map = parse_document_map(text, source="ingestion")
    ingestion = json.loads(text)
    corpus = [doc["document_id"] for doc in manifest["documents"]]
    validate_document_map(document_map, suite_cases, suite_cases, permitted_documents=corpus)
    if sorted(document_map.mapping.values()) != sorted(corpus):
        raise BenchmarkError("Document upload map must cover exactly the verified source corpus")
    derived = {row["canonical_id"]: row for row in ingestion.get("documents", [])}
    for doc in documents:
        row = derived.get(doc["document_id"])
        original = next(item for item in provenance if item["document_id"] == doc["document_id"])
        if row is None or row["raw_sha256"] != original["sha256"]:
            raise BenchmarkError(f"Ingested source differs from pinned raw source: {doc['document_id']}")
        if document_map.mapping.get(row["uuid"]) != doc["document_id"]:
            raise BenchmarkError(f"Ingestion record and upload map disagree for {doc['document_id']}")
        text = Path(row["derived_text_path"]).read_text()
        if sha256_bytes(text.encode()) != row["derived_text_sha256"]:
            raise BenchmarkError(f"Baseline text differs from production extraction: {doc['document_id']}")
        doc["text"] = text
    return ingestion, document_map


def capture(client: httpx.Client, api: str, suite_cases: list, baseline: LexicalBaseline, document_map: DocumentMap, out: Path) -> tuple[list[dict[str, Any]], dict[str, list], dict[str, int]]:
    """Call API then baseline once per case in suite order; no warmup or retries."""
    traces: list[dict[str, Any]] = []
    scored: dict[str, list] = {name: [] for name in TARGETS}
    durations = dict.fromkeys(TARGETS, 0)
    for case in suite_cases:
        for name in TARGETS:
            request = target_request(case)
            started = time.perf_counter()
            try:
                if name == "copilot-local":
                    response = client.post(api.rstrip("/") + "/research/chat", json=request)
                    response.raise_for_status()
                    raw = response.json()
                else:
                    raw = baseline.answer(request["question"], request["company_ids"])
            except Exception as exc:  # captured as an execution error, not behavior
                raw = {"error": f"{type(exc).__name__}: {exc}"}
            latency = int((time.perf_counter() - started) * 1000)
            if name == "copilot-local" and "error" not in raw:
                usage = raw.get("usage") if isinstance(raw.get("usage"), dict) else {}
                if usage.get("provider") != COPILOT_PROVIDER or usage.get("model") != COPILOT_MODEL:
                    raise BenchmarkError(f"{case.id}: Copilot response does not report the declared local provider/model; run aborted")
            durations[name] += latency
            traces.append(
                {
                    "case_id": case.id,
                    "target": name,
                    "request": request,
                    "captured_at": datetime.now(timezone.utc).isoformat(),
                    "latency_ms": latency,
                    "raw_response": raw,
                    "response_sha256": sha256_json(raw),
                }
            )
            response_model = normalize_target_response(raw, latency)
            scored[name].append(_case_result(case, response_model, document_map if name == "copilot-local" else None))
        (out / "captures.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in traces))
        print(case.id, flush=True)
    return traces, scored, durations


def write_target_reports(out: Path, scored: dict[str, list], durations: dict[str, int], config: dict[str, Any], suite_version: str | None, suite_name: str = "evals/core.yaml") -> None:
    for name, details in scored.items():
        target_dir = out / name
        target_dir.mkdir()
        summary = summarize_results(details)
        gate = gate_summary(summary)
        payload = {
            "metadata": {
                "suite": suite_name,
                "suite_version": suite_version,
                "suite_kind": "quality",
                "target": name,
                "case_count": len(details),
                "evaluation_mode": "fresh execution; deterministic no-LLM comparison",
                "started_at": config["capture_started_at"],
                "duration_ms": durations[name],
                "scorer_version": SCORER_VERSION,
                "deterministic_checks": DETERMINISTIC_CHECKS,
                "rubric_evaluation": "not_performed",
                "semantic_review": "not_performed",
                "rubric_review_required_cases": len(details),
                "document_map": {"mapping_sha256": config["document_map_sha256"]} if name == "copilot-local" else None,
                "configuration_sha256": sha256_bytes((out / "config.json").read_bytes()),
            },
            "summary": summary,
            "category_breakdown": _category_breakdown(details),
            "passed": gate["passed"],
            "gate": gate,
            "results": details,
            "target": name,
        }
        write_json(target_dir / "results.json", payload)
        (target_dir / "summary.md").write_text(markdown_report(payload))
        (target_dir / "report.html").write_text(html_report(payload))
        (target_dir / "failures.csv").write_text(failures_csv(details))
        (target_dir / "metrics.csv").write_text(metrics_csv(payload))
        (target_dir / "cases.jsonl").write_text(cases_jsonl(details))
        write_json(target_dir / "manifest.json", artifact_manifest(target_dir))
        print(name, json.dumps(summary), flush=True)


def write_inventory(out: Path) -> None:
    inventory = {str(path.relative_to(out)): sha256_bytes(path.read_bytes()) for path in sorted(out.rglob("*")) if path.is_file()}
    write_json(out / "SHA256SUMS.json", inventory)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copilot", type=Path, required=True, help="Copilot checkout being served (for revision provenance)")
    parser.add_argument("--sources", type=Path, required=True, help="Directory written by fetch_sources.py")
    parser.add_argument("--ingestion", type=Path, required=True, help="equity-ingestion.json written by ingest.py")
    parser.add_argument("--out", type=Path, required=True, help="New capture directory")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise BenchmarkError("Output already exists; choose a new capture directory rather than overwriting a run")
    manifest = load_manifest(args.manifest)
    suite = load_suite(args.suite)
    validate_cases(suite.cases, quality=True)
    documents, provenance = verified_sources(args.sources, manifest)
    ingestion, document_map = load_ingestion(args.ingestion, manifest, provenance, documents, suite.cases)
    baseline = LexicalBaseline(documents)
    copilot = args.copilot.resolve()
    with httpx.Client(timeout=90, trust_env=False) as client:
        health = client.get(args.api.rstrip("/") + "/health").json()
        expected_chunks = sum(row["chunk_count"] for row in ingestion["documents"])
        if health.get("documents") != len(documents) or health.get("chunks") != expected_chunks:
            raise BenchmarkError("API corpus differs from the recorded ingestion; serve and ingest into a fresh state")
        config = {
            "capture_started_at": datetime.now(timezone.utc).isoformat(),
            "suite_sha256": sha256_bytes(args.suite.read_bytes()),
            "evidence_manifest_sha256": sha256_bytes(args.manifest.read_bytes()),
            "runner_sha256": sha256_bytes(Path(__file__).read_bytes()),
            "harness": git_revision(ROOT),
            "copilot": git_revision(copilot),
            "copilot_backend_tree": subprocess.check_output(["git", "-C", str(copilot), "rev-parse", "HEAD:backend"], text=True).strip(),
            "scorer_version": SCORER_VERSION,
            "python": platform.python_version(),
            "api_base_url": args.api,
            "api_provider": COPILOT_PROVIDER,
            "api_model": COPILOT_MODEL,
            "api_request_fields": list(TARGET_REQUEST_FIELDS),
            "api_top_k": 8,
            "baseline": BASELINE_CONFIG,
            "source_documents": len(documents),
            "source_retrieved_on": manifest["retrieved_on"],
            "source_provenance_sha256": sha256_json(provenance),
            "document_map_sha256": document_map.sha256,
            "ingestion_manifest_sha256": sha256_bytes(args.ingestion.read_bytes()),
            "api_settings": ingestion["settings"],
            "api_chunk_count": expected_chunks,
            **CONFIG_TEXT,
        }
        out = require_new_directory(args.out)
        write_json(out / "config.json", config)
        write_json(out / "sources.json", provenance)
        write_json(out / "document-map.json", document_map.mapping)
        write_json(out / "ingestion.json", public_ingestion(ingestion))
        _, scored, durations = capture(client, args.api, suite.cases, baseline, document_map, out)
    suite_name = "evals/core.yaml" if args.suite.resolve() == DEFAULT_SUITE.resolve() else args.suite.name
    write_target_reports(out, scored, durations, config, suite.schema_version, suite_name)
    write_inventory(out)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
