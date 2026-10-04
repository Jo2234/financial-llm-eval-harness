"""Upload complete pinned sources to a fresh Equity Copilot API through its production extraction.

All raw-source hash checks and local extraction happen before the API is touched. The
API must be empty. The recorded upload responses produce the verified
``document-map.json`` (API document UUID -> suite document ID) used for scoring. No
evidence facts, answer keys or case data are uploaded.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.common import (  # noqa: E402
    COMPANY_NAMES,
    COPILOT_MODEL,
    COPILOT_PROVIDER,
    COPILOT_SETTINGS,
    DEFAULT_API,
    DEFAULT_MANIFEST,
    DEFAULT_SUITE,
    DOCUMENT_TYPES,
    EXTRACTION_METHODS,
    BenchmarkError,
    document_ticker,
    git_revision,
    load_manifest,
    require_new_directory,
    sha256_bytes,
    write_json,
)
from fin_eval.document_map import parse_document_map, validate_document_map  # noqa: E402
from fin_eval.runner import load_suite  # noqa: E402

HtmlToText = Callable[[str], str]
PdfToText = Callable[[Path], str]


def plan_uploads(
    manifest: dict[str, Any], sources: Path, derived: Path, html_to_text: HtmlToText, pdf_to_text: PdfToText
) -> list[dict[str, Any]]:
    """Verify every raw source and prepare upload bytes before any API mutation."""
    plan = []
    for doc in manifest["documents"]:
        canonical = doc["document_id"]
        ticker = document_ticker(canonical)
        if ticker not in COMPANY_NAMES:
            raise BenchmarkError(f"{canonical}: unknown company ticker {ticker}")
        if doc["document_type"] not in DOCUMENT_TYPES:
            raise BenchmarkError(f"{canonical}: unsupported document type {doc['document_type']}")
        candidates = [sources / f"{canonical}{suffix}" for suffix in (".html", ".pdf")]
        raw_path = next((path for path in candidates if path.exists()), None)
        if raw_path is None:
            raise BenchmarkError(f"Missing raw source for {canonical} in {sources}")
        raw = raw_path.read_bytes()
        if sha256_bytes(raw) != doc["sha256"]:
            raise BenchmarkError(f"Raw hash mismatch for {canonical}; do not substitute changed sources")
        derived_path = derived / f"{canonical}.txt"
        if raw_path.suffix == ".html":
            derived_path.write_text(html_to_text(raw.decode("utf-8")))
            upload_path, mime = derived_path, "text/plain"
            method = EXTRACTION_METHODS[".html"]
        else:
            derived_path.write_text(pdf_to_text(raw_path))
            upload_path, mime = raw_path, "application/pdf"
            method = EXTRACTION_METHODS[".pdf"]
        content = upload_path.read_bytes()
        metadata = {
            "title": canonical,
            "document_type": DOCUMENT_TYPES[doc["document_type"]],
            "fiscal_year": canonical.split("_")[1],
            "source_url": doc["url"],
            "period_end_date": doc["period_end"],
        }
        if "_q1_" in canonical:
            metadata["fiscal_quarter"] = "1"
        plan.append(
            {
                "canonical_id": canonical,
                "ticker": ticker,
                "source_path": str(raw_path),
                "raw_sha256": doc["sha256"],
                "upload_path": str(upload_path),
                "upload_name": upload_path.name,
                "upload_mime": mime,
                "upload_sha256": sha256_bytes(content),
                "extraction_method": method,
                "derived_text_path": str(derived_path),
                "derived_text_sha256": sha256_bytes(derived_path.read_bytes()),
                "byte_count": len(content),
                "metadata": metadata,
            }
        )
    return plan


def _json(response: httpx.Response) -> Any:
    response.raise_for_status()
    return response.json()


def ingest(client: httpx.Client, plan: list[dict[str, Any]], report: dict[str, Any], suite_cases: list) -> dict[str, Any]:
    health = _json(client.get("/health"))
    if any(health.get(key) != 0 for key in ("companies", "documents", "chunks")):
        raise BenchmarkError("Refusing to mix with an existing corpus; start serve.py with a new --state directory")
    for ticker in sorted({row["ticker"] for row in plan}, key=list(COMPANY_NAMES).index):
        company = _json(client.post("/companies", json={"ticker": ticker, "name": COMPANY_NAMES[ticker]}))
        report["companies"][ticker] = company["id"]
    for row in plan:
        content = Path(row["upload_path"]).read_bytes()
        if sha256_bytes(content) != row["upload_sha256"]:
            raise BenchmarkError(f"Upload bytes changed after planning for {row['canonical_id']}")
        result = _json(
            client.post(
                f"/companies/{report['companies'][row['ticker']]}/documents",
                data=row["metadata"],
                files={"file": (row["upload_name"], content, row["upload_mime"])},
            )
        )
        if result.get("status") != "ready":
            raise BenchmarkError(f"{row['canonical_id']} was not ingested: status={result.get('status')!r}")
        report["document_uuid_to_canonical"][result["id"]] = row["canonical_id"]
        entry = {key: value for key, value in row.items() if key not in {"ticker", "upload_name", "upload_mime"}}
        entry.update(uuid=result["id"], company_uuid=report["companies"][row["ticker"]], chunk_count=result["chunk_count"])
        report["documents"].append(entry)
        print(row["canonical_id"], result["chunk_count"], flush=True)
    document_map = parse_document_map(json.dumps(report["document_uuid_to_canonical"]))
    validate_document_map(document_map, suite_cases, suite_cases, permitted_documents=[row["canonical_id"] for row in plan])
    report["document_map_sha256"] = document_map.sha256
    report["health"] = _json(client.get("/health"))
    if report["health"].get("documents") != len(plan):
        raise BenchmarkError(f"API reports {report['health'].get('documents')} documents after uploading {len(plan)}")
    return report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copilot", type=Path, required=True, help="ai-equity-research-copilot checkout served by serve.py")
    parser.add_argument("--sources", type=Path, required=True, help="Directory written by fetch_sources.py")
    parser.add_argument("--work", type=Path, required=True, help="New directory for ingestion evidence")
    parser.add_argument("--api", default=DEFAULT_API)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--suite", type=Path, default=DEFAULT_SUITE)
    args = parser.parse_args(argv)
    manifest_path, copilot = args.manifest.resolve(), args.copilot.resolve()
    manifest = load_manifest(manifest_path)
    suite_cases = load_suite(args.suite).cases
    work = require_new_directory(args.work).resolve()
    derived = work / "source-faithful-extraction"
    derived.mkdir()
    sys.path.insert(0, str(copilot / "backend"))
    from ai_equity_research_copilot_backend.parsing import parse_document
    from ai_equity_research_copilot_backend.sec import html_to_text

    plan = plan_uploads(
        manifest,
        args.sources.resolve(),
        derived,
        html_to_text,
        lambda path: "\n\n".join(page.text for page in parse_document(path)),
    )
    report = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "equity_base_sha": git_revision(copilot)["head"],
        "endpoint": args.api,
        "provider": COPILOT_PROVIDER,
        "model": COPILOT_MODEL,
        "seed": False,
        "settings": COPILOT_SETTINGS,
        "manifest_sha256": sha256_bytes(manifest_path.read_bytes()),
        "source_dir": str(args.sources.resolve()),
        "documents": [],
        "companies": {},
        "document_uuid_to_canonical": {},
    }
    try:
        with httpx.Client(base_url=args.api, timeout=180, trust_env=False) as client:
            ingest(client, plan, report, suite_cases)
    finally:
        write_json(work / "equity-ingestion.json", report)
    write_json(work / "document-map.json", report["document_uuid_to_canonical"])
    print(json.dumps({key: report["health"].get(key) for key in ("companies", "documents", "chunks")}))


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
