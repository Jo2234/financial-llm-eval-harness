"""Offline integration tests for the public benchmark workflow (no network, no real target).

A synthetic two-document corpus and a mocked Copilot API exercise source verification,
ingestion, capture, strict UUID mapping, request isolation and the allowlisted export.
"""
from __future__ import annotations

import ast
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import httpx
import pytest
import yaml

from benchmark import export_public, fetch_sources, ingest, run, serve
from benchmark.common import COPILOT_SETTINGS, EXTRACTION_METHODS, BenchmarkError, is_public_source_url, sha256_bytes
from fin_eval.document_map import parse_document_map, validate_document_map
from fin_eval.runner import load_suite

ROOT = Path(__file__).resolve().parents[1]
NVDA_UUID = "9e495e50-c08e-4b78-9eaa-6d65df3a0cdb"
AAPL_UUID = "9382c24c-ec5f-4cba-b3d8-6f81ddf33093"
UNKNOWN_UUID = "00000000-0000-4000-8000-000000000000"
SENTINEL = "SENTINEL"
LOCAL = "/Users/someone/private-state"
POINT = "NVIDIA Data Center sales rose 142%."
TEXTS = {
    "nvda_2025_10k": "SENTINEL-SOURCE synthetic NVIDIA filing. Data Center sales rose 142% in this fixture. " * 30,
    "aapl_2025_10k": "SENTINEL-SOURCE synthetic Apple filing. Services sales grew in this fixture. " * 30,
}


def build_corpus(tmp_path: Path) -> dict:
    sources = tmp_path / "sources"
    sources.mkdir()
    documents, evidence = [], []
    for doc_id, text in TEXTS.items():
        raw = f"<html><body><p>{text}</p></body></html>".encode()
        (sources / f"{doc_id}.html").write_bytes(raw)
        (sources / f"{doc_id}.txt").write_text(text)
        documents.append(
            {"document_id": doc_id, "document_type": "10-K", "url": f"https://example.com/{doc_id}.htm",
             "period_end": "2025-01-26", "sha256": sha256_bytes(raw)}
        )
        evidence.append(
            {"evidence_id": f"{doc_id}_ctx", "document_id": doc_id, "normalized_text_sha256": sha256_bytes(text.encode()),
             "context_start_character": 0, "context_sha256": sha256_bytes(text[:1200].encode())}
        )
    manifest = tmp_path / "manifest.yaml"
    manifest.write_text(yaml.safe_dump({"retrieved_on": "2026-09-06", "documents": documents, "evidence": evidence}))
    evidence_rows = {doc: {"document_id": doc, "url": f"https://example.com/{doc}.htm", "excerpt": "locator"} for doc in TEXTS}
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "schema_version": "financial-eval-suite/v2",
                "suite_kind": "quality",
                "cases": [
                    {"id": "nvda_case_001", "category": "factual_extraction", "question": "How did NVIDIA Data Center sales change?",
                     "company_ids": ["NVDA"], "documents": ["nvda_2025_10k"], "expected_answer_points": [POINT],
                     "required_citation_rules": [{"document_id": "nvda_2025_10k"}], "judge_rubric": "Rubric secret",
                     "source_evidence": [evidence_rows["nvda_2025_10k"]]},
                    {"id": "aapl_refusal_002", "category": "refusal", "question": "What is Apple's private target?",
                     "company_ids": ["AAPL"], "documents": ["aapl_2025_10k"], "refusal_expected": True,
                     "judge_rubric": "Refuse", "source_evidence": [evidence_rows["aapl_2025_10k"]]},
                ],
            }
        )
    )
    derived = tmp_path / "derived"
    derived.mkdir()
    rows = []
    for doc_id, uuid in (("nvda_2025_10k", NVDA_UUID), ("aapl_2025_10k", AAPL_UUID)):
        path = derived / f"{doc_id}.txt"
        path.write_text(TEXTS[doc_id])
        rows.append(
            {"canonical_id": doc_id, "uuid": uuid, "company_uuid": "c-" + doc_id, "raw_sha256": documents[len(rows)]["sha256"],
             "source_path": str(sources / f"{doc_id}.html"), "upload_path": str(path), "upload_sha256": sha256_bytes(path.read_bytes()),
             "extraction_method": EXTRACTION_METHODS[".html"], "derived_text_path": str(path), "derived_text_sha256": sha256_bytes(path.read_bytes()),
             "byte_count": 1, "metadata": {"title": doc_id, "document_type": "10-k", "fiscal_year": "2025", "source_url": f"https://example.com/{doc_id}.htm", "period_end_date": "2025-01-26"}, "chunk_count": 3}
        )
    ingestion = tmp_path / "equity-ingestion.json"
    ingestion.write_text(
        json.dumps(
            {"timestamp_utc": "2026-10-04T00:00:00+00:00", "equity_base_sha": "a" * 40, "provider": "local", "settings": COPILOT_SETTINGS,
             "source_dir": str(sources), "documents": rows, "companies": {},
             "health": {"status": "ok", "data_dir": LOCAL, "companies": 2, "documents": 2, "chunks": 6},
             "document_uuid_to_canonical": {NVDA_UUID: "nvda_2025_10k", AAPL_UUID: "aapl_2025_10k"}}
        )
    )
    copilot = tmp_path / "copilot"
    (copilot / "backend").mkdir(parents=True)
    (copilot / "backend" / "app.py").write_text("")
    for args in (["init", "-q"], ["add", "."], ["-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init"]):
        subprocess.run(["git", "-C", str(copilot), *args], check=True)
    record = json.loads(ingestion.read_text())
    record["equity_base_sha"] = subprocess.check_output(["git", "-C", str(copilot), "rev-parse", "HEAD"], text=True).strip()
    record["companies"] = {doc_id.split("_")[0].upper(): "c-" + doc_id for doc_id in TEXTS}
    ingestion.write_text(json.dumps(record))
    return {"sources": sources, "manifest": manifest, "suite": suite, "ingestion": ingestion, "copilot": copilot}


def sentinel_api(requests: list) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "data_dir": LOCAL, "companies": 2, "documents": 2, "chunks": 6})
        body = json.loads(request.content)
        requests.append(body)
        if body["company_ids"] == ["AAPL"]:
            raise RuntimeError(f"{LOCAL}/SENTINEL-ERROR could not open database")
        citation = {"label": "SENTINEL-LABEL", "title": "SENTINEL-TITLE", "section_title": "SENTINEL-SECTION",
                    "excerpt": "SENTINEL-EXCERPT", "nested": {"deep": "SENTINEL-NESTED"}, "document_id": NVDA_UUID,
                    "chunk_id": "39e95b18-42fa-4167-b9d6-bf2e2fe347c4", "score": 0.9, "company_ticker": "NVDA", "page_start": 1}
        unmapped = {**citation, "document_id": UNKNOWN_UUID, "label": "nvda_2025_10k", "excerpt": "nvda_2025_10k"}
        return httpx.Response(
            200,
            json={"answer": "SENTINEL-ANSWER NVIDIA Data Center sales rose 142%.", "key_points": ["SENTINEL-KEYPOINT"],
                  "limitations": [f"SENTINEL-LIMITATION {LOCAL}"], "citations": [citation, unmapped], "confidence": "medium",
                  "retrieval_debug": {"query": "SENTINEL-DEBUG", "chunks": [{"text": "SENTINEL-PASSAGE"}]},
                  "message": "SENTINEL-MESSAGE", "commentary": {"x": "SENTINEL-UNKNOWN"}, "model": "SENTINEL model prose",
                  "message_id": "m-1", "conversation_id": "c-1",
                  "usage": {"provider": "local", "model": "local-deterministic-grounded-v1", "latency_ms": 5,
                            "input_tokens": 1, "output_tokens": 2, "estimated_cost_usd": 0.0}},
        )

    return httpx.MockTransport(handler)


@pytest.fixture
def captured(tmp_path, monkeypatch):
    corpus = build_corpus(tmp_path)
    requests: list = []
    transport = sentinel_api(requests)
    real_client = httpx.Client
    monkeypatch.setattr(run.httpx, "Client", lambda **kwargs: real_client(transport=transport, **kwargs))
    out = tmp_path / "capture"
    run.main(["--copilot", str(corpus["copilot"]), "--sources", str(corpus["sources"]), "--ingestion", str(corpus["ingestion"]),
              "--out", str(out), "--suite", str(corpus["suite"]), "--manifest", str(corpus["manifest"])])
    return {**corpus, "out": out, "requests": requests, "tmp": tmp_path}


def all_text(directory: Path) -> str:
    return "\n".join(path.read_text() for path in sorted(directory.rglob("*")) if path.is_file())


def test_capture_sends_only_question_scope_and_top_k(captured):
    suite = load_suite(captured["suite"])
    assert [sorted(body) for body in captured["requests"]] == [["company_ids", "question", "top_k"]] * 2
    for body, case in zip(captured["requests"], suite.cases, strict=True):
        assert body == {"question": case.question, "company_ids": case.company_ids, "top_k": 8}
        for secret in (case.id, *case.documents, *case.expected_answer_points, case.judge_rubric):
            assert secret not in json.dumps(body)


def test_capture_preserves_raw_uuids_and_scores_through_strict_map(captured):
    out = captured["out"]
    rows = [json.loads(line) for line in (out / "captures.jsonl").read_text().splitlines()]
    assert [(row["case_id"], row["target"]) for row in rows] == [
        ("nvda_case_001", "copilot-local"), ("nvda_case_001", "lexical-baseline"),
        ("aapl_refusal_002", "copilot-local"), ("aapl_refusal_002", "lexical-baseline"),
    ]
    assert rows[0]["raw_response"]["citations"][0]["document_id"] == NVDA_UUID
    results = json.loads((out / "copilot-local" / "results.json").read_text())
    first, refusal = results["results"]
    assert first["raw_response"]["citations"][1]["document_id"] == UNKNOWN_UUID
    assert [c["document_id"] for c in first["citations"]] == ["nvda_2025_10k", UNKNOWN_UUID]
    assert [c["document_map_status"] for c in first["citations"]] == ["mapped", "unmapped"]
    assert first["bad_citations"][0]["reason"] == "document_id is not in the verified document map"
    assert first["citation_precision"] == 0.5
    assert refusal["execution_status"] == "error" and refusal["error"].startswith("RuntimeError")
    assert results["metadata"]["scorer_version"] == "financial-eval-scorer/v3"
    config = json.loads((out / "config.json").read_text())
    assert config["api_request_fields"] == ["question", "company_ids", "top_k"]
    assert config["document_map_sha256"] == parse_document_map((out / "document-map.json").read_text()).sha256
    assert "Ollama" not in json.dumps(config)
    assert LOCAL not in (out / "ingestion.json").read_text() and "source_dir" not in (out / "ingestion.json").read_text()
    inventory = json.loads((out / "SHA256SUMS.json").read_text())
    assert inventory["captures.jsonl"] == sha256_bytes((out / "captures.jsonl").read_bytes())


def test_public_export_removes_all_target_prose_and_paths_without_changing_scores(captured):
    out, public = captured["out"], captured["tmp"] / "public"
    assert SENTINEL in all_text(out)  # the capture really contains every sentinel field
    export_public.export(out, public)
    text = all_text(public)
    assert SENTINEL not in text
    assert "/Users/" not in text and str(captured["tmp"]) not in text
    for name in run.TARGETS:
        original = json.loads((out / name / "results.json").read_text())
        exported = json.loads((public / name / "results.json").read_text())
        for key in ("summary", "gate", "category_breakdown", "passed"):
            assert exported[key] == original[key]
        assert [(r["passed"], r["overall_score"]) for r in exported["results"]] == [
            (r["passed"], r["overall_score"]) for r in original["results"]
        ]
        assert exported["results"][0]["answer_sha256"] == sha256_bytes(original["results"][0]["answer"].encode())
    copilot = json.loads((public / "copilot-local" / "results.json").read_text())["results"]
    assert copilot[1]["error"] == "RuntimeError"
    assert copilot[0]["raw_response"]["citations"][0]["document_id"] == NVDA_UUID
    assert copilot[0]["model"] == "[redacted]"
    assert json.loads((public / "ORIGINAL_SHA256SUMS.json").read_text()) == json.loads((out / "SHA256SUMS.json").read_text())
    policy = json.loads((public / "PUBLIC_EXPORT.json").read_text())
    assert policy["scores_changed"] is False and "cannot independently rescore" in policy["rescore_limitation"]
    public_inventory = json.loads((public / "SHA256SUMS.json").read_text())
    assert all(sha256_bytes((public / name).read_bytes()) == digest for name, digest in public_inventory.items())
    with pytest.raises(BenchmarkError, match="new public export directory"):
        export_public.export(out, public)


def rewrite_inventory(directory: Path) -> None:
    inventory = {str(p.relative_to(directory)): sha256_bytes(p.read_bytes())
                 for p in sorted(directory.rglob("*")) if p.is_file() and p.name != "SHA256SUMS.json"}
    (directory / "SHA256SUMS.json").write_text(json.dumps(inventory))


def test_export_fails_closed_on_tampering_paths_and_unclassified_fields(captured):
    out, tmp = captured["out"], captured["tmp"]
    tampered = tmp / "tampered"
    shutil.copytree(out, tampered)
    (tampered / "copilot-local" / "summary.md").write_text("changed")
    with pytest.raises(BenchmarkError, match="changed after inventory"):
        export_public.export(tampered, tmp / "public-tampered")

    leaky = tmp / "leaky"
    shutil.copytree(out, leaky)
    ingestion = json.loads((leaky / "ingestion.json").read_text())
    ingestion["documents"][0]["metadata"]["title"] = "/Users/someone/filings/nvda.htm"
    (leaky / "ingestion.json").write_text(json.dumps(ingestion))
    rewrite_inventory(leaky)
    with pytest.raises(BenchmarkError, match="invalid public metadata title"):
        export_public.export(leaky, tmp / "public-leaky")

    origin = tmp / "origin"
    shutil.copytree(out, origin)
    sources = json.loads((origin / "sources.json").read_text())
    sources[0]["origin"] = "/srv/private/corpus snapshot"
    (origin / "sources.json").write_text(json.dumps(sources))
    rewrite_inventory(origin)
    with pytest.raises(BenchmarkError, match="Source provenance does not match"):
        export_public.export(origin, tmp / "public-origin")

    # Curated gold text is the documented trust boundary; the path scan is its backstop.
    scanned = tmp / "scanned"
    shutil.copytree(out, scanned)
    for name in run.TARGETS:
        path = scanned / name / "results.json"
        results = json.loads(path.read_text())
        for row in results["results"]:
            row["case_definition"]["notes"] = "Curated from /srv/private/corpus snapshot"
            row["case_fingerprint"] = export_public._case_fingerprint(row["case_definition"])
        path.write_text(json.dumps(results))
    rewrite_inventory(scanned)
    with pytest.raises(BenchmarkError, match="Local filesystem paths"):
        export_public.export(scanned, tmp / "public-scanned")

    unknown = tmp / "unknown"
    shutil.copytree(out, unknown)
    results = json.loads((unknown / "copilot-local" / "results.json").read_text())
    results["results"][0]["new_scorer_field"] = "target text?"
    (unknown / "copilot-local" / "results.json").write_text(json.dumps(results))
    rewrite_inventory(unknown)
    with pytest.raises(BenchmarkError, match="Unclassified result fields"):
        export_public.export(unknown, tmp / "public-unknown")
    assert not any(path.name.startswith(".public") for path in tmp.iterdir())  # staging removed


def test_scorer_result_fields_are_all_classified_for_export(captured):
    row = json.loads((captured["out"] / "copilot-local" / "results.json").read_text())["results"][0]
    assert set(row) <= export_public.EVALUATOR_RESULT_FIELDS | export_public.TARGET_RESULT_FIELDS


def test_run_rejects_existing_output_and_source_drift(tmp_path):
    corpus = build_corpus(tmp_path)
    base = ["--copilot", str(corpus["copilot"]), "--sources", str(corpus["sources"]), "--ingestion", str(corpus["ingestion"]),
            "--suite", str(corpus["suite"]), "--manifest", str(corpus["manifest"])]
    existing = tmp_path / "existing"
    existing.mkdir()
    with pytest.raises(BenchmarkError, match="Output already exists"):
        run.main([*base, "--out", str(existing)])
    (corpus["sources"] / "nvda_2025_10k.txt").write_text(TEXTS["nvda_2025_10k"] + " changed")
    with pytest.raises(BenchmarkError, match="Changed normalized source"):
        run.main([*base, "--out", str(tmp_path / "new")])
    assert not (tmp_path / "new").exists()


def test_ingest_verifies_everything_before_mutating_an_empty_api(tmp_path):
    corpus = build_corpus(tmp_path)
    manifest = yaml.safe_load(corpus["manifest"].read_text())
    derived = tmp_path / "plan"
    derived.mkdir()
    plan = ingest.plan_uploads(manifest, corpus["sources"], derived, lambda html: html, lambda path: "pdf")
    assert [row["canonical_id"] for row in plan] == ["nvda_2025_10k", "aapl_2025_10k"]
    broken = json.loads(json.dumps(manifest))
    broken["documents"][0]["sha256"] = "0" * 64
    with pytest.raises(BenchmarkError, match="Raw hash mismatch"):
        ingest.plan_uploads(broken, corpus["sources"], derived, lambda html: html, lambda path: "pdf")

    calls: list = []
    counts = {"companies": 0, "documents": 0, "chunks": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append((request.method, request.url.path))
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok", "data_dir": LOCAL, **counts})
        if request.url.path == "/companies":
            counts["companies"] += 1
            return httpx.Response(201, json={"id": f"company-{counts['companies']}"})
        counts["documents"] += 1
        counts["chunks"] += 3
        uuid = NVDA_UUID if counts["documents"] == 1 else AAPL_UUID
        return httpx.Response(201, json={"id": uuid, "status": "ready", "chunk_count": 3})

    suite_cases = load_suite(corpus["suite"]).cases
    report = {"companies": {}, "documents": [], "document_uuid_to_canonical": {}}
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as client:
        ingest.ingest(client, plan, report, suite_cases)
    assert report["document_uuid_to_canonical"] == {NVDA_UUID: "nvda_2025_10k", AAPL_UUID: "aapl_2025_10k"}
    assert report["health"]["documents"] == 2 and len(report["document_map_sha256"]) == 64

    calls.clear()
    with httpx.Client(base_url="http://api", transport=httpx.MockTransport(handler)) as client:
        with pytest.raises(BenchmarkError, match="existing corpus"):
            ingest.ingest(client, plan, {"companies": {}, "documents": [], "document_uuid_to_canonical": {}}, suite_cases)
    assert calls == [("GET", "/health")]


def test_fetch_requires_fresh_directory_and_rejects_raw_drift(tmp_path):
    corpus = build_corpus(tmp_path)
    manifest = yaml.safe_load(corpus["manifest"].read_text())
    transport = httpx.MockTransport(lambda request: httpx.Response(200, content=b"<html>changed</html>"))
    with httpx.Client(transport=transport) as client:
        with pytest.raises(BenchmarkError, match="Raw source drift for nvda_2025_10k"):
            fetch_sources.fetch_sources(manifest, tmp_path / "fetched", client, delay_s=0)
        with pytest.raises(BenchmarkError, match="already exists"):
            fetch_sources.fetch_sources(manifest, corpus["sources"], client, delay_s=0)
    assert not (tmp_path / "fetched" / "nvda_2025_10k.html").exists()
    with pytest.raises(SystemExit):
        fetch_sources.main(["--out", str(tmp_path / "ua"), "--user-agent", "python-httpx"])


def test_fetch_writes_verified_sources_with_pinned_normalization(tmp_path):
    pytest.importorskip("bs4")
    from benchmark.common import normalize_html

    raw = b"<html><head><style>x{}</style><script>var a;</script></head><body><p>Data   Center</p><p>sales rose 142%.</p></body></html>"
    text = normalize_html(raw)
    assert text == "Data Center sales rose 142%."
    manifest = {
        "retrieved_on": "2026-09-06",
        "documents": [{"document_id": "nvda_2025_10k", "url": "https://example.com/n.htm", "sha256": sha256_bytes(raw)}],
        "evidence": [{"document_id": "nvda_2025_10k", "normalized_text_sha256": sha256_bytes(text.encode())}],
    }
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw))) as client:
        fetch_sources.fetch_sources(manifest, tmp_path / "fetched", client, delay_s=0)
    assert (tmp_path / "fetched" / "nvda_2025_10k.html").read_bytes() == raw
    assert (tmp_path / "fetched" / "nvda_2025_10k.txt").read_text() == text
    record = json.loads((tmp_path / "fetched" / "fetch-record.json").read_text())
    assert record["libraries"]["beautifulsoup4"]


def test_serve_environment_is_explicitly_local_and_unseeded(tmp_path):
    env = serve.copilot_environment(tmp_path)
    assert env["AIERC_LLM_PROVIDER"] == "local"
    assert env["AIERC_DEMO_MODE"] == "false"
    assert env["AIERC_DATA_DIR"].startswith(str(tmp_path))
    assert "OPENAI_API_KEY" in serve.PROVIDER_KEY_VARIABLES


def test_core_manifest_supports_strict_ingestion_and_mapping():
    manifest = yaml.safe_load((ROOT / "evals/evidence/manifest.yaml").read_text())
    assert all(is_public_source_url(doc["url"]) for doc in manifest["documents"])
    corpus = [doc["document_id"] for doc in manifest["documents"]]
    for doc in corpus:
        assert len({row["normalized_text_sha256"] for row in manifest["evidence"] if row["document_id"] == doc}) == 1
    cases = load_suite(ROOT / "evals/core.yaml").cases
    mapping = {f"uuid-{index}": doc for index, doc in enumerate(corpus)}
    provenance = validate_document_map(parse_document_map(json.dumps(mapping)), cases, cases, permitted_documents=corpus)
    assert provenance["entries"] == 16 and provenance["unmapped_suite_documents"] == []


def test_benchmark_validation_does_not_rely_on_assert_statements():
    for path in (ROOT / "benchmark").glob("*.py"):
        tree = ast.parse(path.read_text())
        assert not any(isinstance(node, ast.Assert) for node in ast.walk(tree)), f"{path.name} uses assert; python -O would skip it"


def test_public_docs_do_not_reference_the_private_report_repository():
    for path in [ROOT / "README.md", ROOT / "CHANGELOG.md", *ROOT.glob("benchmark/*"), *ROOT.glob("examples/*"), *ROOT.glob("evals/**/*.md")]:
        if path.is_file():
            assert "eval-harness-report" not in path.read_text(), path


def test_ingestion_duplicate_keys_fail_before_any_target_access(tmp_path, monkeypatch):
    corpus = build_corpus(tmp_path)
    original = corpus['ingestion'].read_text()
    original = original.replace(f'"{NVDA_UUID}": "nvda_2025_10k"', f'"{NVDA_UUID}": "bogus", "{NVDA_UUID}": "nvda_2025_10k"')
    corpus['ingestion'].write_text(original)
    calls = []
    monkeypatch.setattr(run.httpx, 'Client', lambda **kwargs: calls.append(kwargs))
    from fin_eval.document_map import DocumentMapError
    with pytest.raises(DocumentMapError, match='Duplicate key'):
        run.main(['--copilot', str(corpus['copilot']), '--sources', str(corpus['sources']), '--ingestion', str(corpus['ingestion']),
                  '--suite', str(corpus['suite']), '--manifest', str(corpus['manifest']), '--out', str(tmp_path / 'duplicate-out')])
    assert not calls and not (tmp_path / 'duplicate-out').exists()


@pytest.mark.parametrize('file,container', [
    ('config.json', 'harness'), ('config.json', 'copilot'), ('config.json', 'baseline'), ('config.json', 'api_settings'),
    ('ingestion.json', 'settings'), ('copilot-local/results.json', 'metadata'),
])
def test_export_rejects_unknown_nested_prose_and_nonstandard_paths(captured, file, container):
    out = captured['out']
    path = out / file
    data = json.loads(path.read_text())
    data[container]['private_note'] = 'SENTINEL-PROSE /Volumes/confidential/corpus'
    path.write_text(json.dumps(data))
    rewrite_inventory(out)
    public = captured['tmp'] / 'nested-public'
    with pytest.raises(BenchmarkError, match='Unclassified|Invalid config values'):
        export_public.export(out, public)
    assert not public.exists()


def test_export_rejects_unclassified_payload_top_level(captured):
    out = captured['out']
    path = out / 'copilot-local/results.json'
    data = json.loads(path.read_text())
    data['private_note'] = 'SENTINEL-PROSE'
    path.write_text(json.dumps(data))
    rewrite_inventory(out)
    with pytest.raises(BenchmarkError, match='Unclassified result payload'):
        export_public.export(out, captured['tmp'] / 'payload-public')


def test_export_rejects_nonstandard_local_path_in_reviewed_scalar(captured):
    out = captured['out']
    path = out / 'sources.json'
    data = json.loads(path.read_text())
    data[0]['origin'] = '/Volumes/confidential/filings'
    path.write_text(json.dumps(data))
    rewrite_inventory(out)
    # Rejected before output by the provenance hash link and the fixed-origin validator.
    with pytest.raises(BenchmarkError, match='Source provenance does not match'):
        export_public.export(out, captured['tmp'] / 'volume-public')


@pytest.mark.parametrize('key', ['provider_selection', 'source_scope', 'ordering', 'latency', 'rubric_review'])
def test_export_rejects_replaced_configuration_prose(captured, key):
    out = captured['out']
    path = out / 'config.json'
    data = json.loads(path.read_text())
    data[key] = 'SENTINEL-PROSE'
    path.write_text(json.dumps(data))
    rewrite_inventory(out)
    with pytest.raises(BenchmarkError, match='Invalid config values'):
        export_public.export(out, captured['tmp'] / 'text-public')


def test_pdf_fetch_and_production_extraction_use_complete_verified_bytes(tmp_path):
    fitz = pytest.importorskip('fitz')
    from benchmark.common import normalize_pdf
    sys.path.insert(0, str(ROOT.parent / 'ai-equity-research-copilot' / 'backend'))
    # The production parser is an optional sibling-checkout integration probe.
    try:
        from ai_equity_research_copilot_backend.parsing import parse_document
    except ImportError:
        if os.environ.get('FIN_EVAL_REQUIRE_COPILOT_PARSER') == '1':
            pytest.fail('CI requires the pinned public Copilot sibling parser')
        pytest.skip('Copilot sibling checkout is not installed')
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), 'NVIDIA revenue was $130.5 billion.')
        pdf.new_page().insert_text((72, 72), 'Full second page has supply-chain risks.')
        raw = pdf.tobytes()
    text = normalize_pdf(raw)
    doc = {'document_id': 'nvda_2025_10k', 'document_type': '10-K', 'period_end': '2025-01-26',
           'url': 'https://example.com/synthetic.pdf', 'sha256': sha256_bytes(raw)}
    manifest = {'retrieved_on': '2026-10-04', 'documents': [doc],
                'evidence': [{'document_id': doc['document_id'], 'normalized_text_sha256': sha256_bytes(text.encode())}]}
    sources = tmp_path / 'pdf-sources'
    with httpx.Client(transport=httpx.MockTransport(lambda request: httpx.Response(200, content=raw))) as client:
        fetch_sources.fetch_sources(manifest, sources, client, delay_s=0)
    derived = tmp_path / 'derived-pdf'
    derived.mkdir()
    plan = ingest.plan_uploads(manifest, sources, derived, lambda value: value,
                               lambda path: '\n\n'.join(page.text for page in parse_document(path)))
    assert Path(plan[0]['upload_path']).read_bytes() == raw
    assert 'Full second page' in Path(plan[0]['derived_text_path']).read_text()
    assert plan[0]['raw_sha256'] == sha256_bytes(raw)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda ing: ing.update(provider={"note": "SENTINEL-PROVIDER"}),
        lambda ing: ing.update(endpoint="http://user:SENTINEL@10.0.0.5:8766"),
        lambda ing: ing.update(operator_note="SENTINEL-NOTE"),
        lambda ing: ing["documents"][0].update(uuid={"nested": "SENTINEL-UUID"}),
        lambda ing: ing["documents"][0].update(metadata="SENTINEL-METADATA"),
        lambda ing: ing["documents"][0].update(extraction_method="SENTINEL copied filing paragraph"),
        lambda ing: ing["documents"][0]["metadata"].update(title="SENTINEL source paragraph"),
        lambda ing: ing["documents"][0].update(reviewer_note="SENTINEL"),
        lambda ing: ing["health"].update(note="SENTINEL"),
    ],
)
def test_export_rejects_unclassified_or_prose_ingestion_values(captured, mutate):
    out, tmp = captured["out"], captured["tmp"]
    probe = tmp / "probe"
    shutil.copytree(out, probe)
    ingestion = json.loads((probe / "ingestion.json").read_text())
    mutate(ingestion)
    (probe / "ingestion.json").write_text(json.dumps(ingestion))
    rewrite_inventory(probe)
    with pytest.raises(BenchmarkError):
        export_public.export(probe, tmp / "public-probe")
    assert not (tmp / "public-probe").exists()


@pytest.mark.parametrize(
    "key,value",
    [("python", "3.12.1 SENTINEL"), ("api_model", "SENTINEL-model"), ("suite_sha256", "SENTINEL"), ("api_base_url", "http://10.0.0.5:8766")],
)
def test_export_rejects_unreviewed_config_values(captured, key, value):
    out, tmp = captured["out"], captured["tmp"]
    probe = tmp / "config-probe"
    shutil.copytree(out, probe)
    config = json.loads((probe / "config.json").read_text())
    config[key] = value
    (probe / "config.json").write_text(json.dumps(config))
    rewrite_inventory(probe)
    with pytest.raises(BenchmarkError):
        export_public.export(probe, tmp / "public-config-probe")


# --- Final review: non-gold exported values are typed and linked, end-to-end -------------------

CANARY = "CHECKER_PRIVATE_PROSE_DO_NOT_EXPORT"


def _mutated_export(captured, file, mutate, jsonl=False):
    out, tmp = captured["out"], captured["tmp"]
    probe = tmp / "value-probe"
    shutil.copytree(out, probe)
    path = probe / file
    if jsonl:
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        mutate(rows)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    else:
        data = json.loads(path.read_text())
        mutate(data)
        path.write_text(json.dumps(data))
    rewrite_inventory(probe)
    public = tmp / "value-probe-public"
    with pytest.raises(BenchmarkError):
        export_public.export(probe, public)
    assert not public.exists()
    assert not any(p.name.startswith(".value-probe-public") for p in tmp.iterdir())


CONFIG_PROBES = {
    "source_documents prose": lambda c: c.update(source_documents=CANARY),
    "source_documents wrong count": lambda c: c.update(source_documents=c["source_documents"] + 1),
    "api_chunk_count prose": lambda c: c.update(api_chunk_count=CANARY),
    "api_chunk_count inconsistent": lambda c: c.update(api_chunk_count=c["api_chunk_count"] + 1),
    "api_top_k changed": lambda c: c.update(api_top_k=9),
    "api_top_k bool": lambda c: c.update(api_top_k=True),
    "capture_started_at prose": lambda c: c.update(capture_started_at=CANARY),
    "capture_started_at naive": lambda c: c.update(capture_started_at="2026-10-04T00:00:00"),
    "source_retrieved_on prose": lambda c: c.update(source_retrieved_on=CANARY),
    "scorer_version other": lambda c: c.update(scorer_version="financial-eval-scorer/v2"),
    "missing field": lambda c: c.pop("latency"),
}


@pytest.mark.parametrize("mutate", CONFIG_PROBES.values(), ids=CONFIG_PROBES.keys())
def test_export_rejects_untyped_or_unlinked_config_values(captured, mutate):
    _mutated_export(captured, "config.json", mutate)


METADATA_PROBES = {
    "started_at prose": ("started_at", CANARY),
    "started_at other time": ("started_at", "2020-01-01T00:00:00+00:00"),
    "case_count wrong": ("case_count", 99),
    "case_count prose": ("case_count", CANARY),
    "rubric count wrong": ("rubric_review_required_cases", 0),
    "duration prose": ("duration_ms", CANARY),
    "duration negative": ("duration_ms", -1),
    "configuration hash wrong": ("configuration_sha256", "0" * 64),
    "scorer prose": ("scorer_version", CANARY),
    "suite version prose": ("suite_version", CANARY),
    "checks extended": ("deterministic_checks", ["lexical-answer-point-coverage", "private-note"]),
    "map hash wrong": ("document_map", {"mapping_sha256": "0" * 64}),
    "target wrong": ("target", "lexical-baseline"),
}


@pytest.mark.parametrize("key,value", METADATA_PROBES.values(), ids=METADATA_PROBES.keys())
def test_export_rejects_untyped_or_unlinked_result_metadata(captured, key, value):
    _mutated_export(captured, "copilot-local/results.json", lambda data: data["metadata"].update({key: value}))


SOURCE_PROBES = {
    "origin prose": ("origin", CANARY),
    "url credentials": ("url", "https://FAKE_USER:CHECKER_FAKE_SECRET@example.com/nvda_2025_10k.htm"),
    "url query token": ("url", "https://example.com/nvda_2025_10k.htm?token=CHECKER_FAKE_SECRET"),
    "document type prose": ("document_type", CANARY),
    "period prose": ("period_end", CANARY),
    "count prose": ("normalized_characters", CANARY),
    "unknown field": ("operator_note", CANARY),
}


@pytest.mark.parametrize("key,value", SOURCE_PROBES.values(), ids=SOURCE_PROBES.keys())
def test_export_rejects_source_provenance_values_even_with_consistent_hash(captured, key, value):
    def mutate(config_dir):
        sources = json.loads((config_dir / "sources.json").read_text())
        sources[0][key] = value
        (config_dir / "sources.json").write_text(json.dumps(sources))
        # Make the provenance hash consistent so the value validators themselves are exercised.
        config = json.loads((config_dir / "config.json").read_text())
        config["source_provenance_sha256"] = export_public.sha256_json(sources)
        (config_dir / "config.json").write_text(json.dumps(config))
        for name in run.TARGETS:
            results = json.loads((config_dir / name / "results.json").read_text())
            results["metadata"]["configuration_sha256"] = sha256_bytes((config_dir / "config.json").read_bytes())
            (config_dir / name / "results.json").write_text(json.dumps(results))

    out, tmp = captured["out"], captured["tmp"]
    probe = tmp / "source-probe"
    shutil.copytree(out, probe)
    mutate(probe)
    rewrite_inventory(probe)
    with pytest.raises(BenchmarkError, match="source"):
        export_public.export(probe, tmp / "source-probe-public")
    assert not (tmp / "source-probe-public").exists()


@pytest.mark.parametrize(
    "url",
    [
        "https://FAKE_USER:CHECKER_FAKE_SECRET@example.com/document.html",
        "https://example.com/d.html?sig=CHECKER_FAKE_SECRET",
        "https://10.0.0.5/d.html",
        "https://localhost./document.pdf",
        "https://filings.internal./doc.htm",
        "https://corp.local./doc.htm",
        "https://foo\\bar.example/doc.htm",
        "https://foo..example/doc.htm",
        "https://-foo.example/doc.htm",
        "https://foo-.example/doc.htm",
        "https://127.1/doc.htm",
        "https://127.000.000.001/doc.htm",
        "https://127.0.0x1/doc.htm",
    ],
)
def test_export_rejects_credentialed_or_private_ingestion_source_url(captured, url):
    _mutated_export(captured, "ingestion.json", lambda data: data["documents"][0]["metadata"].update(source_url=url))


CAPTURE_PROBES = {
    "captured_at prose": lambda rows: rows[0].update(captured_at=CANARY),
    "latency prose": lambda rows: rows[0].update(latency_ms=CANARY),
    "request extra gold": lambda rows: rows[0]["request"].update(documents=["nvda_2025_10k"]),
    "request question changed": lambda rows: rows[0]["request"].update(question=CANARY),
    "unknown case": lambda rows: rows[0].update(case_id=CANARY),
}


@pytest.mark.parametrize("mutate", CAPTURE_PROBES.values(), ids=CAPTURE_PROBES.keys())
def test_export_rejects_untyped_or_unlinked_capture_rows(captured, mutate):
    _mutated_export(captured, "captures.jsonl", mutate, jsonl=True)


RESULT_PROBES = {
    "question differs from definition": lambda row: row.update(question=CANARY),
    "fingerprint mismatch": lambda row: row["case_definition"].update(notes=CANARY),
    "score prose": lambda row: row.update(overall_score=CANARY),
    "status prose": lambda row: row.update(execution_status=CANARY),
    "covered point not gold": lambda row: row.update(covered_points=[CANARY]),
    "citation reason prose": lambda row: row.update(bad_citations=[{"index": 0, "reason": CANARY}]),
    "assertion kind prose": lambda row: row.update(unsupported_assertions=[{"kind": CANARY, "text": "x"}]),
}


@pytest.mark.parametrize("mutate", RESULT_PROBES.values(), ids=RESULT_PROBES.keys())
def test_export_rejects_untyped_or_unlinked_result_values(captured, mutate):
    _mutated_export(captured, "copilot-local/results.json", lambda data: mutate(data["results"][0]))


def test_every_exported_config_field_has_a_value_validator():
    assert set(export_public.CONFIG_VALIDATORS) == export_public.CONFIG_FIELDS


@pytest.mark.parametrize("target", run.TARGETS)
def test_invalid_late_payload_is_rejected_before_any_export_file_write(captured, monkeypatch, target):
    out = captured["out"]
    path = out / target / "results.json"
    original = json.loads(path.read_text())
    original["metadata"]["started_at"] = CANARY
    path.write_text(json.dumps(original))
    rewrite_inventory(out)
    writes = []
    monkeypatch.setattr(export_public, "write_json", lambda *args: writes.append(args))
    with pytest.raises(BenchmarkError, match="metadata.*started_at"):
        export_public.export(out, captured["tmp"] / "late-public")
    assert writes == []
    assert not (captured["tmp"] / "late-public").exists()


@pytest.mark.parametrize("mutation", [
    lambda d: d["summary"].update({CANARY: 1}),
    lambda d: d["category_breakdown"].update({CANARY: d["category_breakdown"]["refusal"]}),
    lambda d: d["gate"]["thresholds"].update({CANARY: 1}),
    lambda d: d["results"][0]["diagnostics"].update(missing_citation_issue=1),
    lambda d: d["results"][0].update(estimated_cost_usd=float("inf")),
    lambda d: d["results"][0].pop("case_id"),
])
def test_export_checks_whole_non_gold_metric_and_diagnostic_schema(captured, mutation):
    _mutated_export(captured, "copilot-local/results.json", mutation)


@pytest.mark.parametrize("mutation", [
    lambda rows: rows.append(rows[0].copy()),
    lambda rows: rows.pop(),
    lambda rows: rows[0].update(response_sha256="0" * 64),
    lambda rows: rows[0].update(raw_response={"answer": CANARY}),
])
def test_export_requires_exact_capture_result_links(captured, mutation):
    _mutated_export(captured, "captures.jsonl", mutation, jsonl=True)


@pytest.mark.parametrize("mutation", [
    lambda d: d["documents"].append(d["documents"][0].copy()),
    lambda d: d["health"].update(chunks=-1),
    lambda d: d["health"].update(documents=99),
    lambda d: d["companies"].update(NVDA="different-uuid"),
    lambda d: d.update(equity_base_sha="0" * 40),
])
def test_export_links_whole_ingestion_record(captured, mutation):
    _mutated_export(captured, "ingestion.json", mutation)


def test_original_inventory_cannot_publish_unreviewed_filenames(captured):
    out = captured["out"]
    (out / (CANARY + ".txt")).write_text("synthetic operator note")
    rewrite_inventory(out)
    with pytest.raises(BenchmarkError, match="reviewed workflow artifacts"):
        export_public.export(out, captured["tmp"] / "inventory-public")
    assert not (captured["tmp"] / "inventory-public").exists()


def test_duplicate_non_gold_json_keys_fail_closed(captured):
    out = captured["out"]
    path = out / "config.json"
    text = path.read_text()
    path.write_text(text[:-2] + ', "source_documents": "' + CANARY + '"}\n')
    rewrite_inventory(out)
    with pytest.raises(BenchmarkError, match="Duplicate JSON field"):
        export_public.export(out, captured["tmp"] / "duplicate-public")


def test_target_numeric_fields_are_redacted_instead_of_treated_as_identifiers():
    raw = {"model": CANARY, "confidence": {"note": CANARY},
           "usage": {"provider": "local", "model": "local-deterministic-grounded-v1", "input_tokens": CANARY, "output_tokens": True, "estimated_cost_usd": float("inf")},
           "citations": [{"document_id": NVDA_UUID, "page": CANARY, "page_start": True, "score": CANARY, "company_id": "FAKE_USER:FAKE_SECRET@example.com"}]}
    public = export_public.public_raw_response(raw)
    assert CANARY not in json.dumps(public)
    assert "FAKE_SECRET" not in json.dumps(public)
    assert public["model"] == export_public.REDACTED
    assert public["citations"][0]["document_id"] == NVDA_UUID
