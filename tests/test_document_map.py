"""Verified UUID -> suite document mapping, target request isolation and CLI integration."""

import json

import httpx
import pytest
from typer.testing import CliRunner

from fin_eval.adapters import CopilotApiAdapter, target_request
from fin_eval.cli import app
from fin_eval.document_map import (
    DocumentMapError,
    apply_document_map,
    load_document_map,
    parse_document_map,
    validate_document_map,
)
from fin_eval.models import Citation, EvalCase, TargetResponse
from fin_eval.runner import run_suite
from fin_eval.scoring import score_case

NVDA_UUID = "9e495e50-c08e-4b78-9eaa-6d65df3a0cdb"
AAPL_UUID = "9382c24c-ec5f-4cba-b3d8-6f81ddf33093"
UNKNOWN_UUID = "00000000-0000-4000-8000-000000000000"
MAP = {NVDA_UUID: "nvda_2025_10k", AAPL_UUID: "aapl_2025_10k"}
ANSWER = "NVIDIA fiscal 2025 Data Center sales rose 142%."


def case(**overrides):
    fields = {
        "id": "nvda_case_001",
        "category": "factual_extraction",
        "question": "How did NVIDIA Data Center sales change in fiscal 2025?",
        "company_ids": ["NVDA"],
        "documents": ["nvda_2025_10k"],
        "required_citation_rules": [{"document_id": "nvda_2025_10k"}],
        "expected_answer_points": [ANSWER],
        "judge_rubric": "Secret rubric text",
        "source_evidence": [{"document_id": "nvda_2025_10k", "url": "https://example.com", "excerpt": "Secret evidence"}],
    }
    fields.update(overrides)
    return EvalCase(**fields)


def sec_import_citation(document_id, excerpt="Data Center revenue"):
    """Shape of an Equity Copilot citation for an SEC-imported filing: UUID ID and a filing-style title."""
    return {
        "label": "NVDA 10-K filed 2025-02-26, p. 1",
        "title": "NVDA 10-K filed 2025-02-26",
        "document_id": document_id,
        "chunk_id": "39e95b18-42fa-4167-b9d6-bf2e2fe347c4",
        "excerpt": excerpt,
        "score": 1.1,
        "company_ticker": "NVDA",
    }


def mapped(citations):
    response = TargetResponse(answer=ANSWER, citations=[Citation(**c) for c in citations], raw_response={"citations": citations})
    return apply_document_map(response, parse_document_map(json.dumps(MAP)))


# --- Parsing and validation -----------------------------------------------------


@pytest.mark.parametrize(
    "text,message",
    [
        ("not json", "not valid JSON"),
        ("[]", "non-empty JSON object"),
        ("{}", "non-empty JSON object"),
        ('{"a": "x", "a": "y"}', "Duplicate key"),
        ('{"a": 1}', "must be a string"),
        ('{"a": " "}', "blank"),
        ('{" a": "x"}', "surrounding whitespace"),
        ('{"a": "x", "b": "x"}', "Ambiguous"),
        ('{"a": "x", "x": "y"}', "not suite document IDs"),
    ],
)
def test_malformed_or_ambiguous_maps_fail_loudly(text, message):
    with pytest.raises(DocumentMapError, match=message):
        parse_document_map(text)


def test_ingestion_manifest_form_and_hash_are_accepted(tmp_path):
    path = tmp_path / "ingestion.json"
    path.write_text(json.dumps({"document_uuid_to_canonical": MAP, "settings": {}}))
    document_map = load_document_map(path)
    assert document_map.mapping == MAP
    assert document_map.provenance()["entries"] == 2
    assert len(document_map.sha256) == 64


def test_map_must_target_known_suite_documents_and_cover_selected_cases():
    document_map = parse_document_map(json.dumps({NVDA_UUID: "nvda_2025_10k", AAPL_UUID: "made_up_doc"}))
    with pytest.raises(DocumentMapError, match="unknown suite documents: made_up_doc"):
        validate_document_map(document_map, [case()], [case()])
    partial = parse_document_map(json.dumps({AAPL_UUID: "aapl_2025_10k"}))
    suite = [case(), case(id="aapl", documents=["aapl_2025_10k"], required_citation_rules=[{"document_id": "aapl_2025_10k"}])]
    with pytest.raises(DocumentMapError, match="does not cover documents required by selected cases: nvda_2025_10k"):
        validate_document_map(partial, suite, suite[:1])
    provenance = validate_document_map(parse_document_map(json.dumps(MAP)), suite, suite)
    assert provenance["unmapped_suite_documents"] == []


# --- Scoring with and without a map ---------------------------------------------------


def test_correct_sec_import_uuid_citation_scores_only_with_verified_map():
    citations = [sec_import_citation(NVDA_UUID)]
    unmapped_response = TargetResponse(answer=ANSWER, citations=[Citation(**c) for c in citations])
    without = score_case(case(), unmapped_response)
    assert not without["passed"]
    assert without["missing_required_citations"] == ["document_id=nvda_2025_10k"]

    response = mapped(citations)
    result = score_case(case(), response)
    assert result["passed"], result
    citation = response.citations[0]
    assert citation.document_id == "nvda_2025_10k"
    assert citation.model_extra["source_document_id"] == NVDA_UUID
    assert citation.model_extra["document_map_status"] == "mapped"
    assert response.raw_response["citations"][0]["document_id"] == NVDA_UUID


def test_unmapped_uuid_with_gold_name_in_prose_is_rejected_not_guessed():
    citations = [sec_import_citation(UNKNOWN_UUID, excerpt="nvda_2025_10k says Data Center sales rose 142%")]
    citations[0]["label"] = "nvda_2025_10k"
    result = score_case(case(), mapped(citations))
    assert not result["passed"]
    assert result["citation_precision"] == 0.0
    assert result["bad_citations"][0]["reason"] == "document_id is not in the verified document map"
    assert result["bad_citations"][0]["document_id"] == UNKNOWN_UUID


def test_mapped_wrong_source_scores_as_wrong_and_missing_id_is_unverifiable():
    wrong = score_case(case(), mapped([sec_import_citation(AAPL_UUID, excerpt="nvda_2025_10k")]))
    assert not wrong["passed"]
    assert wrong["bad_citations"][0]["document_id"] == "aapl_2025_10k"
    missing = score_case(case(), mapped([{"label": "nvda_2025_10k", "excerpt": "Data Center"}]))
    assert not missing["passed"]
    assert "no document_id" in missing["bad_citations"][0]["reason"]


def test_mixed_correct_and_unmapped_citations_reduce_precision():
    result = score_case(case(), mapped([sec_import_citation(NVDA_UUID), sec_import_citation(UNKNOWN_UUID)]))
    assert result["citation_recall"] == 1.0
    assert result["citation_precision"] == 0.5


# --- Target request isolation -------------------------------------------------------


def test_copilot_request_contains_only_question_company_scope_and_top_k(monkeypatch):
    requests = []

    def fake_post(url, json, timeout):
        requests.append(json)
        return httpx.Response(
            200,
            json={"answer": ANSWER, "citations": [sec_import_citation(NVDA_UUID)]},
            request=httpx.Request("POST", url),
        )

    monkeypatch.setattr(httpx, "post", fake_post)
    eval_case = case()
    response = CopilotApiAdapter("http://127.0.0.1:9").answer(eval_case)
    assert requests == [{"question": eval_case.question, "company_ids": ["NVDA"], "top_k": 8}]
    assert requests[0] == target_request(eval_case)
    sent = json.dumps(requests[0])
    for secret in [eval_case.id, "nvda_2025_10k", ANSWER, "Secret rubric text", "Secret evidence"]:
        assert secret not in sent
    assert response.raw_response["citations"][0]["document_id"] == NVDA_UUID


# --- Runner and CLI -------------------------------------------------------------------


def write_suite(tmp_path):
    suite = tmp_path / "suite.json"
    other = case(id="aapl_case_002", company_ids=["AAPL"], documents=["aapl_2025_10k"],
                 required_citation_rules=[{"document_id": "aapl_2025_10k"}], source_evidence=[])
    suite.write_text(json.dumps({"cases": [case().model_dump(), other.model_dump()]}))
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"nvda_case_001": {"answer": ANSWER, "citations": [sec_import_citation(NVDA_UUID)]}}))
    map_path = tmp_path / "map.json"
    map_path.write_text(json.dumps(MAP))
    return suite, fixture, map_path


def test_run_suite_records_map_provenance_and_preserves_raw_uuid(tmp_path):
    suite, fixture, map_path = write_suite(tmp_path)
    selected = ["nvda_case_001"]
    without = run_suite(suite, out=tmp_path / "unmapped", fixture=str(fixture), case_ids=selected)
    assert not without["passed"]
    payload = run_suite(suite, out=tmp_path / "mapped", fixture=str(fixture), document_map=map_path, case_ids=selected)
    assert payload["passed"]
    row = payload["results"][0]
    assert row["raw_response"]["citations"][0]["document_id"] == NVDA_UUID
    assert row["citations"][0]["document_id"] == "nvda_2025_10k"
    assert row["citations"][0]["source_document_id"] == NVDA_UUID
    provenance = payload["metadata"]["document_map"]
    assert provenance["mapping_sha256"] == load_document_map(map_path).sha256
    assert provenance["source"] == str(map_path)
    config = json.loads((tmp_path / "mapped" / "config.json").read_text())
    assert config["document_map"]["mapping_sha256"] == provenance["mapping_sha256"]
    assert payload["metadata"]["scorer_version"] == "financial-eval-scorer/v3"
    assert payload["metadata"]["semantic_review"] == "not_performed"


def test_invalid_map_fails_before_any_target_request(tmp_path, monkeypatch):
    suite, _, map_path = write_suite(tmp_path)
    map_path.write_text(json.dumps({AAPL_UUID: "aapl_unknown"}))
    calls = []
    monkeypatch.setattr(httpx, "post", lambda *args, **kwargs: calls.append(args))
    with pytest.raises(DocumentMapError):
        run_suite(suite, target="copilot-api", base_url="http://127.0.0.1:9", out=tmp_path / "run", document_map=map_path)
    assert calls == []
    assert not (tmp_path / "run").exists()


def test_cli_document_map_option_and_loud_failure(tmp_path):
    suite, fixture, map_path = write_suite(tmp_path)
    runner = CliRunner()
    ok = runner.invoke(
        app,
        ["run", "--suite", str(suite), "--fixture", str(fixture), "--document-map", str(map_path), "--case-id", "nvda_case_001", "--out", str(tmp_path / "ok")],
    )
    assert ok.exit_code == 0, ok.output
    map_path.write_text('{"a": "x", "a": "y"}')
    bad = runner.invoke(
        app,
        ["run", "--suite", str(suite), "--fixture", str(fixture), "--document-map", str(map_path), "--out", str(tmp_path / "bad")],
    )
    assert bad.exit_code == 2
    assert "Duplicate key" in bad.output


def test_direct_document_map_cannot_bypass_structural_validation():
    from fin_eval.document_map import DocumentMap
    with pytest.raises(DocumentMapError, match='Ambiguous'):
        validate_document_map(DocumentMap({NVDA_UUID: 'nvda_2025_10k', AAPL_UUID: 'nvda_2025_10k'}), [case()], [case()])
    with pytest.raises(DocumentMapError, match='string target IDs'):
        validate_document_map(DocumentMap({NVDA_UUID: 1}), [case()], [case()])


def test_permitted_corpus_cannot_silently_erase_required_suite_documents():
    with pytest.raises(DocumentMapError, match='outside the permitted source corpus'):
        validate_document_map(parse_document_map(json.dumps(MAP)), [case()], [case()], permitted_documents=['aapl_2025_10k'])
