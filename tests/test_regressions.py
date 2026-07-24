"""Regression coverage for execution, adapter, and comparison integrity."""

import json

import httpx
import pytest
from typer.testing import CliRunner

from fin_eval.adapters import CopilotApiAdapter, MockAdapter, normalize_target_response
from fin_eval.cli import app
from fin_eval.models import Citation, EvalCase, TargetResponse
from fin_eval.runner import compare_runs, run_suite
from fin_eval.scoring import aggregate, score_case


@pytest.mark.parametrize("refusal_expected", [False, True])
def test_transport_errors_are_not_behavioral_evidence(refusal_expected):
    case = EvalCase(
        id="case",
        category="test",
        question="q",
        expected_answer_points=["Sales rose"],
        refusal_expected=refusal_expected,
    )
    result = score_case(case, TargetResponse(error="HTTP 404"))
    assert result["execution_status"] == "error"
    assert result["refusal_correct"] is None
    assert result["refused"] is None
    assert not result["behavior_evaluated"]
    assert not result["severe_hallucination"]
    assert result["unsupported_claim_count"] == 0
    assert not result["passed"]


def test_behavior_denominator_excludes_unavailable_answers():
    case = EvalCase(id="refusal", category="test", question="q", refusal_expected=True)
    rows = [
        score_case(case, response)
        for response in [
            TargetResponse(answer="I cannot answer from these documents."),
            TargetResponse(error="timeout"),
            TargetResponse(answer="   "),
        ]
    ]
    summary = aggregate(rows)
    assert summary["behavior_evaluated_cases"] == 1
    assert summary["behavior_unavailable_cases"] == 2
    assert summary["refusal_accuracy"] == 1
    assert summary["error_rate"] == pytest.approx(1 / 3)
    assert summary["severe_hallucination_count"] == 0
    assert aggregate(rows[1:])["refusal_accuracy"] == 0


def test_page_url_and_extra_metadata_survive_all_adapters(monkeypatch):
    case = EvalCase(
        id="page",
        category="test",
        question="q",
        expected_answer_points=["Margins improved"],
        required_citation_rules=[{"document_id": "doc", "page": 7}],
    )
    citation = {"document_id": "doc", "page": 7, "url": "https://example.com/filing", "source_version": "2025"}
    payload = {"answer": "Margins improved", "citations": [citation]}
    monkeypatch.setattr(
        httpx,
        "post",
        lambda *args, **kwargs: httpx.Response(
            200, json=payload, request=httpx.Request("POST", "https://example.com/research/chat")
        ),
    )
    responses = [
        TargetResponse(answer=payload["answer"], citations=[Citation(**citation)]),
        normalize_target_response(payload, 5),
        MockAdapter({case.id: payload}).answer(case),
        CopilotApiAdapter("https://example.com").answer(case),
    ]
    for response in responses:
        assert score_case(case, response)["passed"]
        assert response.citations[0].model_dump() | citation == response.citations[0].model_dump()


def make_runs(tmp_path):
    suite = tmp_path / "suite.json"
    suite.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "id": "easy",
                        "category": "test",
                        "question": "q1",
                        "expected_answer_points": ["Margins improved"],
                    },
                    {"id": "hard", "category": "test", "question": "q2", "expected_answer_points": ["Revenue fell"]},
                ]
            }
        )
    )
    fixture = tmp_path / "fixture.json"
    fixture.write_text(json.dumps({"hard": {"answer": "Other content"}}))
    run_suite(suite, out=tmp_path / "baseline", fixture=str(fixture))
    return suite, fixture


def test_removed_failure_does_not_pass_gate_or_count_as_fixed(tmp_path):
    suite, fixture = make_runs(tmp_path)
    run_suite(suite, out=tmp_path / "candidate", fixture=str(fixture), case_ids=["easy"])
    result = compare_runs(tmp_path / "baseline", tmp_path / "candidate")
    assert not result["regression_pass"]
    assert not result["comparable"]
    assert result["removed_cases"] == ["hard"]
    assert result["fixed_failures"] == []
    assert result["delta"] == {}
    cli = CliRunner().invoke(
        app, ["compare", "--baseline", str(tmp_path / "baseline"), "--candidate", str(tmp_path / "candidate"), "--gate"]
    )
    assert cli.exit_code == 1


def test_same_cohort_actual_fix_passes_and_changed_definitions_do_not(tmp_path):
    suite, _ = make_runs(tmp_path)
    run_suite(suite, out=tmp_path / "candidate")
    result = compare_runs(tmp_path / "baseline", tmp_path / "candidate")
    assert result["regression_pass"]
    assert result["fixed_failures"] == ["hard"]
    data = json.loads(suite.read_text())
    data["cases"][1]["expected_answer_points"] = ["Different expected result"]
    suite.write_text(json.dumps(data))
    run_suite(suite, out=tmp_path / "changed")
    result = compare_runs(tmp_path / "baseline", tmp_path / "changed")
    assert not result["regression_pass"]
    assert result["changed_cases"] == ["hard"]
    assert result["fixed_failures"] == []


@pytest.mark.parametrize("field", ["scorer_version", "case_fingerprint"])
def test_unversioned_artifacts_cannot_pass_a_regression_gate(tmp_path, field):
    suite, _ = make_runs(tmp_path)
    run_suite(suite, out=tmp_path / "candidate")
    path = tmp_path / "candidate" / "results.json"
    data = json.loads(path.read_text())
    if field == "scorer_version":
        del data["metadata"][field]
    else:
        del data["results"][0][field]
    path.write_text(json.dumps(data))
    result = compare_runs(tmp_path / "baseline", tmp_path / "candidate")
    assert not result["comparable"]
    assert not result["regression_pass"]


def test_empty_case_selection_is_not_a_run(tmp_path):
    suite, _ = make_runs(tmp_path)
    with pytest.raises(ValueError, match="No cases selected"):
        run_suite(suite, out=tmp_path / "empty", case_ids=["typo"])


def test_quality_suite_requires_complete_explicit_fixture(tmp_path):
    suite = tmp_path / "quality.json"
    suite.write_text(
        json.dumps(
            {
                "suite_kind": "quality",
                "cases": [
                    {
                        "id": "quality",
                        "category": "factual",
                        "question": "What happened to sales?",
                        "expected_answer_points": ["Sales increased 12%"],
                        "documents": ["doc"],
                        "required_citation_rules": [{"document_id": "doc"}],
                        "source_evidence": [
                            {
                                "document_id": "doc",
                                "url": "https://example.com/source",
                                "excerpt": "Sales increased 12%.",
                            }
                        ],
                    }
                ],
            }
        )
    )
    with pytest.raises(ValueError, match="explicit --fixture"):
        run_suite(suite, out=tmp_path / "run")
    fixture = tmp_path / "responses.json"
    fixture.write_text("{}")
    with pytest.raises(ValueError, match="missing selected cases"):
        run_suite(suite, out=tmp_path / "run", fixture=str(fixture))
    fixture.write_text(
        json.dumps({"quality": {"answer": "Sales increased 12%", "citations": [{"document_id": "doc"}]}})
    )
    result = run_suite(suite, out=tmp_path / "run", fixture=str(fixture))
    assert result["passed"]
    assert result["metadata"]["evaluation_mode"] == "fixture-replay"


def test_quality_validation_rejects_rubrics_or_unpinned_evidence():
    from fin_eval.runner import validate_cases

    case = EvalCase(
        id="bad",
        category="test",
        question="q",
        expected_answer_points=["The answer should state the direction"],
        documents=["doc"],
    )
    with pytest.raises(ValueError, match="factual, not rubric"):
        validate_cases([case], quality=True)
    case.expected_answer_points = ["Sales rose"]
    with pytest.raises(ValueError, match="evidence excerpt and URL"):
        validate_cases([case], quality=True)


def test_downloadable_reports_preserve_historical_rescore_context(tmp_path):
    from fin_eval.runner import html_report, markdown_report

    suite, _ = make_runs(tmp_path)
    payload = run_suite(suite, out=tmp_path / "rescore")
    payload["metadata"].update({
        "started_at": "2026-07-13T00:00:00+00:00",
        "rescored_at": "2026-09-06T00:00:00+00:00",
        "original_artifact": "historical/2026-07-13/v1/live/results.json",
    })
    for report in (markdown_report(payload), html_report(payload)):
        assert "Historical response rescore" in report
        assert "2026-07-13T00:00:00+00:00" in report
        assert "2026-09-06T00:00:00+00:00" in report
        assert "historical/2026-07-13/v1/live/results.json" in report
        assert "no new target requests" in report
    assert "## Severe Hallucinations" not in markdown_report(payload)
    assert "## Deterministic Severe Flags" in markdown_report(payload)


def test_html_report_labels_explicit_fixture_replay(tmp_path):
    from fin_eval.runner import html_report

    suite, fixture = make_runs(tmp_path)
    payload = run_suite(suite, out=tmp_path / "fixture", fixture=str(fixture))
    assert "fixture-replay" in html_report(payload)
    assert "not_performed" in html_report(payload)
