from __future__ import annotations

import json

import pytest

from fin_eval import runner
from fin_eval.models import Citation, TargetResponse


def write_suite(tmp_path, cases):
    suite = tmp_path / "suite.yaml"
    suite.write_text(json.dumps({"cases": cases}))
    return suite


def test_load_cases_accepts_top_level_list_or_cases_mapping(tmp_path):
    case = {
        "id": "case_1",
        "category": "factual_extraction",
        "question": "q",
        "expected_answer_points": ["Revenue increased"],
    }
    mapped_suite = write_suite(tmp_path, [case])
    list_suite = tmp_path / "list.yaml"
    list_suite.write_text(json.dumps([case]))

    assert [c.id for c in runner.load_cases(mapped_suite)] == ["case_1"]
    assert [c.id for c in runner.load_cases(list_suite)] == ["case_1"]


def test_adapter_for_requires_base_url_for_copilot_api():
    with pytest.raises(ValueError, match="base_url is required"):
        runner.adapter_for("copilot-api")


def test_adapter_for_rejects_unknown_target():
    with pytest.raises(ValueError, match="Unknown target"):
        runner.adapter_for("does-not-exist")


def test_run_suite_with_mock_adapter_writes_machine_and_human_reports(tmp_path):
    suite = write_suite(
        tmp_path,
        [
            {
                "id": "factual_1",
                "category": "factual_extraction",
                "question": "q",
                "expected_answer_points": ["Revenue increased due to Data Center demand"],
                "documents": ["nvda_10k"],
            },
            {
                "id": "refusal_1",
                "category": "refusal",
                "question": "What is next quarter's EPS?",
                "expected_answer_points": [],
                "refusal_expected": True,
            },
        ],
    )
    out_dir = tmp_path / "run"

    payload = runner.run_suite(suite=suite, target="mock", out=out_dir)

    assert payload["passed"] is True
    assert payload["gate"]["passed"] is True
    assert payload["gate"]["violations"] == []
    assert payload["target"] == "mock"
    assert payload["metadata"]["target"] == "mock"
    assert payload["metadata"]["case_count"] == 2
    assert payload["summary"]["total_cases"] == 2
    assert payload["summary"]["failed_cases"] == 0
    assert set(payload["category_breakdown"]) == {"factual_extraction", "refusal"}
    assert [r["case_id"] for r in payload["results"]] == ["factual_1", "refusal_1"]
    assert payload["results"][0]["answer"]
    citation = payload["results"][0]["citations"][0]
    assert {
        "document_id": "nvda_10k",
        "chunk_id": None,
        "label": "nvda_10k, p. 1",
        "excerpt": "Revenue increased due to Data Center demand",
        "section_title": None,
    }.items() <= citation.items()

    results_json = json.loads((out_dir / "results.json").read_text())
    config_json = json.loads((out_dir / "config.json").read_text())
    summary_md = (out_dir / "summary.md").read_text()
    report_html = (out_dir / "report.html").read_text()
    failures_csv = (out_dir / "failures.csv").read_text()

    assert results_json["passed"] is True
    assert config_json["target"] == "mock"
    assert config_json["thresholds"] == runner.THRESHOLDS
    assert (out_dir / "metrics.csv").read_text().startswith("scope,name,metric,value")
    assert len((out_dir / "cases.jsonl").read_text().splitlines()) == 2
    manifest = json.loads((out_dir / "manifest.json").read_text())
    assert manifest["schema"] == "financial-eval-run-manifest/v1"
    assert "results.json" in {artifact["name"] for artifact in manifest["artifacts"]}
    assert "# Financial QA Evaluation Report" in summary_md
    assert "- `target`: mock" in summary_md
    assert "- `pass`: True" in summary_md
    assert "## Category Metrics" in summary_md
    assert "<title>Financial QA Eval Report</title>" in report_html
    assert "<td>factual_1</td>" in report_html
    assert failures_csv.splitlines() == ["case_id,category,difficulty,overall_score,error,answer"]


def test_run_suite_uses_adapter_for_each_case_and_preserves_response_metrics(tmp_path, monkeypatch):
    suite = write_suite(
        tmp_path,
        [
            {
                "id": "case_a",
                "category": "factual_extraction",
                "question": "q1",
                "expected_answer_points": ["Revenue increased due to demand"],
                "documents": ["doc_a"],
            },
            {
                "id": "case_b",
                "category": "factual_extraction",
                "question": "q2",
                "expected_answer_points": ["Margin improved due to mix"],
                "documents": ["doc_b"],
            },
        ],
    )

    class RecordingAdapter:
        def __init__(self):
            self.case_ids = []

        def answer(self, case):
            self.case_ids.append(case.id)
            return TargetResponse(
                answer=case.expected_answer_points[0],
                citations=[Citation(document_id=case.documents[0])],
                raw_response={"case_id": case.id},
                latency_ms=123,
                model="unit-test",
                estimated_cost_usd=0.03,
            )

    adapter = RecordingAdapter()
    monkeypatch.setattr(runner, "adapter_for", lambda *args, **kwargs: adapter)

    payload = runner.run_suite(suite=suite, target="custom", out=tmp_path / "run")

    assert adapter.case_ids == ["case_a", "case_b"]
    assert payload["passed"] is True
    assert payload["summary"]["median_latency_ms"] == 123
    assert payload["summary"]["total_estimated_cost_usd"] == pytest.approx(0.06)
    assert [r["latency_ms"] for r in payload["results"]] == [123, 123]


def test_run_suite_gate_fails_on_severe_hallucination_and_reports_failure(tmp_path, monkeypatch):
    suite = write_suite(
        tmp_path,
        [
            {
                "id": "trap_1",
                "category": "adversarial",
                "question": "q",
                "expected_answer_points": ["Answer from source context"],
                "documents": ["doc1"],
                "must_not_include": ["price target"],
            }
        ],
    )

    class HallucinatingAdapter:
        def answer(self, case):
            return TargetResponse(
                answer="Answer from source context. The price target is $500.",
                citations=[Citation(document_id="doc1")],
            )

    monkeypatch.setattr(runner, "adapter_for", lambda *args, **kwargs: HallucinatingAdapter())
    out_dir = tmp_path / "run"

    payload = runner.run_suite(suite=suite, target="custom", out=out_dir)

    assert payload["passed"] is False
    assert payload["gate"]["passed"] is False
    assert {
        "metric": "severe_hallucination_count",
        "operator": "<=",
        "threshold": 0,
        "actual": 1,
    } in payload["gate"]["violations"]
    assert payload["summary"]["severe_hallucination_count"] == 1
    assert payload["results"][0]["unsupported_claim_count"] == 1
    assert payload["results"][0]["passed"] is False
    assert "trap_1,adversarial" in (out_dir / "failures.csv").read_text()
    assert "trap_1" in (out_dir / "summary.md").read_text()


def test_markdown_and_html_reports_include_failures_and_status():
    payload = {
        "target": "mock",
        "passed": False,
        "summary": {
            "total_cases": 1,
            "passed_cases": 0,
            "failed_cases": 1,
            "overall_score": 0.4,
            "answer_point_recall": 0.0,
            "citation_precision": 0.0,
            "error_rate": 1.0,
            "severe_hallucination_count": 1,
        },
        "metadata": {"suite": "suite.yaml", "started_at": "2026-06-25T00:00:00Z", "duration_ms": 10},
        "gate": {
            "violations": [
                {"metric": "overall_score", "operator": ">=", "threshold": 0.8, "actual": 0.4}
            ]
        },
        "category_breakdown": {
            "factual_extraction": {
                "total_cases": 1,
                "overall_score": 0.4,
                "passed_cases": 0,
                "error_rate": 1.0,
            }
        },
        "results": [
            {
                "case_id": "bad_case",
                "category": "factual_extraction",
                "overall_score": 0.4,
                "passed": False,
                "severe_hallucination": True,
                "unsupported_claim_count": 2,
                "latency_ms": 100,
                "estimated_cost_usd": 0.01,
                "error": "timeout",
            }
        ],
    }

    markdown = runner.markdown_report(payload)
    html = runner.html_report(payload)

    assert "- `pass`: False" in markdown
    assert "## Gate Violations" in markdown
    assert "| bad_case | factual_extraction | 0.400 | timeout |" in markdown
    assert "- `bad_case`: unsupported_claim_count=2" in markdown
    assert "<td>bad_case</td>" in html
    assert '<td class="fail">fail</td>' in html
