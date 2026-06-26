import pytest

from fin_eval.models import Citation, EvalCase, TargetResponse
from fin_eval.scoring import aggregate, point_covered, score_case
from fin_eval.runner import load_cases


def test_score_case_with_citation_passes():
    case = EvalCase(id="c1", category="factual", question="q", expected_answer_points=["Data Center revenue growth"], documents=["doc1"], refusal_expected=False)
    response = TargetResponse(answer="Data Center revenue growth was strong.", citations=[Citation(document_id="doc1")])
    result = score_case(case, response)
    assert result["passed"]


def test_refusal_case_detected():
    case = EvalCase(id="c2", category="refusal", question="q", expected_answer_points=[], refusal_expected=True)
    response = TargetResponse(answer="I do not have enough cited context to answer.")
    result = score_case(case, response)
    assert result["refusal_correct"]
    assert result["citation_precision"] == 1.0
    assert result["citation_recall"] == 1.0
    assert result["passed"]


def test_point_coverage_uses_keyword_recall_instead_of_exact_match():
    assert point_covered(
        "Demand for accelerated AI computing was the main Data Center driver.",
        "Data Center revenue growth was driven by demand for accelerated computing and AI.",
    )
    assert not point_covered(
        "The company discussed gross margin and share repurchases.",
        "Data Center revenue growth was driven by demand for accelerated computing and AI.",
    )


def test_missing_answer_points_reduce_recall_and_list_missing_points():
    case = EvalCase(
        id="missing",
        category="cited_summary",
        question="q",
        expected_answer_points=[
            "Data Center revenue growth was a key driver.",
            "Management cited demand for accelerated computing and AI.",
        ],
        documents=["nvda_10k"],
    )
    response = TargetResponse(
        answer="Data Center revenue growth was a key driver.",
        citations=[Citation(document_id="nvda_10k")],
    )

    result = score_case(case, response)

    assert result["answer_point_recall"] == 0.5
    assert result["missing_points"] == ["Management cited demand for accelerated computing and AI."]
    assert not result["passed"]


def test_wrong_citation_document_penalizes_precision_but_not_citation_presence():
    case = EvalCase(
        id="bad_cite",
        category="factual_extraction",
        question="q",
        expected_answer_points=["Data Center revenue growth"],
        documents=["nvda_10k"],
    )
    response = TargetResponse(
        answer="Data Center revenue growth was strong.",
        citations=[Citation(document_id="aapl_10k"), Citation(document_id="msft_10k")],
    )

    result = score_case(case, response)

    assert result["citation_precision"] == 0.0
    assert result["citation_recall"] == 0.0
    assert result["bad_citations"] == [
        {
            "index": 0,
            "document_id": "aapl_10k",
            "chunk_id": None,
            "reason": "does not match required documents, chunks, or citation rules",
        },
        {
            "index": 1,
            "document_id": "msft_10k",
            "chunk_id": None,
            "reason": "does not match required documents, chunks, or citation rules",
        },
    ]
    assert result["missing_required_citations"] == ["document_id=nvda_10k"]
    assert not result["passed"]


def test_citation_label_can_satisfy_document_match_when_document_id_missing():
    case = EvalCase(
        id="label_cite",
        category="factual_extraction",
        question="q",
        expected_answer_points=["Data Center revenue growth"],
        documents=["nvda_10k"],
    )
    response = TargetResponse(
        answer="Data Center revenue growth was strong.",
        citations=[Citation(label="nvda_10k, MD&A, p. 32")],
    )

    result = score_case(case, response)

    assert result["citation_precision"] == 1.0
    assert result["passed"]


def test_must_not_include_terms_create_severe_hallucination_and_force_failure():
    case = EvalCase(
        id="trap",
        category="adversarial",
        question="q",
        expected_answer_points=["The answer should remain source-grounded"],
        documents=["doc1"],
        must_not_include=["price target", "buy recommendation"],
    )
    response = TargetResponse(
        answer="The answer should remain source-grounded, but it also gives a price target.",
        citations=[Citation(document_id="doc1")],
    )

    result = score_case(case, response)

    assert result["unsupported_claim_count"] == 1
    assert result["severe_hallucination"]
    assert not result["passed"]


def test_refusal_expected_but_answered_is_incorrect_and_severe():
    case = EvalCase(
        id="unsupported",
        category="refusal",
        question="What is next quarter's EPS?",
        expected_answer_points=[],
        refusal_expected=True,
    )
    response = TargetResponse(answer="Next quarter EPS will be $4.20.", citations=[])

    result = score_case(case, response)

    assert result["refusal_correct"] is False
    assert result["severe_hallucination"]
    assert not result["passed"]


def test_refusal_when_answer_is_expected_fails_refusal_correctness():
    case = EvalCase(
        id="false_refusal",
        category="factual_extraction",
        question="q",
        expected_answer_points=["Revenue increased due to demand"],
        documents=["doc1"],
    )
    response = TargetResponse(answer="I cannot answer from the provided context.")

    result = score_case(case, response)

    assert result["refusal_correct"] is False
    assert result["format_score"] == 0.6
    assert not result["passed"]


def test_response_error_fails_format_and_case_even_with_good_answer():
    case = EvalCase(
        id="error",
        category="factual_extraction",
        question="q",
        expected_answer_points=["Revenue increased due to demand"],
        documents=["doc1"],
    )
    response = TargetResponse(
        answer="Revenue increased due to demand.",
        citations=[Citation(document_id="doc1")],
        error="timeout",
    )

    result = score_case(case, response)

    assert result["format_score"] == 0.8
    assert result["error"] == "timeout"
    assert not result["passed"]


def test_aggregate_summarizes_quality_latency_cost_and_errors():
    results = [
        {
            "passed": True,
            "overall_score": 1.0,
            "answer_point_recall": 1.0,
            "citation_precision": 1.0,
            "citation_recall": 1.0,
            "refusal_correct": True,
            "severe_hallucination": False,
            "latency_ms": 100,
            "estimated_cost_usd": 0.10,
            "error": None,
        },
        {
            "passed": False,
            "overall_score": 0.5,
            "answer_point_recall": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "refusal_correct": False,
            "severe_hallucination": True,
            "latency_ms": 500,
            "estimated_cost_usd": 0.20,
            "error": "timeout",
        },
    ]

    summary = aggregate(results)

    assert summary["total_cases"] == 2
    assert summary["passed_cases"] == 1
    assert summary["failed_cases"] == 1
    assert summary["overall_score"] == 0.75
    assert summary["refusal_accuracy"] == 0.5
    assert summary["severe_hallucination_count"] == 1
    assert summary["median_latency_ms"] == 300
    assert summary["p95_latency_ms"] == 100
    assert summary["total_estimated_cost_usd"] == pytest.approx(0.30)
    assert summary["error_rate"] == 0.5


def test_aggregate_empty_results_returns_zeroed_summary():
    summary = aggregate([])

    assert {
        "total_cases": 0,
        "passed_cases": 0,
        "failed_cases": 0,
        "overall_score": 0.0,
        "answer_point_recall": 0.0,
        "citation_precision": 0.0,
        "citation_recall": 0.0,
        "refusal_accuracy": 0.0,
        "severe_hallucination_count": 0,
        "median_latency_ms": 0,
        "p95_latency_ms": 0,
        "total_estimated_cost_usd": 0,
        "error_rate": 0.0,
    }.items() <= summary.items()
    assert summary["format_score"] == 0.0
    assert summary["latency_score"] == 0.0
    assert summary["cost_score"] == 0.0
    assert summary["unsupported_claim_count"] == 0
    assert summary["total_tokens"] == 0
    assert summary["cost_per_case_usd"] == 0.0


def test_core_suite_has_50_cases():
    cases = load_cases("evals/core.yaml")
    assert len(cases) >= 50
