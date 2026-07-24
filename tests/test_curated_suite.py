"""Grounded reference and deliberately incorrect contrasts exercise the same scorer."""

import json
from pathlib import Path

import pytest

from fin_eval.models import EvalCase, TargetResponse
from fin_eval.runner import load_cases, load_suite, validate_cases
from fin_eval.scoring import detect_refusal, score_case

ROOT = Path(__file__).resolve().parents[1]
CASES = {case.id: case for case in load_cases(ROOT / "evals/core.yaml")}
FIXTURES = json.loads((ROOT / "evals/fixtures/core_factual.json").read_text())["responses"]
CONTRASTS = json.loads((ROOT / "evals/fixtures/contrasts.json").read_text())["contrasts"]


def test_all_fifty_cases_have_grounded_targets_or_explicit_refusal():
    document = load_suite(ROOT / "evals/core.yaml")
    assert document.suite_kind == "quality"
    assert len(document.cases) == 50
    assert set(CASES) == set(FIXTURES)
    assert validate_cases(document.cases, quality=True)["case_count"] == 50
    for case in document.cases:
        assert case.judge_rubric
        if case.refusal_expected:
            assert case.expected_answer_points == []
        else:
            evidence_facts = {fact for evidence in case.source_evidence for fact in evidence.get("facts", [])}
            assert set(case.expected_answer_points) <= evidence_facts
            assert all(evidence.get("source_sha256") for evidence in case.source_evidence)


@pytest.mark.parametrize("case_id", sorted(CASES))
def test_curated_reference_response_passes(case_id):
    result = score_case(CASES[case_id], TargetResponse(**FIXTURES[case_id]))
    assert result["passed"], result


@pytest.mark.parametrize("contrast", CONTRASTS, ids=[row["name"] for row in CONTRASTS])
def test_contrasting_responses(contrast):
    result = score_case(CASES[contrast["case_id"]], TargetResponse(**contrast["response"]))
    assert result["passed"] == contrast["expected_passed"], result


@pytest.mark.parametrize(
    "answer",
    [
        "Inventory was insufficient to meet supply commitments.",
        "There was insufficient supply.",
        "The company did not have enough capital to expand.",
    ],
)
def test_financial_shortages_are_not_refusals(answer):
    assert not detect_refusal(answer)


@pytest.mark.parametrize("pattern", ["(", ""])
def test_invalid_answer_pattern_is_rejected_before_evaluation(pattern):
    case = EvalCase(
        id="invalid",
        category="factual",
        question="q",
        expected_answer_points=["Sales rose"],
        required_answer_patterns=[pattern],
    )
    with pytest.raises(ValueError, match="invalid required answer pattern"):
        validate_cases([case])
