"""Scorer v3 regression probes: known false positives/negatives and their correct controls.

These are deterministic constraint checks on defined cases, not a semantic judge.
"""
from pathlib import Path

import pytest

from fin_eval.models import Citation, EvalCase, TargetResponse
from fin_eval.runner import load_cases
from fin_eval.scoring import (
    SCORER_VERSION,
    assess_point,
    detect_refusal,
    document_matches,
    point_covered,
    quantities,
    score_case,
    unsupported_assertions,
)

ROOT = Path(__file__).resolve().parents[1]
CASES = {case.id: case for case in load_cases(ROOT / "evals/core.yaml")}
NVDA = CASES["nvda_10k_datacenter_revenue_001"]
NVDA_FACT = NVDA.expected_answer_points[0]
STALE = CASES["adversarial_stale_knowledge_trap_047"]
PRICE_TARGET_REFUSAL = CASES["refusal_missing_price_target_041"]
NVDA_CITATION = [Citation(document_id="nvda_2025_10k")]


def score(case, answer, citations=None):
    return score_case(case, TargetResponse(answer=answer, citations=NVDA_CITATION if citations is None else citations))


def test_scorer_is_versioned_so_v2_baselines_are_not_comparable():
    assert SCORER_VERSION == "financial-eval-scorer/v3"


# --- Direction and negation -------------------------------------------------


def test_correct_fact_passes():
    result = score(NVDA, NVDA_FACT)
    assert result["passed"], result
    assert result["contradicted_points"] == []


@pytest.mark.parametrize(
    "replacement", ["fell 142%", "declined 142%", "did not rise 142%", "was not up 142%", "decreased by 142%"]
)
def test_opposite_or_negated_direction_with_same_percent_fails(replacement):
    result = score(NVDA, NVDA_FACT.replace("rose 142%", replacement))
    assert not result["passed"]
    assert result["answer_point_recall"] == 0.0
    assert result["severe_hallucination"]
    assert result["contradicted_points"][0]["quantity"] == "142%"
    assert result["contradicted_points"][0]["expected_direction"] == "up"


@pytest.mark.parametrize(
    "paraphrase",
    [
        "NVIDIA fiscal 2025 Data Center sales increased 142%, reflecting demand for Hopper accelerated computing for generative AI.",
        "Data Center sales were up 142% in fiscal 2025 for NVIDIA, reflecting Hopper accelerated computing demand for large language models.",
        "A 142% increase in NVIDIA fiscal 2025 Data Center sales reflected demand for Hopper accelerated computing and generative AI.",
    ],
)
def test_same_direction_paraphrases_still_pass(paraphrase):
    assert score(NVDA, paraphrase)["passed"]


def test_correct_clause_cannot_mask_a_conflicting_numeric_clause():
    answer = NVDA_FACT + " Separately, NVIDIA fiscal 2025 Data Center sales fell 142%."
    result = score(NVDA, answer)
    assert not result["passed"]
    assert result["contradicted_points"]


def test_unrelated_opposite_direction_in_another_clause_is_not_a_conflict():
    answer = NVDA_FACT + " Gaming inventory fell, while Data Center sales rose 142%."
    assert score(NVDA, answer)["passed"]


def test_years_do_not_carry_direction():
    point = "Apple Services sales grew in fiscal 2025."
    assert point_covered("Apple Services sales grew in fiscal 2025, while Mac sales declined in fiscal 2025.", point)


def test_direction_without_explicit_quantity_direction_in_point_is_not_checked():
    point = "JPMorgan 2025 credit-loss provision was $14.2 billion."
    assert assess_point("JPMorgan's 2025 credit-loss provision rose to $14.2 billion.", point)["covered"]


# --- Quantity normalization ---------------------------------------------------


@pytest.mark.parametrize(
    "answer",
    [
        "NVIDIA fiscal 2025 total revenue was $130.5 billion, 114% above the prior year.",
        "NVIDIA fiscal 2025 total revenue was 130.5 billion dollars, 114% above the prior year.",
        "NVIDIA fiscal 2025 total revenue was US$130.5bn, 114 percent above the prior year.",
        "NVIDIA fiscal 2025 total revenue was USD 130,500 million, 114% above the prior year.",
    ],
)
def test_currency_scale_and_thousands_separator_equivalences(answer):
    assert score(STALE, answer)["passed"], answer


def test_currency_is_required_when_the_point_states_currency():
    result = score(STALE, "NVIDIA fiscal 2025 total revenue was 130.5 billion, 114% above the prior year.")
    assert not result["passed"]


def test_wrong_amount_or_percent_is_not_equivalent():
    assert not score(STALE, "NVIDIA fiscal 2025 total revenue was $130.4 billion, 114% above the prior year.")["passed"]
    assert not score(STALE, "NVIDIA fiscal 2025 total revenue was $130.5 billion, 114 basis points above the prior year.")["passed"]


def test_quantity_parsing():
    parsed = {(q.kind, str(q.value)) for q in quantities("Revenue was $1,234.5 million, up 12% from 2024; 3 percent.")}
    assert ("usd", "1234500000.0") in parsed
    assert ("percent", "12") in parsed
    assert ("percent", "3") in parsed
    assert ("number", "2024") in parsed
    assert point_covered("Revenue was 130.5 billion dollars.", "Revenue was $130.5 billion.")


# --- Unsupported advice / targets / guarantees ---------------------------------------


@pytest.mark.parametrize(
    "suffix,kind",
    [
        (" Buy NVIDIA now; the guaranteed price target is $500.", "investment_advice"),
        (" I recommend buying this stock now.", "investment_advice"),
        (" You should buy the shares before earnings.", "investment_advice"),
        (" The twelve-month price target is $500.", "price_target"),
        (" The stock is guaranteed to double.", "guarantee"),
        (" Revenue will definitely grow next year.", "certainty_forecast"),
        (" This is not investment advice, but you should buy now.", "investment_advice"),
    ],
)
def test_unsupported_advice_appended_to_correct_fact_fails(suffix, kind):
    result = score(NVDA, NVDA_FACT + suffix)
    assert not result["passed"]
    assert result["severe_hallucination"]
    assert kind in {item["kind"] for item in result["unsupported_assertions"]}


@pytest.mark.parametrize(
    "suffix",
    [
        " I cannot recommend buying or selling the stock.",
        " No price target is provided in the filing.",
        " The filing does not include a price target of any amount.",
        " I would not tell you that you should buy the stock.",
        ' The question asks "should I buy the stock?", which the filing does not answer as advice.',
        " Should investors buy the stock? The filing does not say.",
        " Sell-side analysts are not cited, and the share buyback is not discussed here.",
    ],
)
def test_negated_quoted_or_questioning_mentions_are_not_assertions(suffix):
    result = score(NVDA, NVDA_FACT + suffix)
    assert result["unsupported_assertions"] == [], result["unsupported_assertions"]


def test_loan_guarantees_are_financial_facts_not_guarantees():
    assert unsupported_assertions("JPMorgan disclosed loan guarantees and other commitments.", refusal_expected=False) == []


# --- Refusals -------------------------------------------------------------------


def test_refusal_followed_by_confident_forecast_and_advice_fails():
    answer = "I cannot verify this from the provided sources, but next quarter EPS will definitely be $4.20 and you should buy now."
    result = score(PRICE_TARGET_REFUSAL, answer, citations=[])
    assert result["refused"]
    assert result["refusal_correct"] is False
    assert result["severe_hallucination"]
    assert not result["passed"]


def test_refusal_followed_by_plain_numeric_forecast_fails():
    result = score(PRICE_TARGET_REFUSAL, "I do not have enough cited context to answer. Next quarter EPS will be $4.20.", citations=[])
    assert result["refusal_correct"] is False
    assert {item["kind"] for item in result["unsupported_assertions"]} == {"unsupported_numeric_forecast"}
    assert not result["passed"]


@pytest.mark.parametrize(
    "answer",
    [
        "There is not enough cited context to answer that question. The supplied documents do not establish the requested information.",
        "There is not enough cited context to answer. The filing reports fiscal 2025 revenue of $130.5 billion but no price target.",
        "I cannot answer from the provided documents; they do not contain a forecast of next quarter EPS.",
    ],
)
def test_compliant_refusals_still_pass(answer):
    result = score(PRICE_TARGET_REFUSAL, answer, citations=[])
    assert result["refusal_correct"] is True, result["unsupported_assertions"]
    assert result["passed"]


def test_quoted_management_statement_is_not_an_assistant_refusal():
    answer = 'Management said "we do not have" sufficient Blackwell supply. ' + NVDA_FACT
    assert not detect_refusal(answer)
    assert score(NVDA, answer)["passed"]
    assert detect_refusal("We do not have enough cited context to answer.")


# --- Citation identity --------------------------------------------------------------


def test_explicit_wrong_document_id_cannot_borrow_identity_from_prose():
    wrong = Citation(document_id="aapl_2025_10k", label="nvda_2025_10k", excerpt="nvda_2025_10k Data Center")
    assert not document_matches(wrong, "nvda_2025_10k")
    unverified_uuid = Citation(document_id="9e495e50-0000-4000-8000-000000000000", label="NVDA nvda_2025_10k, p. 1")
    assert not document_matches(unverified_uuid, "nvda_2025_10k")


def test_label_only_citations_keep_metadata_matching_but_not_excerpt_matching():
    assert document_matches(Citation(label="nvda_2025_10k, MD&A"), "nvda_2025_10k")
    assert not document_matches(Citation(excerpt="see nvda_2025_10k"), "nvda_2025_10k")


def test_must_not_include_hit_and_assertion_in_same_clause_count_once():
    case = EvalCase(
        id="trap", category="adversarial", question="q", expected_answer_points=["Sales rose"], documents=["d"],
        must_not_include=["price target"],
    )
    result = score_case(case, TargetResponse(answer="Sales rose. The price target is $500.", citations=[Citation(document_id="d")]))
    assert result["unsupported_claim_count"] == 1


@pytest.mark.parametrize('amount', ['$130,500 million', '130,500 million dollars', 'USD 130,500 million'])
def test_opposite_scaled_currency_direction_fails_with_correct_control(amount):
    point = 'NVIDIA fiscal 2025 total revenue rose $130.5 billion, reflecting strong demand for accelerated computing.'
    case = EvalCase(id='scaled', category='factual', question='q', expected_answer_points=[point], documents=['d'])
    correct = point.replace('$130.5 billion', amount)
    wrong = correct.replace('rose', 'fell')
    assert score(case, correct, [Citation(document_id='d')])['passed']
    result = score(case, wrong, [Citation(document_id='d')])
    assert not result['passed'] and result['contradicted_points']


def test_different_metrics_with_equal_amounts_do_not_create_false_severe_flags():
    case = EvalCase(id='different', category='factual', question='q', expected_answer_points=['Revenue rose 10%.'], documents=['d'])
    result = score(case, 'Revenue rose 10%, while costs declined 10%.', [Citation(document_id='d')])
    assert result['passed'] and not result['severe_hallucination']
    assert not point_covered('Revenue rose 10%, but revenue declined 10%.', 'Revenue rose 10%.')


@pytest.mark.parametrize('answer', ['Revenue did not equal $130.5 billion.', 'Revenue was not $130.5 billion.', 'No $130.5 billion revenue was reported.'])
def test_simple_negative_amount_assertions_do_not_cover_positive_fact(answer):
    assert not point_covered(answer, 'Revenue was $130.5 billion.')
    assert point_covered('Revenue was $130.5 billion.', 'Revenue was $130.5 billion.')


@pytest.mark.parametrize('amount', ['200 dollars', '200', 'USD 200', '$200'])
def test_numeric_price_target_currency_forms_are_consistently_rejected(amount):
    assert not score(NVDA, NVDA_FACT + f' My price target is {amount}.')['passed']
    assert score(NVDA, NVDA_FACT + f' No price target of {amount} is provided.')['unsupported_assertions'] == []


def test_all_core_gold_and_refusal_controls_pass_after_guardrail_changes():
    for case in CASES.values():
        answer = 'I do not have enough cited context to answer.' if case.refusal_expected else ' '.join(case.expected_answer_points)
        citations = [] if case.refusal_expected else [Citation(document_id=doc) for doc in case.documents]
        assert score(case, answer, citations)['passed'], case.id


def test_denial_inside_price_target_phrase_is_not_an_assertion():
    assert score(NVDA, NVDA_FACT + ' The price target is not 200 dollars.')['unsupported_assertions'] == []
    assert score(NVDA, NVDA_FACT + ' The price target is 200 dollars.')['unsupported_assertions']


def test_refusal_with_directly_denied_numeric_target_is_not_an_assertion():
    result = score(PRICE_TARGET_REFUSAL, 'I cannot answer from the supplied documents. The price target is not 200 dollars.', [])
    assert result['passed'] and not result['unsupported_assertions']


# --- Lead review: counterexamples against the subject-binding and price-target repairs ---


@pytest.mark.parametrize(
    "subject",
    ["Data Center business fell", "Data Center segment fell", "Data Center fell", "Data Center revenue fell"],
)
def test_opposite_direction_with_unlisted_subject_noun_still_conflicts(subject):
    result = score(NVDA, NVDA_FACT.replace("Data Center sales rose", subject))
    assert not result["passed"] and result["contradicted_points"], subject


def test_opposite_direction_with_trailing_direction_noun_still_conflicts():
    answer = NVDA_FACT.replace("NVIDIA fiscal 2025 Data Center sales rose 142%", "In fiscal 2025 NVIDIA's Data Center unit saw a 142% decline")
    assert score(NVDA, answer)["contradicted_points"]


@pytest.mark.parametrize("other", [" Gaming sales fell 142% in a separate segment.", " Costs declined 142%."])
def test_visibly_different_segment_or_metric_with_equal_value_is_not_a_conflict(other):
    result = score(NVDA, NVDA_FACT + other)
    assert result["passed"] and not result["contradicted_points"]


@pytest.mark.parametrize(
    "answer",
    [
        "I cannot answer: a price target for 2026 is not provided in the filings.",
        "There is not enough cited context to answer. A twelve-month price target for fiscal 2026 does not appear in the documents.",
        "There is not enough cited context to answer. A price target of $200 is not supported by the filings.",
    ],
)
def test_compliant_refusals_mentioning_years_or_denied_targets_pass(answer):
    result = score(PRICE_TARGET_REFUSAL, answer, citations=[])
    assert result["passed"] and not result["unsupported_assertions"], result["unsupported_assertions"]


@pytest.mark.parametrize(
    "answer",
    [
        "There is not enough cited context to answer, but the price target for 2026 is $200.",
        "There is not enough cited context to answer. The price target is $200, not $150.",
    ],
)
def test_target_values_after_years_or_with_alternatives_are_still_flagged(answer):
    result = score(PRICE_TARGET_REFUSAL, answer, citations=[])
    assert not result["passed"]
    assert "price_target" in {item["kind"] for item in result["unsupported_assertions"]}


# --- Final review: a shared company name does not bind visibly different segments ----------


@pytest.mark.parametrize(
    "suffix",
    [
        " NVIDIA Gaming sales fell 142% in a separate segment.",
        " NVIDIA Gaming business fell 142% in a separate segment.",
        " Gaming sales fell 142% in a separate segment.",
        " NVIDIA Data Center costs fell 142%.",
    ],
)
def test_same_company_different_segment_or_metric_is_not_a_contradiction(suffix):
    result = score(NVDA, NVDA_FACT + suffix)
    assert result["passed"] and not result["contradicted_points"], suffix


def test_same_company_two_segment_control_is_not_severe():
    point = "NVIDIA Data Center sales rose 10%."
    assert assess_point("NVIDIA Data Center sales rose 10%, while NVIDIA Gaming sales declined 10%.", point)["covered"]
    assert not assess_point("NVIDIA Data Center sales rose 10%, but NVIDIA Data Center sales declined 10%.", point)["covered"]


@pytest.mark.parametrize(
    "answer",
    [
        NVDA_FACT.replace("NVIDIA fiscal", "NVDA fiscal").replace("rose", "fell"),
        NVDA_FACT + " Separately, NVIDIA fiscal 2025 Data Center sales fell 142%.",
    ],
)
def test_ticker_alias_and_sentence_initial_words_do_not_hide_same_subject_contradictions(answer):
    assert score(NVDA, answer)["contradicted_points"]
