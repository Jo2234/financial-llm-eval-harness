from __future__ import annotations

import re
from statistics import median
from typing import Any

from .models import Citation, EvalCase, TargetResponse

DEFAULT_MAX_LATENCY_MS = 8_000
REFUSAL_PATTERNS = [
    r"\bnot enough\b",
    r"\binsufficient\b",
    r"\bdo not have\b",
    r"\bdon't have\b",
    r"\bcannot answer\b",
    r"\bcan't answer\b",
    r"\bcannot determine\b",
    r"\bnot provided\b",
    r"\bnot available\b",
    r"\bnot in (?:the )?(?:provided |source |cited )?(?:documents|context|sources)\b",
]
STOPWORDS = {
    "about",
    "above",
    "after",
    "also",
    "among",
    "answer",
    "because",
    "been",
    "being",
    "between",
    "cited",
    "could",
    "documents",
    "during",
    "from",
    "growth",
    "have",
    "include",
    "into",
    "main",
    "more",
    "provided",
    "revenue",
    "should",
    "source",
    "that",
    "their",
    "there",
    "these",
    "this",
    "tied",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def normalize(text: str | None) -> str:
    return re.sub(r"\s+", " ", (text or "").lower()).strip()


def words(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", normalize(text))


def meaningful_words(text: str | None) -> list[str]:
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def numbers(text: str | None) -> set[str]:
    return set(re.findall(r"\$?\d+(?:\.\d+)?%?", normalize(text)))


def point_covered(answer: str, point: str) -> bool:
    answer_n = normalize(answer)
    point_n = normalize(point)
    if not point_n:
        return True
    if point_n in answer_n:
        return True

    point_numbers = numbers(point)
    if point_numbers and not point_numbers.issubset(numbers(answer)):
        return False

    point_terms = meaningful_words(point)
    if not point_terms:
        return True

    answer_terms = set(meaningful_words(answer))
    covered_terms = sum(1 for term in point_terms if term in answer_terms or term in answer_n)
    threshold = 0.45 if len(point_terms) >= 5 else 0.66
    return covered_terms / len(point_terms) >= threshold


def detect_refusal(answer: str) -> bool:
    answer_n = normalize(answer)
    return any(re.search(pattern, answer_n) for pattern in REFUSAL_PATTERNS)


def citation_extra(citation: Citation, key: str) -> Any:
    if hasattr(citation, key):
        return getattr(citation, key)
    return (citation.model_extra or {}).get(key)


def citation_text(citation: Citation) -> str:
    parts = [
        citation.document_id,
        citation.chunk_id,
        citation.label,
        citation.excerpt,
        citation.section_title,
        str(citation.page) if citation.page is not None else None,
        citation.url,
    ]
    for value in (citation.model_extra or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return normalize(" ".join(part for part in parts if part))


def citation_is_structured(citation: Citation) -> bool:
    return any(
        bool(value)
        for value in [
            citation.document_id,
            citation.chunk_id,
            citation.label,
            citation.excerpt,
            citation.section_title,
            citation.page,
            citation.url,
        ]
    )


def document_matches(citation: Citation, document_id: str) -> bool:
    target = normalize(document_id)
    if not target:
        return True
    if normalize(citation.document_id) == target:
        return True
    return target in citation_text(citation)


def rule_matches(citation: Citation, rule: dict[str, Any]) -> bool:
    document_id = rule.get("document_id")
    if document_id and not document_matches(citation, str(document_id)):
        return False

    chunk_id = rule.get("chunk_id")
    if chunk_id and normalize(citation.chunk_id) != normalize(str(chunk_id)):
        return False

    section_contains = rule.get("section_contains")
    if section_contains:
        section_text = normalize(" ".join(part for part in [citation.section_title, citation.excerpt, citation.label] if part))
        if normalize(str(section_contains)) not in section_text:
            return False

    label_contains = rule.get("label_contains")
    if label_contains and normalize(str(label_contains)) not in normalize(citation.label):
        return False

    excerpt_contains = rule.get("excerpt_contains")
    if excerpt_contains and normalize(str(excerpt_contains)) not in normalize(citation.excerpt):
        return False

    page = rule.get("page")
    if page is not None and str(citation.page) != str(page) and normalize(str(page)) not in citation_text(citation):
        return False

    return True


def rule_label(rule: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in sorted(rule.items()) if value not in (None, "", [])]
    return ", ".join(parts) if parts else "<any citation>"


def score_citations(case: EvalCase, response: TargetResponse, refused: bool) -> dict[str, Any]:
    citations = response.citations
    structured = [citation for citation in citations if citation_is_structured(citation)]
    unstructured_indexes = [index for index, citation in enumerate(citations) if not citation_is_structured(citation)]

    explicit_rules = list(case.required_citation_rules)
    required_rules = explicit_rules or ([{"document_id": document_id} for document_id in case.documents] if not case.refusal_expected else [])

    acceptable_chunks = {normalize(chunk_id) for chunk_id in case.acceptable_citation_chunk_ids}

    matched_rules: list[dict[str, Any]] = []
    missing_rules: list[dict[str, Any]] = []
    for rule in required_rules:
        if any(rule_matches(citation, rule) for citation in structured):
            matched_rules.append(rule)
        else:
            missing_rules.append(rule)

    good_indexes: set[int] = set()
    bad_citations: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        if index in unstructured_indexes:
            bad_citations.append({"index": index, "reason": "citation is empty or unstructured"})
            continue
        chunk_ok = normalize(citation.chunk_id) in acceptable_chunks if acceptable_chunks else False
        rule_ok = any(rule_matches(citation, rule) for rule in required_rules)
        document_ok = any(document_matches(citation, document_id) for document_id in case.documents)
        unconstrained_ok = not required_rules and not case.documents
        if chunk_ok or rule_ok or document_ok or unconstrained_ok:
            good_indexes.add(index)
        else:
            bad_citations.append(
                {
                    "index": index,
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "reason": "does not match required documents, chunks, or citation rules",
                }
            )

    if citations:
        citation_precision = len(good_indexes) / len(citations)
    elif case.refusal_expected and refused:
        citation_precision = 1.0
    else:
        citation_precision = 0.0 if required_rules or case.documents else 1.0

    if required_rules:
        citation_recall = len(matched_rules) / len(required_rules)
    elif case.refusal_expected and refused:
        citation_recall = 1.0
    elif case.documents:
        citation_recall = 1.0 if citations else 0.0
    else:
        citation_recall = 1.0

    return {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "bad_citations": bad_citations,
        "missing_required_citations": [rule_label(rule) for rule in missing_rules],
        "structured_citation_count": len(structured),
    }


def score_format(case: EvalCase, response: TargetResponse, refused: bool, citation_score: dict[str, Any]) -> float:
    checks: list[bool] = [
        bool((response.answer or "").strip()),
        response.error is None,
        response.latency_ms is not None and response.latency_ms >= 0,
    ]

    citations_required = not (case.refusal_expected and refused) and bool(case.documents or case.required_citation_rules)
    if citations_required:
        checks.append(bool(response.citations))
        checks.append(citation_score["structured_citation_count"] == len(response.citations) and bool(response.citations))
    elif response.citations:
        checks.append(citation_score["structured_citation_count"] == len(response.citations))

    if case.answer_type and "cited" in normalize(case.answer_type):
        checks.append(bool(response.citations))

    return sum(float(check) for check in checks) / len(checks) if checks else 0.0


def score_latency(case: EvalCase, response: TargetResponse) -> float:
    latency_ms = max(int(response.latency_ms or 0), 0)
    max_latency_ms = case.max_latency_ms or DEFAULT_MAX_LATENCY_MS
    if latency_ms <= max_latency_ms:
        return 1.0
    return max(max_latency_ms / latency_ms, 0.0)


def score_cost(case: EvalCase, response: TargetResponse) -> float:
    cost = response.estimated_cost_usd
    if cost is None or case.max_estimated_cost_usd is None:
        return 1.0
    if cost <= case.max_estimated_cost_usd:
        return 1.0
    if cost <= 0:
        return 1.0
    return max(case.max_estimated_cost_usd / cost, 0.0)


def score_case(case: EvalCase, response: TargetResponse) -> dict[str, Any]:
    answer = response.answer or ""
    expected = case.expected_answer_points
    covered = [point for point in expected if point_covered(answer, point)]
    missing = [point for point in expected if point not in covered]
    answer_recall = len(covered) / len(expected) if expected else 1.0

    refused = detect_refusal(answer)
    refusal_correct = refused if case.refusal_expected else not refused

    answer_n = normalize(answer)
    bad_terms = [term for term in case.must_not_include if normalize(term) and normalize(term) in answer_n]

    citation_score = score_citations(case, response, refused)
    format_score = score_format(case, response, refused, citation_score)
    latency_score = score_latency(case, response)
    cost_score = score_cost(case, response)

    missing_citation_issue = bool(citation_score["missing_required_citations"])
    unsupported_claim_count = len(bad_terms)
    if case.refusal_expected and not refused:
        unsupported_claim_count += 1
    if not case.refusal_expected and expected and not response.citations:
        unsupported_claim_count += 1

    severe = bool(bad_terms) or (case.refusal_expected and not refused)
    overall = (
        0.35 * answer_recall
        + 0.25 * citation_score["citation_precision"]
        + 0.15 * citation_score["citation_recall"]
        + 0.15 * float(refusal_correct)
        + 0.10 * format_score
    )

    latency_failure = case.max_latency_ms is not None and latency_score < 1.0
    cost_failure = case.max_estimated_cost_usd is not None and cost_score < 1.0
    passed = (
        overall >= 0.8
        and answer_recall >= 0.8
        and citation_score["citation_precision"] >= 0.8
        and citation_score["citation_recall"] >= 0.75
        and refusal_correct
        and format_score >= 0.75
        and not severe
        and not latency_failure
        and not cost_failure
        and response.error is None
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "passed": passed,
        "overall_score": overall,
        "answer_point_recall": answer_recall,
        "covered_points": covered,
        "missing_points": missing,
        "citation_precision": citation_score["citation_precision"],
        "citation_recall": citation_score["citation_recall"],
        "bad_citations": citation_score["bad_citations"],
        "missing_required_citations": citation_score["missing_required_citations"],
        "refusal_correct": refusal_correct,
        "refused": refused,
        "format_score": format_score,
        "latency_score": latency_score,
        "cost_score": cost_score,
        "unsupported_claim_count": unsupported_claim_count,
        "must_not_include_hits": bad_terms,
        "severe_hallucination": severe,
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
        "estimated_cost_usd": response.estimated_cost_usd or 0.0,
        "error": response.error,
        "diagnostics": {
            "missing_citation_issue": missing_citation_issue,
            "latency_budget_ms": case.max_latency_ms or DEFAULT_MAX_LATENCY_MS,
            "cost_budget_usd": case.max_estimated_cost_usd,
        },
    }


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    index = int(pct * (len(values) - 1))
    return values[index]


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {
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
            "format_score": 0.0,
            "latency_score": 0.0,
            "cost_score": 0.0,
            "unsupported_claim_count": 0,
            "total_tokens": 0,
            "cost_per_case_usd": 0.0,
            "cost_per_successful_answer_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "error_rate": 0.0,
        }

    def avg(key: str) -> float:
        return sum(float(result.get(key) or 0.0) for result in results) / total

    latencies = sorted(int(result.get("latency_ms") or 0) for result in results)
    total_cost = sum(float(result.get("estimated_cost_usd") or 0.0) for result in results)
    passed_count = sum(1 for result in results if result["passed"])
    successful_cost = sum(float(result.get("estimated_cost_usd") or 0.0) for result in results if result["passed"])
    return {
        "total_cases": total,
        "passed_cases": passed_count,
        "failed_cases": total - passed_count,
        "overall_score": avg("overall_score"),
        "answer_point_recall": avg("answer_point_recall"),
        "citation_precision": avg("citation_precision"),
        "citation_recall": avg("citation_recall"),
        "refusal_accuracy": sum(1 for result in results if result["refusal_correct"]) / total,
        "format_score": avg("format_score"),
        "latency_score": avg("latency_score"),
        "cost_score": avg("cost_score"),
        "severe_hallucination_count": sum(1 for result in results if result["severe_hallucination"]),
        "unsupported_claim_count": sum(int(result.get("unsupported_claim_count") or 0) for result in results),
        "median_latency_ms": median(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "total_input_tokens": sum(int(result.get("input_tokens") or 0) for result in results),
        "total_output_tokens": sum(int(result.get("output_tokens") or 0) for result in results),
        "total_tokens": sum(int(result.get("total_tokens") or 0) for result in results),
        "total_estimated_cost_usd": total_cost,
        "cost_per_case_usd": total_cost / total,
        "cost_per_successful_answer_usd": successful_cost / passed_count if passed_count else 0.0,
        "error_rate": sum(1 for result in results if result.get("error")) / total,
    }
