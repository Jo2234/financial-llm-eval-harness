"""Report safety and offline evidence retention checks."""

import json
from copy import deepcopy
from html import unescape
from html.parser import HTMLParser

from fin_eval.reporting import render_report


class Elements(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.tags = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


def test_report_escapes_untrusted_evidence_and_preserves_input():
    malicious = '</pre><script>alert("x")</script>'
    row = {
        "case_id": 'x" onclick="alert(1)',
        "category": "factual_extraction",
        "question": malicious,
        "overall_score": 0.25,
        "passed": False,
        "answer": malicious,
        "missing_points": [malicious],
        "error": malicious,
        "case_definition": {"refusal_expected": True, "company_ids": ["EXAMPLE"]},
    }
    payload = {
        "target": malicious,
        "passed": False,
        "summary": {"total_cases": 1, "passed_cases": 0, "failed_cases": 1, "overall_score": 0.25},
        "metadata": {"original_artifact": malicious},
        "results": [row],
    }
    original = deepcopy(payload)
    report = render_report(payload, [malicious])
    assert payload == original
    assert malicious not in report
    elements = Elements(report).tags
    assert len([tag for tag, _ in elements if tag == "script"]) == 1  # Only trusted filter logic.
    assert all("onclick" not in attrs for _, attrs in elements)
    case = next(attrs for tag, attrs in elements if tag == "details" and attrs.get("class") == "case")
    assert case["data-case-id"] == row["case_id"]
    assert case["data-refusal"] == "true"
    assert "example" in case["data-search"]
    assert json.dumps(row, indent=2, ensure_ascii=False) in unescape(report)


def test_empty_report_keeps_zero_counts_and_native_details_available():
    payload = {
        "target": "mock",
        "passed": False,
        "summary": {"total_cases": 0, "passed_cases": 0, "failed_cases": 0, "overall_score": 0},
        "results": [],
    }
    report = render_report(payload, [])
    assert "0 of 0 cases" in report
    assert "No matching cases" in report
    assert "<noscript>" in report
    assert "Run gate and violations" in report
