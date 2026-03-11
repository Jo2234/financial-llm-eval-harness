from __future__ import annotations

import json
import time
from typing import Any

import httpx
import yaml

from .models import Citation, EvalCase, TargetResponse


def _citation_from_payload(payload: Any) -> Citation:
    if isinstance(payload, Citation):
        return payload
    if isinstance(payload, str):
        return Citation(label=payload, excerpt=payload)
    if not isinstance(payload, dict):
        return Citation(label=str(payload))

    return Citation(
        document_id=payload.get("document_id") or payload.get("documentId") or payload.get("doc_id") or payload.get("source_id"),
        chunk_id=payload.get("chunk_id") or payload.get("chunkId") or payload.get("id"),
        label=payload.get("label") or payload.get("source") or payload.get("title"),
        excerpt=payload.get("excerpt") or payload.get("text") or payload.get("content"),
        section_title=payload.get("section_title") or payload.get("section") or payload.get("heading"),
    )


def _usage_value(usage: dict[str, Any], *keys: str) -> Any:
    for key in keys:
        if key in usage:
            return usage[key]
    return None


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def normalize_target_response(data: dict[str, Any], latency_ms: int) -> TargetResponse:
    usage = data.get("usage") or data.get("metadata", {}).get("usage") or {}
    if not isinstance(usage, dict):
        usage = {}

    citations_payload = data.get("citations") or data.get("sources") or data.get("references") or []
    if not isinstance(citations_payload, list):
        citations_payload = [citations_payload]

    return TargetResponse(
        answer=data.get("answer") or data.get("response") or data.get("content") or data.get("message") or "",
        citations=[_citation_from_payload(citation) for citation in citations_payload],
        raw_response=data,
        latency_ms=latency_ms,
        model=data.get("model") or _usage_value(usage, "model", "model_name"),
        input_tokens=_int_or_none(_usage_value(usage, "input_tokens", "prompt_tokens")),
        output_tokens=_int_or_none(_usage_value(usage, "output_tokens", "completion_tokens")),
        estimated_cost_usd=_float_or_none(_usage_value(usage, "estimated_cost_usd", "cost_usd", "cost")),
        error=data.get("error"),
    )


def load_fixture_responses(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        if path.endswith(".json"):
            payload = json.load(handle)
        else:
            payload = yaml.safe_load(handle)

    if isinstance(payload, dict) and "responses" in payload and isinstance(payload["responses"], dict):
        return payload["responses"]
    if isinstance(payload, dict):
        return payload
    raise ValueError("Fixture responses must be a mapping of case_id to response payload")


class MockAdapter:
    def __init__(self, fixture_responses: dict[str, Any] | None = None):
        self.fixture_responses = fixture_responses or {}

    def answer(self, case: EvalCase) -> TargetResponse:
        if case.id in self.fixture_responses:
            fixture = self.fixture_responses[case.id]
            if isinstance(fixture, TargetResponse):
                return fixture
            if isinstance(fixture, str):
                fixture = {"answer": fixture}
            if not isinstance(fixture, dict):
                return TargetResponse(error=f"Invalid fixture response for {case.id}: expected object or string")
            return normalize_target_response(fixture, latency_ms=int(fixture.get("latency_ms") or 5))

        if case.refusal_expected:
            answer = "I do not have enough cited context to answer that."
            citations: list[Citation] = []
        else:
            points = case.expected_answer_points[:2] or ["The answer is supported by the provided financial documents."]
            answer = " ".join(points)
            doc = case.documents[0] if case.documents else "demo_doc"
            citations = [Citation(document_id=doc, label=f"{doc}, p. 1", excerpt=answer[:160])]
        return TargetResponse(answer=answer, citations=citations, raw_response={"adapter": "mock"}, latency_ms=5, model="mock-fixture", estimated_cost_usd=0.0)


class CopilotApiAdapter:
    def __init__(self, base_url: str, endpoint: str = "/research/chat", timeout_s: float = 20.0):
        self.base_url = base_url.rstrip("/")
        self.endpoint = endpoint if endpoint.startswith("/") else f"/{endpoint}"
        self.timeout_s = timeout_s

    def answer(self, case: EvalCase) -> TargetResponse:
        started = time.perf_counter()
        try:
            payload: dict[str, Any] = {
                "case_id": case.id,
                "company_ids": case.company_ids,
                "documents": case.documents,
                "question": case.question,
                "top_k": 8,
            }
            response = httpx.post(f"{self.base_url}{self.endpoint}", json=payload, timeout=self.timeout_s)
            response.raise_for_status()
            data = response.json()
            return normalize_target_response(data, latency_ms=int((time.perf_counter() - started) * 1000))
        except Exception as exc:
            return TargetResponse(error=str(exc), latency_ms=int((time.perf_counter() - started) * 1000))
