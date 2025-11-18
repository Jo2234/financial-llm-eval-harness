from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Citation(BaseModel):
    model_config = ConfigDict(extra="allow")

    document_id: str | None = None
    chunk_id: str | None = None
    label: str | None = None
    excerpt: str | None = None
    section_title: str | None = None
    page: int | str | None = None
    url: str | None = None


class EvalCase(BaseModel):
    model_config = ConfigDict(extra="allow")

    id: str
    category: str
    difficulty: str = "medium"
    question: str
    expected_answer_points: list[str] = Field(default_factory=list)
    refusal_expected: bool = False
    company_ids: list[str] = Field(default_factory=list)
    documents: list[str] = Field(default_factory=list)
    required_citation_rules: list[dict[str, Any]] = Field(default_factory=list)
    acceptable_citation_chunk_ids: list[str] = Field(default_factory=list)
    must_not_include: list[str] = Field(default_factory=list)
    answer_type: str | None = None
    notes: str | None = None
    judge_rubric: str | None = None
    max_latency_ms: int | None = None
    max_estimated_cost_usd: float | None = None
    tags: list[str] = Field(default_factory=list)


class TargetResponse(BaseModel):
    model_config = ConfigDict(extra="allow")

    answer: str = ""
    citations: list[Citation] = Field(default_factory=list)
    raw_response: dict[str, Any] = Field(default_factory=dict)
    latency_ms: int = 0
    model: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    estimated_cost_usd: float | None = 0.0
    error: str | None = None


class TargetAdapter(Protocol):
    def answer(self, case: EvalCase) -> TargetResponse:
        ...
