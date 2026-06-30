from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from .models import EvalCase

SCHEMA_VERSION = "financial-eval-suite/v1"


class EvalSuite(BaseModel):
    """Top-level suite document used by YAML/JSON eval files."""

    model_config = ConfigDict(extra="allow")

    schema_version: str | None = Field(default=None, description="Optional suite schema version marker.")
    name: str | None = Field(default=None, description="Human-readable suite name.")
    description: str | None = None
    cases: list[EvalCase]


class RunArtifact(BaseModel):
    """Minimal schema for generated results.json artifacts."""

    model_config = ConfigDict(extra="allow")

    metadata: dict[str, Any]
    summary: dict[str, Any]
    category_breakdown: dict[str, dict[str, Any]] = Field(default_factory=dict)
    passed: bool
    gate: dict[str, Any]
    results: list[dict[str, Any]]
    target: str


def schema_bundle() -> dict[str, Any]:
    """Return JSON schemas for user-authored suites and generated run artifacts."""

    return {
        "schema_version": SCHEMA_VERSION,
        "eval_case": EvalCase.model_json_schema(),
        "eval_suite": EvalSuite.model_json_schema(),
        "run_artifact": RunArtifact.model_json_schema(),
    }


def write_schema_bundle(path: str | Path) -> None:
    Path(path).write_text(json.dumps(schema_bundle(), indent=2) + "\n")


def validate_run_artifact(payload: dict[str, Any]) -> RunArtifact:
    return RunArtifact(**payload)
