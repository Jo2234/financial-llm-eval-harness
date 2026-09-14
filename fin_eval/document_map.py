"""Verified mapping from target document identifiers (for example API UUIDs) to suite document IDs.

A map is supplied explicitly by the caller, validated before any target request, and
applied only to the scorer-facing citation identity. Raw target responses are never
rewritten. A map verifies identity consistency; it does not prove that a target's
attribution is true. Strong provenance comes from recording the map from actual
upload responses after exact source-hash checks (see ``benchmark/ingest.py``).
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import Citation, EvalCase, TargetResponse

MAPPED = "mapped"
UNMAPPED = "unmapped"
MISSING_DOCUMENT_ID = "missing_document_id"


class DocumentMapError(ValueError):
    """Raised when a document map is malformed, ambiguous or inconsistent with a suite."""


def _reject_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise DocumentMapError(f"Duplicate key in document map: {key!r}")
        result[key] = value
    return result


def mapping_sha256(mapping: dict[str, str]) -> str:
    """Hash of canonical sorted JSON, matching the benchmark runner's ``document_map_sha256``."""
    return hashlib.sha256(json.dumps(mapping, sort_keys=True).encode()).hexdigest()


@dataclass(frozen=True)
class DocumentMap:
    mapping: dict[str, str]
    source: str | None = None
    file_sha256: str | None = None

    @property
    def sha256(self) -> str:
        return mapping_sha256(self.mapping)

    def provenance(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "file_sha256": self.file_sha256,
            "mapping_sha256": self.sha256,
            "entries": len(self.mapping),
            "canonical_documents": sorted(self.mapping.values()),
        }


def parse_document_map(text: str, source: str | None = None) -> DocumentMap:
    """Parse a JSON object of ``{target_id: suite_document_id}``.

    Also accepts an ingestion manifest containing ``document_uuid_to_canonical``.
    Duplicate keys, non-string or blank entries, surrounding whitespace, duplicate
    canonical targets and keys that are themselves canonical targets are rejected.
    """
    try:
        payload = json.loads(text, object_pairs_hook=_reject_duplicate_keys)
    except json.JSONDecodeError as exc:
        raise DocumentMapError(f"Document map is not valid JSON: {exc}") from exc
    if isinstance(payload, dict) and "document_uuid_to_canonical" in payload:
        payload = payload["document_uuid_to_canonical"]
    if not isinstance(payload, dict) or not payload:
        raise DocumentMapError("Document map must be a non-empty JSON object of target_id -> suite document_id")

    problems = []
    for key, value in payload.items():
        if not isinstance(value, str):
            problems.append(f"{key!r}: canonical document ID must be a string, got {type(value).__name__}")
            continue
        for label, item in (("target ID", key), ("canonical document ID", value)):
            if not item.strip():
                problems.append(f"{key!r}: {label} is blank")
            elif item != item.strip():
                problems.append(f"{key!r}: {label} has surrounding whitespace")
    if problems:
        raise DocumentMapError("Invalid document map entries:\n" + "\n".join(problems))

    seen: dict[str, str] = {}
    duplicates = []
    for key, value in payload.items():
        if value in seen:
            duplicates.append(f"{value!r} is mapped from both {seen[value]!r} and {key!r}")
        seen[value] = key
    if duplicates:
        raise DocumentMapError("Ambiguous document map; each suite document needs exactly one target ID:\n" + "\n".join(duplicates))
    overlapping = sorted(set(payload) & set(payload.values()))
    if overlapping:
        raise DocumentMapError("Document map keys must be target IDs, not suite document IDs: " + ", ".join(overlapping))
    return DocumentMap(mapping=dict(payload), source=source, file_sha256=hashlib.sha256(text.encode()).hexdigest())


def load_document_map(path: str | Path) -> DocumentMap:
    path = Path(path)
    try:
        text = path.read_text(encoding="utf-8")
    except OSError as exc:
        raise DocumentMapError(f"Cannot read document map {path}: {exc}") from exc
    return parse_document_map(text, source=str(path))


def suite_documents(cases: Iterable[EvalCase]) -> set[str]:
    """Every document identity a suite defines: case documents, citation rules and evidence."""
    documents: set[str] = set()
    for case in cases:
        documents.update(case.documents)
        documents.update(str(rule["document_id"]) for rule in case.required_citation_rules if rule.get("document_id"))
        documents.update(
            str(item["document_id"]) for item in case.source_evidence if isinstance(item, dict) and item.get("document_id")
        )
    return documents


def validate_document_map(
    document_map: DocumentMap,
    suite_cases: Iterable[EvalCase],
    selected_cases: Iterable[EvalCase],
    permitted_documents: Iterable[str] | None = None,
) -> dict[str, Any]:
    """Fail loudly unless every mapped target is a known suite document and selected cases are covered.

    ``permitted_documents`` (for example an evidence-manifest inventory) narrows the
    allowed canonical IDs further. Mapping extra corpus documents is allowed so that
    citations to legitimately wrong sources score as wrong rather than unverified.
    """
    # run_suite also accepts a directly constructed DocumentMap. Enforce the
    # parser's structural guarantees for that path, not just file-based maps.
    if not isinstance(document_map.mapping, dict) or any(not isinstance(key, str) or not isinstance(value, str) for key, value in document_map.mapping.items()):
        raise DocumentMapError("Document map entries must be string target IDs -> string canonical IDs")
    parse_document_map(json.dumps(document_map.mapping))
    known = suite_documents(suite_cases)
    required = suite_documents(selected_cases)
    if permitted_documents is not None:
        permitted = set(permitted_documents)
        outside = sorted(required - permitted)
        if outside:
            raise DocumentMapError("Selected suite documents are outside the permitted source corpus: " + ", ".join(outside))
        known &= permitted
    unknown = sorted(set(document_map.mapping.values()) - known)
    if unknown:
        raise DocumentMapError("Document map targets unknown suite documents: " + ", ".join(unknown))
    missing = sorted(required - set(document_map.mapping.values()))
    if missing:
        raise DocumentMapError("Document map does not cover documents required by selected cases: " + ", ".join(missing))
    return {**document_map.provenance(), "unmapped_suite_documents": sorted(known - set(document_map.mapping.values()))}


def _mapped_citation(citation: Citation, mapping: dict[str, str]) -> Citation:
    original = citation.document_id
    if not original:
        status, document_id = MISSING_DOCUMENT_ID, None
    elif original in mapping:
        status, document_id = MAPPED, mapping[original]
    else:
        status, document_id = UNMAPPED, original
    fields = citation.model_dump()
    fields.update(document_id=document_id, source_document_id=original, document_map_status=status)
    return Citation(**fields)


def apply_document_map(response: TargetResponse, document_map: DocumentMap | None) -> TargetResponse:
    """Return a response with scorer-facing citation identities canonicalized.

    ``raw_response`` is left untouched. Unknown IDs keep their original value and are
    marked ``unmapped``; nothing is inferred from labels, titles or excerpts.
    """
    if document_map is None:
        return response
    citations = [_mapped_citation(citation, document_map.mapping) for citation in response.citations]
    return response.model_copy(update={"citations": citations})
