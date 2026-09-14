"""Fetch the exact pinned public sources into a new directory, failing closed on any drift.

Each raw file must match its manifest SHA-256 and its normalized text must match every
evidence entry's normalized-text hash. Nothing is written for a document until both
checks pass. Requires the pinned extraction libraries in benchmark/requirements.txt.
"""
from __future__ import annotations

import argparse
import re
import sys
import time
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Any

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.common import (  # noqa: E402
    DEFAULT_MANIFEST,
    BenchmarkError,
    load_manifest,
    normalize_html,
    normalize_pdf,
    require_new_directory,
    sha256_bytes,
    write_json,
)

USER_AGENT_RE = re.compile(r"\S+\s+\S*@\S+\.\S+")


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def fetch_sources(manifest: dict[str, Any], out: Path, client: httpx.Client, delay_s: float = 0.2) -> list[dict[str, Any]]:
    out = require_new_directory(out)
    records = []
    for document in manifest["documents"]:
        doc_id = document["document_id"]
        expected_text = {row["normalized_text_sha256"] for row in manifest["evidence"] if row["document_id"] == doc_id}
        if len(expected_text) != 1:
            raise BenchmarkError(f"{doc_id}: manifest needs exactly one normalized-text hash, found {len(expected_text)}")
        response = client.get(document["url"])
        response.raise_for_status()
        raw = response.content
        if sha256_bytes(raw) != document["sha256"]:
            raise BenchmarkError(
                f"Raw source drift for {doc_id}; obtain the pinned snapshot or deliberately curate a new suite version"
            )
        suffix = ".pdf" if raw.startswith(b"%PDF") else ".html"
        text = normalize_pdf(raw) if suffix == ".pdf" else normalize_html(raw)
        if {sha256_bytes(text.encode())} != expected_text:
            raise BenchmarkError(f"Normalization drift for {doc_id}; use the pinned extraction library versions")
        (out / f"{doc_id}{suffix}").write_bytes(raw)
        (out / f"{doc_id}.txt").write_text(text)
        records.append(
            {
                "document_id": doc_id,
                "url": document["url"],
                "sha256": document["sha256"],
                "normalized_text_sha256": next(iter(expected_text)),
                "retrieved_at": datetime.now(timezone.utc).isoformat(),
            }
        )
        print(doc_id, flush=True)
        time.sleep(delay_s)
    write_json(
        out / "fetch-record.json",
        {
            "manifest_retrieved_on": manifest.get("retrieved_on"),
            "libraries": {name: _version(name) for name in ("beautifulsoup4", "PyMuPDF", "httpx")},
            "documents": records,
        },
    )
    return records


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out", type=Path, required=True, help="New directory for raw and normalized sources")
    parser.add_argument(
        "--user-agent", required=True, help="Identifying 'Name email@example.com' User-Agent required by SEC fair access"
    )
    parser.add_argument("--delay", type=float, default=0.2, help="Seconds between requests")
    args = parser.parse_args(argv)
    if not USER_AGENT_RE.search(args.user_agent):
        parser.error("--user-agent must identify you with a name and contact email, e.g. 'Jane Doe jane@example.com'")
    manifest = load_manifest(args.manifest)
    with httpx.Client(headers={"User-Agent": args.user_agent}, timeout=60, follow_redirects=True) as client:
        fetch_sources(manifest, args.out, client, delay_s=args.delay)


if __name__ == "__main__":
    try:
        main()
    except BenchmarkError as exc:
        raise SystemExit(str(exc)) from exc
