"""Shared helpers for the benchmark scripts. Nothing here contacts a network or target."""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SUITE = ROOT / "evals" / "core.yaml"
DEFAULT_MANIFEST = ROOT / "evals" / "evidence" / "manifest.yaml"
DEFAULT_API = "http://127.0.0.1:8766"

# Equity Copilot configuration used for the September 2026 runs. serve.py sets these
# environment variables; ingest.py and run.py record the same constants.
COPILOT_PROVIDER = "local"
COPILOT_MODEL = "local-deterministic-grounded-v1"
COPILOT_SETTINGS = {
    "embedding_dimensions": 128,
    "chunk_target_tokens": 800,
    "chunk_overlap_tokens": 80,
    "retrieval_min_score": 0.04,
    "max_upload_mb": 10,
}
COMPANY_NAMES = {
    "NVDA": "NVIDIA Corporation",
    "AAPL": "Apple Inc.",
    "MSFT": "Microsoft Corporation",
    "JPM": "JPMorgan Chase & Co.",
    "XOM": "Exxon Mobil Corporation",
    "TSLA": "Tesla, Inc.",
}
DOCUMENT_TYPES = {"10-K": "10-k", "10-Q": "10-q", "earnings_call": "earnings_transcript", "earnings_release": "8-k"}
EXTRACTION_METHODS = {
    ".html": "sec.html_to_text, complete text without SEC download truncation",
    ".pdf": "raw PDF through production parse_document",
}
HEX_SHA_RE = re.compile(r"[0-9a-f]{40,64}")
IDENTIFIER_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._:-]{0,127}")


def is_hex_digest(value: Any, lengths: tuple[int, ...] = (64,)) -> bool:
    return isinstance(value, str) and len(value) in lengths and bool(re.fullmatch(r"[0-9a-f]+", value))


def is_count(value: Any) -> bool:
    return type(value) is int and value >= 0


def is_timestamp(value: Any) -> bool:
    """A parseable, timezone-bearing ISO-8601 timestamp."""
    from datetime import datetime

    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})", value) or len(value) > 40:
        return False
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).tzinfo is not None
    except ValueError:
        return False


def is_date(value: Any) -> bool:
    from datetime import date

    if not isinstance(value, str) or not re.fullmatch(r"\d{4}-\d{2}-\d{2}", value):
        return False
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


def is_public_source_url(url: Any) -> bool:
    """A credential-free public https document URL: named host, no port, query or fragment."""
    import ipaddress
    from urllib.parse import urlsplit

    if not isinstance(url, str) or len(url) > 500 or any(c.isspace() or ord(c) < 32 for c in url):
        return False
    try:
        parts = urlsplit(url)
        port = parts.port
    except ValueError:
        return False
    host = parts.hostname or ""
    # Ordinary ASCII DNS labels, with a letter-led final label. Reject terminal
    # root dots (local-suffix bypasses) and legacy numeric IPv4 host spellings.
    if len(host) > 253 or not re.fullmatch(r"(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z](?:[a-z0-9-]{0,61}[a-z0-9])?", host):
        return False
    try:
        ipaddress.ip_address(host)
        return False
    except ValueError:
        pass
    return (
        parts.scheme == "https"
        and "." in host
        and not host.endswith((".local", ".internal", ".localhost"))
        and port is None
        and "@" not in parts.netloc
        and not (parts.query or parts.fragment)
        and bool(re.fullmatch(r"[A-Za-z0-9._~/%-]*", parts.path))
    )


def is_loopback_url(url: Any) -> bool:
    """An unauthenticated http loopback base URL with no path, query or fragment."""
    from urllib.parse import urlsplit

    if not isinstance(url, str):
        return False
    try:
        parts = urlsplit(url)
        port_ok = parts.port is None or 0 < parts.port < 65536
    except ValueError:
        return False
    return (
        parts.scheme == "http"
        and parts.hostname in {"127.0.0.1", "localhost", "::1"}
        and port_ok
        and not (parts.username or parts.password or parts.query or parts.fragment)
        and parts.path in {"", "/"}
    )

LOCAL_PATH_RE = re.compile(r"(?:/Users/|/home/|/private/|/var/folders/|/tmp/|/root/|/Volumes/|/mnt/|/opt/|/srv/|file://|\b[A-Za-z]:\\{1,2})")


def checked_fields(value: Any, allowed: set[str], label: str) -> dict[str, Any]:
    """Reject unclassified container fields rather than shallow-copy arbitrary content."""
    if not isinstance(value, dict):
        raise BenchmarkError(f"{label} must be an object")
    unknown = sorted(set(value) - allowed)
    if unknown:
        raise BenchmarkError(f"Unclassified {label} fields: " + ", ".join(unknown))
    return dict(value)


def checked_settings(value: Any) -> dict[str, Any]:
    settings = checked_fields(value, set(COPILOT_SETTINGS), "settings")
    if settings != COPILOT_SETTINGS or any(type(settings[key]) is not type(expected) for key, expected in COPILOT_SETTINGS.items()):
        raise BenchmarkError("Settings differ from the reviewed benchmark configuration")
    return settings


class BenchmarkError(RuntimeError):
    """A reproducibility or integrity check failed; no result should be trusted."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    """Canonical sorted-JSON hash, as recorded for provenance and document maps."""
    return sha256_bytes(json.dumps(value, sort_keys=True).encode())


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")


def require_new_directory(path: Path) -> Path:
    """Create a directory that did not exist; evidence directories are never reused."""
    path = Path(path)
    if path.exists():
        raise BenchmarkError(f"{path} already exists; choose a new directory so earlier evidence stays immutable")
    path.mkdir(parents=True)
    return path


def load_manifest(path: Path) -> dict[str, Any]:
    manifest = yaml.safe_load(Path(path).read_text())
    if not isinstance(manifest, dict) or not manifest.get("documents") or not manifest.get("evidence"):
        raise BenchmarkError(f"{path} is not an evidence manifest with documents and evidence")
    return manifest


def document_ticker(document_id: str) -> str:
    return document_id.split("_", 1)[0].upper()


def normalize_html(raw: bytes) -> str:
    """Pinned-manifest normalization: BeautifulSoup html.parser, no script/style, collapsed whitespace."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(raw.decode("utf-8"), "html.parser")
    for element in soup(["script", "style"]):
        element.decompose()
    return re.sub(r"\s+", " ", soup.get_text(" ")).strip()


def normalize_pdf(raw: bytes) -> str:
    """Pinned-manifest normalization: PyMuPDF page text in order, collapsed whitespace."""
    import fitz

    with fitz.open(stream=raw, filetype="pdf") as pdf:
        text = " ".join(page.get_text() for page in pdf)
    return re.sub(r"\s+", " ", text).strip()


def git_revision(path: Path) -> dict[str, Any]:
    """Commit and cleanliness only; file names from ``git status`` are not recorded."""

    def git(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(path), *args], text=True).strip()

    status = git("status", "--porcelain")
    return {"head": git("rev-parse", "HEAD"), "clean": not status, "changed_paths": len(status.splitlines())}


def find_local_paths(text: str, extra: list[str] | None = None) -> list[str]:
    hits = [match.group(0) for match in LOCAL_PATH_RE.finditer(text)]
    # Short values such as "/" or "/c" would match ordinary URLs; generic patterns cover those roots.
    hits += [value for value in (extra or []) if len(value) >= 8 and value in text]
    return hits
