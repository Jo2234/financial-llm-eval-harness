"""Serve an unmodified Equity Copilot checkout on loopback with a fresh, unseeded corpus.

The deterministic ``local`` provider is selected explicitly; no automatic provider
fallback, LLM or paid API is used. Provider API-key variables are removed from this
process environment before the app is imported.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from benchmark.common import (  # noqa: E402
    COPILOT_MODEL,
    COPILOT_PROVIDER,
    COPILOT_SETTINGS,
    git_revision,
    require_new_directory,
    write_json,
)

PROVIDER_KEY_VARIABLES = ("OPENAI_API_KEY", "ANTHROPIC_API_KEY", "AIERC_OPENAI_API_KEY", "AIERC_ANTHROPIC_API_KEY")


def copilot_environment(state: Path) -> dict[str, str]:
    return {
        # Importing the Copilot module builds a seeded default app; keep it out of the measured corpus.
        "AIERC_DATA_DIR": str(state / "import-time-demo"),
        "AIERC_LLM_PROVIDER": COPILOT_PROVIDER,
        "AIERC_LOCAL_MODEL_NAME": COPILOT_MODEL,
        "AIERC_DEMO_MODE": "false",
        "AIERC_EMBEDDING_DIMENSIONS": str(COPILOT_SETTINGS["embedding_dimensions"]),
        "AIERC_CHUNK_TARGET_TOKENS": str(COPILOT_SETTINGS["chunk_target_tokens"]),
        "AIERC_CHUNK_OVERLAP_TOKENS": str(COPILOT_SETTINGS["chunk_overlap_tokens"]),
        "AIERC_RETRIEVAL_MIN_SCORE": str(COPILOT_SETTINGS["retrieval_min_score"]),
        "AIERC_MAX_UPLOAD_MB": str(COPILOT_SETTINGS["max_upload_mb"]),
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--copilot", type=Path, required=True, help="ai-equity-research-copilot checkout")
    parser.add_argument("--state", type=Path, required=True, help="New directory for API state")
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args(argv)
    state = require_new_directory(args.state).resolve()
    copilot = args.copilot.resolve()
    for name in PROVIDER_KEY_VARIABLES:
        os.environ.pop(name, None)
    os.environ.update(copilot_environment(state))
    write_json(
        state / "serve-config.json",
        {
            "started_at": datetime.now(timezone.utc).isoformat(),
            "copilot": git_revision(copilot),
            "provider": COPILOT_PROVIDER,
            "model": COPILOT_MODEL,
            "settings": COPILOT_SETTINGS,
            "seed": False,
            "host": "127.0.0.1",
            "port": args.port,
        },
    )
    sys.path.insert(0, str(copilot / "backend"))
    import uvicorn
    from ai_equity_research_copilot_backend.main import create_app

    uvicorn.run(create_app(data_dir=state / "source-only", seed=False), host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
