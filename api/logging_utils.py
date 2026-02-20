"""
JSONL logging helpers for Guideline IQ.

Log format and fields are preserved to keep behaviour/audit outputs unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any

from .settings import LOG_DIR


logger = logging.getLogger("guideline_iq")

REFUSAL_LOG = LOG_DIR / "refusals.jsonl"
ALL_LOG = LOG_DIR / "responses.jsonl"


def log_refusal(
    request_id: str,
    query: str,
    sources: list[dict],
    reason: str,
    *,
    classifier: dict | None = None,
    top_score: float | None = None,
    usage: dict | None = None,
) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        entry: dict[str, Any] = {
            "event": "refusal",
            "request_id": request_id,
            "query": query,
            "retrieved": [
                {"id": s["id"], "score": float(s.get("score", 0.0))}
                for s in sources
            ],
            "reason": reason,
            "classifier": classifier,
            "top_score": float(top_score) if top_score is not None else None,
            "ts": time.time(),
        }
        if usage:
            entry["usage"] = usage
        with REFUSAL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # don't fail requests if logging fails
        logger.exception("Failed to log refusal: %s", exc)


def log_response(payload: dict) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        entry = dict(payload)
        entry["ts"] = time.time()
        with ALL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # don't fail requests if logging fails
        logger.exception("Failed to log response: %s", exc)

