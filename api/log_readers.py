"""
Helpers for reading recent JSONL logs.
"""

from __future__ import annotations

import json
from pathlib import Path


def tail_jsonl(path: str, limit: int = 50) -> list[dict]:
    """
    Return newest-first JSON objects from a JSONL file.

    - Returns [] if file doesn't exist.
    - Ignores malformed JSON lines.
    """
    file_path = Path(path)
    if not file_path.exists():
        return []

    try:
        lines = file_path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []

    items: list[dict] = []
    for raw in reversed(lines):
        if len(items) >= limit:
            break
        try:
            parsed = json.loads(raw)
        except Exception:
            continue
        if isinstance(parsed, dict):
            items.append(parsed)
    return items

