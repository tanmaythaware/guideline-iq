"""
Lightweight types and helpers used across the RAG core.
"""

from __future__ import annotations

from typing import Dict, List


def format_sources_for_prompt(sources: List[Dict]) -> str:
    lines: list[str] = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s['id']}\n{s['text']}")
    return "\n\n".join(lines)


__all__ = ["format_sources_for_prompt"]

