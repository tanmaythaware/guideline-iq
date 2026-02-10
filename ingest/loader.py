"""
Knowledge base loader for Guideline IQ.

Importing this module has no side effects; callers explicitly invoke load_docs.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List


def load_docs(path: Path) -> List[Dict]:
    docs: list[dict] = []
    with path.open("r") as f:
        for line in f:
            line = line.strip()
            if line:
                docs.append(json.loads(line))
    return docs


__all__ = ["load_docs"]

