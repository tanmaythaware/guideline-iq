"""
Application settings for Guideline IQ.

Centralizes environment variables and policy thresholds so that the rest of
the codebase can import from here without changing default values.
"""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv


# Load env vars once at import time (matches prior behaviour).
load_dotenv()


# OpenAI configuration.
OPENAI_API_KEY: str | None = os.getenv("OPENAI_API_KEY")


# Data and logging paths.
DATA_PATH: Path = Path(os.getenv("DATA_PATH", "data/finance_guidelines.jsonl"))
LOG_DIR: Path = Path(os.getenv("LOG_DIR", "logs"))


# Guardrail thresholds (defaults unchanged).
DOMAIN_CONF_THRESHOLD: float = float(os.getenv("DOMAIN_CONF_THRESHOLD", "0.6"))
RETRIEVAL_SCORE_THRESHOLD: float = float(os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.6"))


# Retrieval configuration.
TOP_K: int = int(os.getenv("TOP_K", "2"))


# Admin configuration.
ADMIN_TOKEN: str = os.getenv("ADMIN_TOKEN", "")


# Abuse protection configuration.
API_ACCESS_KEY: str = os.getenv("API_ACCESS_KEY", "")
ASK_RATE_LIMIT: str = os.getenv("ASK_RATE_LIMIT", "10/minute")

