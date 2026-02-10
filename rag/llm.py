"""
OpenAI LLM and embedding helpers for Guideline IQ.

Behaviour (models, retries, timeouts, and failure handling) is preserved.
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Callable, Dict, List, TypeVar

from dotenv import load_dotenv
from openai import APIConnectionError, APIError, APITimeoutError, OpenAI

from .prompts import SYSTEM_PROMPT, build_user_prompt, CLASSIFIER_PROMPT


logger = logging.getLogger("guideline_iq.rag")

# Load environment variables for OPENAI_API_KEY.
load_dotenv()


EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"
CLASSIFIER_MODEL_NAME = "gpt-4o-mini"


# OpenAI client initialized once, with request timeout.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)


# Retry configuration.
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds


T = TypeVar("T")


class SafeFailureError(Exception):
    """Raised when all retries are exhausted for OpenAI API calls."""


def with_retry(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = MAX_RETRIES,
) -> T:
    last_exception: Exception | None = None

    for attempt in range(max_retries):
        try:
            return func()
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2**attempt)
                logger.warning(
                    "%s failed (attempt %d/%d): %s. Retrying in %.1fs...",
                    operation_name,
                    attempt + 1,
                    max_retries,
                    str(exc),
                    delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s failed after %d attempts: %s",
                    operation_name,
                    max_retries,
                    str(exc),
                )
        except Exception as exc:
            logger.error(
                "%s failed with non-retryable error: %s", operation_name, str(exc)
            )
            raise

    raise SafeFailureError(
        f"{operation_name} failed after {max_retries} attempts: {str(last_exception)}"
    )


def embed_texts(texts: List[str]):
    logger.info("Embedding %d texts.", len(texts))

    def _call():
        return client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)

    return with_retry(_call, f"embed_texts({len(texts)} texts)")


def generate_answer(query: str, sources: list[dict]) -> str:
    from .types import format_sources_for_prompt  # avoid circular import

    sources_text = format_sources_for_prompt(sources)

    logger.info("Generating answer with %d sources.", len(sources))

    def _call():
        return client.chat.completions.create(
            model=GENERATION_MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_user_prompt(query, sources_text)},
            ],
            temperature=0.2,
        )

    resp = with_retry(_call, "generate_answer")
    return resp.choices[0].message.content.strip()


def classify_query_domain(query: str) -> Dict:

    try:
        logger.info("Classifying query domain.")

        def _call():
            return client.chat.completions.create(
                model=CLASSIFIER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": CLASSIFIER_PROMPT},
                    {"role": "user", "content": query},
                ],
                temperature=0.0,
            )

        resp = with_retry(_call, "classify_query_domain")
        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)
        label = parsed.get("label", "unsure")
        confidence = float(parsed.get("confidence", 0.0))
        return {"label": label, "confidence": confidence, "raw": content}
    except SafeFailureError:
        raise
    except Exception as exc:
        logger.exception("Query domain classification failed (non-retryable): %s", exc)
        return {"label": "unsure", "confidence": 0.0, "raw": None}


__all__ = ["SafeFailureError", "embed_texts", "generate_answer", "classify_query_domain"]

