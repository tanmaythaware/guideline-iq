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

    resp = with_retry(_call, f"embed_texts({len(texts)} texts)")
    # Embeddings API returns usage in response.usage
    usage = getattr(resp, "usage", None)
    if usage:
        tokens = usage.total_tokens
    else:
        tokens = 0
    return resp, tokens


def generate_answer(query: str, sources: list[dict]) -> tuple[str, dict]:
    from .types import format_sources_for_prompt 

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
    usage = getattr(resp, "usage", None)
    if usage:
        token_usage = {
            "prompt_tokens": usage.prompt_tokens,
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        }
    else:
        token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
    return resp.choices[0].message.content.strip(), token_usage


def normalize_classifier_label(raw_label: str | None) -> str:
    if not raw_label:
        return "unsure"

    label = raw_label.strip().lower()

    finance_terms = {
        "finance",
        "financial",
        "financial regulation",
        "financial regulations",
        "financial promotions",
        "promotion compliance",
        "consumer duty",
        "risk warnings",
        "compliance",
        "regulatory",
    }

    non_finance_terms = {
        "non_finance",
        "non-finance",
        "general",
        "other",
        "out_of_domain",
        "out-of-domain",
    }

    if label in finance_terms:
        return "finance"
    if label in non_finance_terms:
        return "non_finance"

    # soft matching for common model phrasing
    if "financ" in label or "regulat" in label or "compliance" in label:
        return "finance"
    if "non" in label and "finance" in label:
        return "non_finance"

    return "unsure"


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
        # Normalize label to "finance", "non_finance", "unsure"
        label = normalize_classifier_label(parsed.get("label", "unsure"))
        confidence = float(parsed.get("confidence", 0.0))
        usage = getattr(resp, "usage", None)
        if usage:
            token_usage = {
                "prompt_tokens": usage.prompt_tokens,
                "completion_tokens": usage.completion_tokens,
                "total_tokens": usage.total_tokens,
            }
        else:
            token_usage = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
        return {"label": label, "confidence": confidence, "raw": content, "token_usage": token_usage}
    except SafeFailureError:
        raise
    except Exception as exc:
        logger.exception("Query domain classification failed (non-retryable): %s", exc)
        return {"label": "unsure", "confidence": 0.0, "raw": None}


__all__ = ["SafeFailureError", "embed_texts", "generate_answer", "classify_query_domain"]

