import os
import json
import logging
import time
from typing import List, Dict, Callable, TypeVar

import numpy as np
from openai import OpenAI, APITimeoutError, APIError, APIConnectionError

from dotenv import load_dotenv

# load dotenv variables for OpenAI API key
load_dotenv()

# Module logger for RAG core steps (embeddings, retrieval, generation).
logger = logging.getLogger("guideline_iq.rag")

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"
CLASSIFIER_MODEL_NAME = "gpt-4o-mini"


# OpenAI client initialized once.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"), timeout=30.0)

# Retry configuration
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0  # seconds

T = TypeVar('T')


class SafeFailureError(Exception):
    """Raised when all retries are exhausted for OpenAI API calls."""
    pass


def with_retry(
    func: Callable[[], T],
    operation_name: str,
    max_retries: int = MAX_RETRIES,
) -> T:
    """
    Wrapper for OpenAI API calls with timeout, retry, and safe failure.
    
    Retries on transient errors (timeout, connection, rate limit).
    After max_retries, raises SafeFailureError.
    """
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            return func()
        except (APITimeoutError, APIConnectionError, APIError) as exc:
            last_exception = exc
            if attempt < max_retries - 1:
                delay = RETRY_BASE_DELAY * (2 ** attempt)  # exponential backoff
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
            # Non-API errors (e.g., JSON parsing) should not be retried
            logger.error("%s failed with non-retryable error: %s", operation_name, str(exc))
            raise
    
    # All retries exhausted
    raise SafeFailureError(
        f"{operation_name} failed after {max_retries} attempts: {str(last_exception)}"
    )

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts with OpenAI and return a 2D numpy array of vectors."""
    # Keep embeddings batched to reduce API calls.
    logger.info("Embedding %d texts.", len(texts))
    # OpenAI returns embeddings in the same order as inputs.
    def _call():
        return client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
    
    resp = with_retry(_call, f"embed_texts({len(texts)} texts)")
    vectors = [item.embedding for item in resp.data]
    return np.array(vectors, dtype=np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine similarity between two 1D vectors."""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class InMemoryVectorStore:
    def __init__(self) -> None:
        """Initialize empty in-memory storage for docs and embeddings."""
        self._docs: List[Dict[str, str]] = []
        self._embeddings: List[np.ndarray] = []

    def add(self, docs: List[Dict[str, str]]) -> None:
        """Embed each document text and store it alongside its metadata."""
        if not docs:
            return
        # Embed docs once and persist vectors in memory for fast retrieval.
        texts = [d["text"] for d in docs]
        vectors = embed_texts(texts)
        for doc, vec in zip(docs, vectors):
            self._docs.append({"id": doc["id"], "text": doc["text"]})
            self._embeddings.append(vec)

    def query(self, q: str, top_k: int = 1) -> List[Dict[str, str]]:
        """Embed query and return the top_k most similar documents."""
        if not self._docs:
            return []
        # Embed the query and score against each stored vector.
        q_vec = embed_texts([q])[0]
        scored = []
        for doc, vec in zip(self._docs, self._embeddings):
            score = cosine_similarity(q_vec, vec)
            scored.append({"id": doc["id"], "text": doc["text"], "score": score})
        logger.info("Scored %d docs for query.", len(scored))
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k]

def format_sources_for_prompt(sources: list[dict]) -> str:
    """
    Turn retrieved sources into a readable block for the LLM.
    """
    # Compact, numbered format makes source citation easy to follow.
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] {s['id']}\n{s['text']}")
    return "\n\n".join(lines)

def generate_answer(query: str, sources: list[dict]) -> str:
    """
    Use an LLM to generate an answer grounded ONLY in retrieved sources.
    """
    from prompts import SYSTEM_PROMPT, build_user_prompt

    # Build a strict, sources-only prompt to reduce hallucinations.
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
    """
    Classify whether a query is in the finance/compliance domain.

    Returns a dict like:
    {
        "label": "finance" | "non_finance" | "unsure",
        "confidence": float between 0 and 1,
        "raw": <original model output str>,
    }
    """
    system_msg = (
        "You are a domain classifier for a finance and compliance assistant.\n"
        "Decide if the user's question is primarily about finance, financial "
        "regulation, consumer duty, risk warnings, or financial promotions. "
        "If the question is about any of these topics, classify it as 'finance'. "
        "If the question is not about any of these topics, classify it as 'non_finance'. "
        "If you are unsure, classify it as 'unsure'.\n\n"
        "Respond ONLY with JSON in this exact format:\n"
        '{\"label\": \"finance\" | \"non_finance\" | \"unsure\", '
        '\"confidence\": <number between 0 and 1>}.\n'
        "Do not add any explanation or extra text."
    )

    try:
        logger.info("Classifying query domain.")
        
        def _call():
            return client.chat.completions.create(
                model=CLASSIFIER_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_msg},
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
        # Retries exhausted: propagate to caller for safe refusal handling
        raise
    except Exception as exc:
        # Non-retryable errors (e.g., JSON parsing): fall back to uncertain classification
        logger.exception("Query domain classification failed (non-retryable): %s", exc)
        return {"label": "unsure", "confidence": 0.0, "raw": None}
