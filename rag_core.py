import os
import json
import logging
from typing import List, Dict

import numpy as np
from openai import OpenAI

from dotenv import load_dotenv

# load dotenv variables for OpenAI API key
load_dotenv()

# Module logger for RAG core steps (embeddings, retrieval, generation).
logger = logging.getLogger("guideline_iq.rag")

EMBEDDING_MODEL_NAME = "text-embedding-3-small"
GENERATION_MODEL_NAME = "gpt-4o-mini"
CLASSIFIER_MODEL_NAME = "gpt-4o-mini"


# OpenAI client initialized once.
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embed_texts(texts: List[str]) -> np.ndarray:
    """Embed texts with OpenAI and return a 2D numpy array of vectors."""
    # Keep embeddings batched to reduce API calls.
    logger.info("Embedding %d texts.", len(texts))
    # OpenAI returns embeddings in the same order as inputs.
    resp = client.embeddings.create(model=EMBEDDING_MODEL_NAME, input=texts)
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
    resp = client.chat.completions.create(
        model=GENERATION_MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": build_user_prompt(query, sources_text)},
        ],
        temperature=0.2,
    )

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
        resp = client.chat.completions.create(
            model=CLASSIFIER_MODEL_NAME,
            messages=[
                {"role": "system", "content": system_msg},
                {"role": "user", "content": query},
            ],
            temperature=0.0,
        )
        content = resp.choices[0].message.content.strip()
        parsed = json.loads(content)
        label = parsed.get("label", "unsure")
        confidence = float(parsed.get("confidence", 0.0))
        return {"label": label, "confidence": confidence, "raw": content}
    except Exception as exc:
        # On any failure, fall back to an uncertain, non-finance classification.
        logger.exception("Query domain classification failed: %s", exc)
        return {"label": "unsure", "confidence": 0.0, "raw": None}
