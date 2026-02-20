"""
Retrieval core for Guideline IQ.
"""

from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np

from .llm import embed_texts


logger = logging.getLogger("guideline_iq.rag")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._docs: List[Dict[str, str]] = []
        self._embeddings: List[np.ndarray] = []

    def add(self, docs: List[Dict[str, str]]) -> None:
        if not docs:
            return
        texts = [d["text"] for d in docs]
        resp, _ = embed_texts(texts)
        vectors = [item.embedding for item in resp.data]
        for doc, vec in zip(docs, vectors):
            self._docs.append({"id": doc["id"], "text": doc["text"]})
            self._embeddings.append(np.array(vec, dtype=np.float32))

    def query(self, q: str, top_k: int = 1) -> tuple[List[Dict[str, str]], int]:
        if not self._docs:
            return [], 0
        resp, tokens = embed_texts([q])
        q_vec = np.array(resp.data[0].embedding, dtype=np.float32)
        scored: list[Dict[str, str]] = []
        for doc, vec in zip(self._docs, self._embeddings):
            score = cosine_similarity(q_vec, vec)
            scored.append({"id": doc["id"], "text": doc["text"], "score": score})
        logger.info("Scored %d docs for query.", len(scored))
        scored.sort(key=lambda x: x["score"], reverse=True)
        return scored[:top_k], tokens


__all__ = ["InMemoryVectorStore"]

