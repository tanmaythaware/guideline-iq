"""
Pydantic models for the Guideline IQ API.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel


class Citation(BaseModel):
    index: Optional[int] = None
    id: str
    text: str
    score: float


class RetrievalResult(BaseModel):
    id: str
    score: float


class RetrievalInfo(BaseModel):
    top_k: int
    results: List[RetrievalResult]
    duration_ms: float


class AskRequest(BaseModel):
    q: str


class AskResponse(BaseModel):
    request_id: str
    query: str
    answer: str
    decision: str
    confidence: str
    citations: List[Citation]
    classifier: Optional[Dict[str, Any]] = None
    policy: Dict[str, Any]
    refusal_reason: Optional[str] = None
    retrieval: RetrievalInfo


__all__ = [
    "Citation",
    "RetrievalResult",
    "RetrievalInfo",
    "AskRequest",
    "AskResponse",
]

