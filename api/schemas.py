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


class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class UsageInfo(BaseModel):
    classifier: TokenUsage
    retrieval_embedding: TokenUsage
    generation: Optional[TokenUsage] = None
    total_tokens: int = 0
    estimated_cost_usd: float = 0.0


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
    usage: Optional[UsageInfo] = None


__all__ = [
    "Citation",
    "RetrievalResult",
    "RetrievalInfo",
    "AskRequest",
    "AskResponse",
]

