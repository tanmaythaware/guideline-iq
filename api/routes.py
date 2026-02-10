"""
FastAPI routes for Guideline IQ.

The /ask implementation is structurally identical to the existing logic:
- domain classification guardrail
- retrieval with embeddings + cosine similarity
- grounded generation
- strict refusals (no citations on refusals)
- JSONL logging for auditability
"""

from __future__ import annotations

from contextlib import asynccontextmanager
import logging
import time
import uuid
from typing import Optional

from fastapi import APIRouter, FastAPI, HTTPException, Query

from ingest.loader import load_docs
from rag.core import InMemoryVectorStore
from rag.llm import SafeFailureError, classify_query_domain, generate_answer
from rag.prompts import REFUSAL_TEXT, SAFE_FAILURE_TEXT

from .logging_utils import log_refusal, log_response
from .schemas import AskResponse
from .settings import (
    DATA_PATH,
    DOMAIN_CONF_THRESHOLD,
    RETRIEVAL_SCORE_THRESHOLD,
    TOP_K,
)


logger = logging.getLogger("guideline_iq")

router = APIRouter()

store = InMemoryVectorStore()
startup_error: Optional[str] = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize embeddings once at startup; keep app alive on failure."""
    global startup_error
    try:
        docs = load_docs(DATA_PATH)
        logger.info("Startup: embedding %d guideline docs.", len(docs))
        store.add(docs)
        logger.info("Startup: embeddings ready.")
    except Exception as exc:
        startup_error = str(exc)
        logger.exception("Startup: embedding failed.")
    yield


def score_to_confidence(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


def _safe_failure_response(
    request_id: str,
    query: str,
    failure_reason: str,
    *,
    classifier: dict | None = None,
    sources: list[dict] | None = None,
) -> AskResponse:
    sources = sources or []
    top_score = float(sources[0]["score"]) if sources else 0.0

    log_refusal(
        request_id,
        query,
        sources,
        failure_reason,
        classifier=classifier,
        top_score=top_score if sources else None,
    )

    payload = {
        "request_id": request_id,
        "query": query,
        "answer": SAFE_FAILURE_TEXT,
        "decision": "refuse",
        "confidence": "low",
        "citations": [],
        "classifier": classifier,
        "policy": {
            "domain_conf_threshold": DOMAIN_CONF_THRESHOLD,
            "retrieval_score_threshold": RETRIEVAL_SCORE_THRESHOLD,
        },
        "refusal_reason": failure_reason,
        "retrieval": {
            "top_k": len(sources),
            "results": [
                {"id": s["id"], "score": float(s.get("score", 0.0))}
                for s in sources
            ],
            "duration_ms": 0.0,
        },
    }

    logger.info(
        "request_id=%s event=request_end decision=refuse reason=%s",
        request_id,
        failure_reason,
    )
    log_response(payload)
    return AskResponse(**payload)


@router.get("/health")
def health():
    return {"status": "ok"}


@router.get("/ready")
def ready():
    if startup_error:
        return {"status": "error", "detail": startup_error}
    return {"status": "ok"}


@router.get("/ask", response_model=AskResponse)
def ask(q: str = Query(..., description="User question")):
    request_id = str(uuid.uuid4())
    logger.info("request_id=%s event=request_start q=%s", request_id, q)

    if startup_error:
        logger.error(
            "request_id=%s embeddings unavailable: %s", request_id, startup_error
        )
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings unavailable during startup. "
                "Check OPENAI_API_KEY and restart."
            ),
        )

    # 1) Domain classification guardrail.
    cls_start = time.perf_counter()
    try:
        classifier = classify_query_domain(q)
    except SafeFailureError as exc:
        logger.error(
            "request_id=%s classification failed after retries: %s", request_id, exc
        )
        return _safe_failure_response(request_id, q, "classification_failure")
    cls_ms = (time.perf_counter() - cls_start) * 1000

    classifier_label = classifier.get("label", "unsure")
    classifier_conf = float(classifier.get("confidence", 0.0))
    logger.info(
        "request_id=%s event=classification_end ms=%.1f label=%s confidence=%.2f",
        request_id,
        cls_ms,
        classifier_label,
        classifier_conf,
    )

    classifier_pass = (
        classifier_label == "finance" and classifier_conf >= DOMAIN_CONF_THRESHOLD
    )

    if not classifier_pass:
        sources: list[dict] = []
        retrieval_ms = 0.0
        top_score = 0.0
        answer = REFUSAL_TEXT
        decision = "refuse"
        confidence = "low"
        citations: list[dict] = []
        reason = "out_of_domain"

        log_refusal(
            request_id,
            q,
            sources,
            reason,
            classifier=classifier,
            top_score=top_score,
        )

        payload = {
            "request_id": request_id,
            "query": q,
            "answer": answer,
            "decision": decision,
            "confidence": confidence,
            "citations": citations,
            "classifier": classifier,
            "policy": {
                "domain_conf_threshold": DOMAIN_CONF_THRESHOLD,
                "retrieval_score_threshold": RETRIEVAL_SCORE_THRESHOLD,
            },
            "refusal_reason": reason,
            "retrieval": {
                "top_k": 0,
                "results": [],
                "duration_ms": retrieval_ms,
            },
        }

        logger.info(
            "request_id=%s event=request_end decision=%s confidence=%s top_score=%.4f",
            request_id,
            decision,
            confidence,
            top_score,
        )
        log_response(payload)
        return AskResponse(**payload)

    # 2) Retrieval guardrail.
    retrieval_start = time.perf_counter()
    try:
        sources = store.query(q, top_k=TOP_K)
    except SafeFailureError as exc:
        logger.error(
            "request_id=%s retrieval failed after retries: %s", request_id, exc
        )
        return _safe_failure_response(
            request_id, q, "retrieval_failure", classifier=classifier
        )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    logger.info(
        "request_id=%s event=retrieval_end ms=%.1f results=%d",
        request_id,
        retrieval_ms,
        len(sources),
    )

    if not sources:
        answer = REFUSAL_TEXT
        decision = "refuse"
        confidence = "low"
        citations: list[dict] = []
        top_score = 0.0
        reason = "no_relevant_docs"
        log_refusal(
            request_id,
            q,
            sources,
            reason,
            classifier=classifier,
            top_score=top_score,
        )
    else:
        top_score = float(sources[0]["score"])
        if top_score < RETRIEVAL_SCORE_THRESHOLD:
            answer = REFUSAL_TEXT
            decision = "refuse"
            confidence = "low"
            citations = []
            reason = "low_retrieval_score_pre_generation"
            log_refusal(
                request_id,
                q,
                sources,
                reason,
                classifier=classifier,
                top_score=top_score,
            )
        else:
            # 3) Generation.
            gen_start = time.perf_counter()
            try:
                answer = generate_answer(q, sources)
            except SafeFailureError as exc:
                logger.error(
                    "request_id=%s generation failed after retries: %s",
                    request_id,
                    exc,
                )
                return _safe_failure_response(
                    request_id,
                    q,
                    "generation_failure",
                    classifier=classifier,
                    sources=sources,
                )
            gen_ms = (time.perf_counter() - gen_start) * 1000
            logger.info(
                "request_id=%s event=generation_end ms=%.1f", request_id, gen_ms
            )

            decision = "answer"
            reason = None

            if REFUSAL_TEXT in answer:
                decision = "refuse"
                reason = "llm_refusal"
                confidence = "low"
                citations = []
                log_refusal(
                    request_id,
                    q,
                    sources,
                    reason,
                    classifier=classifier,
                    top_score=top_score,
                )
            else:
                confidence = score_to_confidence(top_score)
                citations = [
                    {
                        "index": i,
                        "id": s["id"],
                        "text": s["text"],
                        "score": float(s["score"]),
                    }
                    for i, s in enumerate(sources, start=1)
                ]

    payload = {
        "request_id": request_id,
        "query": q,
        "answer": answer,
        "decision": decision,
        "confidence": confidence,
        "citations": citations,
        "classifier": classifier,
        "policy": {
            "domain_conf_threshold": DOMAIN_CONF_THRESHOLD,
            "retrieval_score_threshold": RETRIEVAL_SCORE_THRESHOLD,
        },
        "refusal_reason": reason if decision == "refuse" else None,
        "retrieval": {
            "top_k": TOP_K,
            "results": [
                {"id": s["id"], "score": float(s.get("score", 0.0))}
                for s in sources
            ],
            "duration_ms": retrieval_ms,
        },
    }

    logger.info(
        "request_id=%s event=request_end decision=%s confidence=%s top_score=%.4f",
        request_id,
        decision,
        confidence,
        float(top_score),
    )
    log_response(payload)
    return AskResponse(**payload)


__all__ = ["router", "lifespan"]

