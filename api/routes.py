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

from fastapi import APIRouter, FastAPI, HTTPException, Header, Query

from ingest.loader import load_docs
from rag.core import InMemoryVectorStore
from rag.llm import SafeFailureError, classify_query_domain, generate_answer
from rag.prompts import REFUSAL_TEXT, SAFE_FAILURE_TEXT

from .log_readers import tail_jsonl
from .logging_utils import log_refusal, log_response
from .schemas import AskResponse, TokenUsage, UsageInfo
from .settings import (
    ADMIN_TOKEN,
    DATA_PATH,
    DOMAIN_CONF_THRESHOLD,
    LOG_DIR,
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


def calculate_cost(
    classifier_tokens: int,
    embedding_tokens: int,
    generation_prompt_tokens: int = 0,
    generation_completion_tokens: int = 0,
) -> float:
    """
    Calculate estimated cost in USD based on OpenAI pricing (as of 2024).
    - text-embedding-3-small: $0.02 per 1M tokens
    - gpt-4o-mini: $0.15 per 1M input tokens, $0.60 per 1M output tokens
    """
    embedding_cost = (embedding_tokens / 1_000_000) * 0.02
    classifier_cost = (classifier_tokens / 1_000_000) * 0.15  # input only for classifier
    gen_input_cost = (generation_prompt_tokens / 1_000_000) * 0.15
    gen_output_cost = (generation_completion_tokens / 1_000_000) * 0.60
    return embedding_cost + classifier_cost + gen_input_cost + gen_output_cost


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

    # For safe failures, token usage depends on where it failed
    classifier_token_usage = TokenUsage(total_tokens=0)
    if classifier and "token_usage" in classifier:
        cls_usage = classifier.get("token_usage", {})
        classifier_token_usage = TokenUsage(
            prompt_tokens=cls_usage.get("prompt_tokens", 0),
            completion_tokens=cls_usage.get("completion_tokens", 0),
            total_tokens=cls_usage.get("total_tokens", 0),
        )
    
    retrieval_token_usage = TokenUsage(total_tokens=0)
    
    usage_info = UsageInfo(
        classifier=classifier_token_usage,
        retrieval_embedding=retrieval_token_usage,
        generation=None,
        total_tokens=classifier_token_usage.total_tokens,
        estimated_cost_usd=calculate_cost(classifier_token_usage.total_tokens, 0),
    )

    log_refusal(
        request_id,
        query,
        sources,
        failure_reason,
        classifier=classifier,
        top_score=top_score if sources else None,
        usage=usage_info.dict(),
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
        "usage": usage_info.dict(),
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


def _require_admin_token(x_admin_token: str | None) -> None:
    if not ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="ADMIN_TOKEN not set")
    if not x_admin_token or x_admin_token != ADMIN_TOKEN:
        raise HTTPException(status_code=401, detail="Unauthorized")


@router.get("/logs/refusals")
def get_refusal_logs(
    limit: int = Query(50),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    _require_admin_token(x_admin_token)
    return tail_jsonl(str(LOG_DIR / "refusals.jsonl"), limit=limit)


@router.get("/logs/responses")
def get_response_logs(
    limit: int = Query(50),
    x_admin_token: str | None = Header(default=None, alias="X-Admin-Token"),
):
    _require_admin_token(x_admin_token)
    return tail_jsonl(str(LOG_DIR / "responses.jsonl"), limit=limit)


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
    classifier_token_usage = classifier.get("token_usage", {})
    cls_tokens = TokenUsage(
        prompt_tokens=classifier_token_usage.get("prompt_tokens", 0),
        completion_tokens=classifier_token_usage.get("completion_tokens", 0),
        total_tokens=classifier_token_usage.get("total_tokens", 0),
    )
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
        retrieval_token_usage = TokenUsage(total_tokens=0)

        # Out-of-domain: only classifier tokens used
        total_tokens = cls_tokens.total_tokens
        estimated_cost = calculate_cost(cls_tokens.total_tokens, 0)
        usage_info = UsageInfo(
            classifier=cls_tokens,
            retrieval_embedding=retrieval_token_usage,
            generation=None,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )

        log_refusal(
            request_id,
            q,
            sources,
            reason,
            classifier=classifier,
            top_score=top_score,
            usage=usage_info.dict(),
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
            "usage": usage_info.dict(),
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
        sources, embedding_tokens = store.query(q, top_k=TOP_K)
    except SafeFailureError as exc:
        logger.error(
            "request_id=%s retrieval failed after retries: %s", request_id, exc
        )
        return _safe_failure_response(
            request_id, q, "retrieval_failure", classifier=classifier
        )
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000
    retrieval_token_usage = TokenUsage(total_tokens=embedding_tokens)
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
        # No relevant docs: classifier + retrieval embedding tokens
        total_tokens = cls_tokens.total_tokens + retrieval_token_usage.total_tokens
        estimated_cost = calculate_cost(
            cls_tokens.total_tokens,
            retrieval_token_usage.total_tokens,
        )
        usage_info = UsageInfo(
            classifier=cls_tokens,
            retrieval_embedding=retrieval_token_usage,
            generation=None,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )
        log_refusal(
            request_id,
            q,
            sources,
            reason,
            classifier=classifier,
            top_score=top_score,
            usage=usage_info.dict(),
        )
    else:
        top_score = float(sources[0]["score"])
        if top_score < RETRIEVAL_SCORE_THRESHOLD:
            answer = REFUSAL_TEXT
            decision = "refuse"
            confidence = "low"
            citations = []
            reason = "low_retrieval_score_pre_generation"
            # Low retrieval score: classifier + retrieval embedding tokens (no generation)
            total_tokens = cls_tokens.total_tokens + retrieval_token_usage.total_tokens
            estimated_cost = calculate_cost(
                cls_tokens.total_tokens,
                retrieval_token_usage.total_tokens,
            )
            usage_info = UsageInfo(
                classifier=cls_tokens,
                retrieval_embedding=retrieval_token_usage,
                generation=None,
                total_tokens=total_tokens,
                estimated_cost_usd=estimated_cost,
            )
            log_refusal(
                request_id,
                q,
                sources,
                reason,
                classifier=classifier,
                top_score=top_score,
                usage=usage_info.dict(),
            )
        else:
            # 3) Generation.
            gen_start = time.perf_counter()
            try:
                answer, gen_token_usage_dict = generate_answer(q, sources)
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
            gen_token_usage = TokenUsage(
                prompt_tokens=gen_token_usage_dict.get("prompt_tokens", 0),
                completion_tokens=gen_token_usage_dict.get("completion_tokens", 0),
                total_tokens=gen_token_usage_dict.get("total_tokens", 0),
            )
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
                # LLM refusal: all tokens used (classifier + retrieval + generation)
                total_tokens = (
                    cls_tokens.total_tokens
                    + retrieval_token_usage.total_tokens
                    + gen_token_usage.total_tokens
                )
                estimated_cost = calculate_cost(
                    cls_tokens.total_tokens,
                    retrieval_token_usage.total_tokens,
                    gen_token_usage.prompt_tokens,
                    gen_token_usage.completion_tokens,
                )
                usage_info = UsageInfo(
                    classifier=cls_tokens,
                    retrieval_embedding=retrieval_token_usage,
                    generation=gen_token_usage,
                    total_tokens=total_tokens,
                    estimated_cost_usd=estimated_cost,
                )
                log_refusal(
                    request_id,
                    q,
                    sources,
                    reason,
                    classifier=classifier,
                    top_score=top_score,
                    usage=usage_info.dict(),
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

    # Calculate token usage and cost
    if decision == "answer" and "gen_token_usage" in locals():
        total_tokens = (
            cls_tokens.total_tokens
            + retrieval_token_usage.total_tokens
            + gen_token_usage.total_tokens
        )
        estimated_cost = calculate_cost(
            cls_tokens.total_tokens,
            retrieval_token_usage.total_tokens,
            gen_token_usage.prompt_tokens,
            gen_token_usage.completion_tokens,
        )
        usage_info = UsageInfo(
            classifier=cls_tokens,
            retrieval_embedding=retrieval_token_usage,
            generation=gen_token_usage,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
        )
    else:
        # Refusal before generation - no generation tokens
        total_tokens = cls_tokens.total_tokens + retrieval_token_usage.total_tokens
        estimated_cost = calculate_cost(
            cls_tokens.total_tokens,
            retrieval_token_usage.total_tokens,
        )
        usage_info = UsageInfo(
            classifier=cls_tokens,
            retrieval_embedding=retrieval_token_usage,
            generation=None,
            total_tokens=total_tokens,
            estimated_cost_usd=estimated_cost,
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
        "refusal_reason": reason if decision == "refuse" else None,
        "retrieval": {
            "top_k": TOP_K,
            "results": [
                {"id": s["id"], "score": float(s.get("score", 0.0))}
                for s in sources
            ],
            "duration_ms": retrieval_ms,
        },
        "usage": usage_info.dict() if usage_info else None,
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

