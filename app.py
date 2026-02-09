from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ingest import docs
from rag_core import InMemoryVectorStore, generate_answer, classify_query_domain
import json
import time
import uuid
from pathlib import Path
from prompts import REFUSAL_TEXT


load_dotenv()

# Configure simple, readable logs for local development.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("guideline_iq")

store = InMemoryVectorStore()
startup_error: str | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    """Initialize embeddings once at startup; keep app alive on failure."""
    global startup_error
    try:
        # Precompute embeddings for the static guideline docs.
        logger.info("Startup: embedding %d guideline docs.", len(docs))
        store.add(docs)
        logger.info("Startup: embeddings ready.")
    except Exception as exc:
        # Avoid crashing the app; surface a clear message on /ask.
        startup_error = str(exc)
        logger.exception("Startup: embedding failed.")
    yield


app = FastAPI(lifespan=lifespan)

LOG_DIR = Path("logs")
REFUSAL_LOG = LOG_DIR / "refusals.jsonl"
ALL_LOG = LOG_DIR / "responses.jsonl"

# Policy thresholds for guardrails.
DOMAIN_CONF_THRESHOLD = 0.6
RETRIEVAL_SCORE_THRESHOLD = 0.6


def log_refusal(
    request_id: str,
    query: str,
    sources: list[dict],
    reason: str,
    *,
    classifier: dict | None = None,
    top_score: float | None = None,
) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        entry = {
            "event": "refusal",
            "request_id": request_id,
            "query": query,
            "retrieved": [
                {"id": s["id"], "score": float(s.get("score", 0.0))}
                for s in sources
            ],
            "reason": reason,
            "classifier": classifier,
            "top_score": float(top_score) if top_score is not None else None,
            "ts": time.time(),
        }
        with REFUSAL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # keep the request from failing if logging fails
        logger.exception("Failed to log refusal: %s", exc)

def log_response(payload: dict) -> None:
    try:
        LOG_DIR.mkdir(exist_ok=True)
        entry = dict(payload)
        entry["ts"] = time.time()
        with ALL_LOG.open("a") as f:
            f.write(json.dumps(entry) + "\n")
    except Exception as exc:  # keep the request from failing if logging fails
        logger.exception("Failed to log response: %s", exc)

def score_to_confidence(score: float) -> str:
    if score >= 0.6:
        return "high"
    if score >= 0.35:
        return "medium"
    return "low"


@app.get("/ask")
def ask(q: str):
    request_id = str(uuid.uuid4())
    logger.info("request_id=%s event=request_start q=%s", request_id, q)

    if startup_error:
        logger.error("request_id=%s embeddings unavailable: %s", request_id, startup_error)
        raise HTTPException(
            status_code=503,
            detail="Embeddings unavailable during startup. Check OPENAI_API_KEY and restart.",
        )

    # 1) Domain classification guardrail: is this even a finance/compliance question?
    cls_start = time.perf_counter()
    classifier = classify_query_domain(q)
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
        # Hard refusal: out-of-domain or low classifier confidence.
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

        response = {
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
        log_response(response)
        return response

    # 2) Retrieval guardrail: only proceed to generation if we have a strong match.
    retrieval_start = time.perf_counter()
    sources = store.query(q, top_k=2)
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
            # Too weak a match: refuse before any generation to save cost and avoid stretching sources.
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
            # 3) Generation: we have an in-domain query AND strong retrieval.
            gen_start = time.perf_counter()
            answer = generate_answer(q, sources)
            gen_ms = (time.perf_counter() - gen_start) * 1000
            logger.info("request_id=%s event=generation_end ms=%.1f", request_id, gen_ms)

            decision = "answer"
            reason = None

            if REFUSAL_TEXT in answer:
                # LLM itself refused given the provided sources.
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
                    {"id": s["id"], "text": s["text"], "score": float(s["score"])}
                    for s in sources
                ]

    response = {
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
            "top_k": 2,
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
    log_response(response)
    return response