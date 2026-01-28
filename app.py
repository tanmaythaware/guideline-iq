from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ingest import docs
from rag_core import InMemoryVectorStore, generate_answer
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

def log_refusal(request_id: str, query: str, sources: list[dict], reason: str) -> None:
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
        citations = []
        log_refusal(request_id, q, sources, "no_documents_loaded")
    else:
        gen_start = time.perf_counter()
        answer = generate_answer(q, sources)
        gen_ms = (time.perf_counter() - gen_start) * 1000
        logger.info("request_id=%s event=generation_end ms=%.1f", request_id, gen_ms)

        top_score = float(sources[0]["score"])
        decision = "answer"
        reason = None

        if (REFUSAL_TEXT in answer) or (top_score < 0.35):
            decision = "refuse"
            reason = "llm_refusal" if REFUSAL_TEXT in answer else "low_retrieval_score"
            log_refusal(request_id, q, sources, reason)

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
        float(sources[0]["score"]) if sources else 0.0,
    )
    log_response(response)
    return response