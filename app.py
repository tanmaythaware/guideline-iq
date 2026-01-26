from contextlib import asynccontextmanager
import logging

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException

from ingest import docs
from rag_core import InMemoryVectorStore, generate_answer


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



@app.get("/ask")
def ask(q: str):
    """Handle /ask requests with retrieval + generation."""
    logger.info("Request: /ask q=%s", q)
    if startup_error:
        logger.error("Embeddings unavailable: %s", startup_error)
        raise HTTPException(
            status_code=503,
            detail=(
                "Embeddings unavailable during startup. "
                "Check OPENAI_API_KEY and billing/quota, then restart. "
                f"Error: {startup_error}"
            ),
        )
    sources = store.query(q, top_k=2)
    if not sources:
        logger.info("No documents available in store.")
        return {"query": q, "answer": "No documents loaded.", "sources": []}

    answer = generate_answer(q, sources)
    logger.info(
        "Top match id=%s score=%.4f",
        sources[0]["id"],
        sources[0]["score"],
    )

    return {
        "query": q,
        "answer": answer,
        "sources": sources,
    }
