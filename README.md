# Guideline IQ — Finance-first RAG Demo

Guideline IQ is an **interview-ready** Retrieval-Augmented Generation (RAG) system for finance and compliance. It answers only when (1) the question is in scope, (2) retrieval finds strong evidence, and (3) the model can ground an answer in that evidence. Otherwise it **refuses** with clear, auditable reasons—no citations on refusals.

---

## What it does

**End-to-end flow:** Query → **Domain classifier** → **Retrieve** → **Generate (grounded)** → Answer or refusal + sources

- **Ingest**: Loads FCA-style guideline snippets from JSONL (`data/finance_guidelines.jsonl`).
- **Embed**: OpenAI `text-embedding-3-small`; embeddings built once at startup.
- **Retrieve**: Cosine similarity over in-memory vectors; returns top‑k chunks with scores.
- **Generate**: GPT‑4o‑mini answers **only** from retrieved source text; prompts enforce grounding and refusal when sources are insufficient.
- **Refuse**: Consistent refusal text when sources don’t support an answer: *"I don't know based on the provided sources yet."*

---

## Guardrails (why it’s strict and auditable)

### 1. Domain classification guardrail

Before any retrieval or generation, a small model call classifies the query as **finance** vs **non_finance** vs **unsure**, with a confidence score.

- **Pass**: `label == "finance"` and `confidence >= DOMAIN_CONF_THRESHOLD` (default `0.6`).
- **Fail**: Out-of-domain or low confidence → immediate refusal, no retrieval, no citations. Logged as `refusal_reason: "out_of_domain"`.

This keeps the system from answering off-topic questions (e.g. “What is earth?”) and gives a clear, auditable signal in logs.

### 2. Retrieval score threshold

Even if the query is in scope, we only call the generator when retrieval is strong.

- **Pass**: `top_score >= RETRIEVAL_SCORE_THRESHOLD` (default `0.6`).
- **Fail**: Low similarity → refusal **before** generation (saves cost, avoids stretching weak sources). Logged as `refusal_reason: "low_retrieval_score_pre_generation"`.

So we answer only when there is strong evidence in the guideline corpus.

### 3. LLM-level refusal

The generator is prompted to use **only** the provided source text. If it judges the sources insufficient, it responds with the exact refusal phrase above. We treat that as a refusal (`refusal_reason: "llm_refusal"`) and **do not** surface any citations.

### 4. Safe failure (timeout + retries)

All OpenAI calls (embeddings, classification, generation) are wrapped with:

- **Timeout**: 30 seconds per request.
- **Retries**: Up to 3 attempts with **exponential backoff** (1s, 2s, 4s) on transient errors (timeout, connection, rate limit).
- **After exhaustion**: We do **not** crash or return partial data. We return a safe refusal: *"I can't answer safely right now."* with `refusal_reason` set to `classification_failure`, `retrieval_failure`, or `generation_failure`, and log it like any other refusal.

So the system degrades gracefully under API issues and stays auditable.

---

## Auditing and logs

- **Refusals**: Every refusal is appended to `logs/refusals.jsonl` with `event`, `request_id`, `query`, `retrieved` (ids + scores), `reason`, `classifier` (label + confidence), `top_score`, and `ts`.
- **All responses**: Every `/ask` response is appended to `logs/responses.jsonl` with the full payload (answer, decision, confidence, citations, classifier, policy thresholds, refusal_reason, retrieval results, timestamp).

Refusal reasons you’ll see:

- `out_of_domain` — classifier didn’t pass.
- `no_relevant_docs` — no chunks retrieved.
- `low_retrieval_score_pre_generation` — retrieval score below threshold.
- `llm_refusal` — model said sources are insufficient.
- `classification_failure` / `retrieval_failure` / `generation_failure` — safe failure after retries.

This supports **auditability**, **strict refusals**, **low cost** (no generation on weak matches), and **knowledge expansion** (e.g. reviewing in-scope refusals to add or refine guidelines).

---

## Tech stack

- **Python**, **FastAPI** (API), **Streamlit** (demo UI)
- **OpenAI**: `text-embedding-3-small`, `gpt-4o-mini` (generation + domain classifier)
- **In-memory vector store** (cosine similarity, no DB)
- **JSONL** for knowledge base and logs

---

## Project structure

```
api/
  app.py              # FastAPI app + startup wiring
  routes.py           # /ask, /health, /ready
  schemas.py          # Pydantic request/response models
  settings.py         # env vars + thresholds
  logging_utils.py    # log_response / log_refusal

rag/
  core.py             # retrieval (vector store, cosine similarity)
  llm.py              # OpenAI: embeddings, generation, domain classifier, retry wrapper
  prompts.py          # system prompts + refusal text
  types.py            # helpers (e.g. format_sources_for_prompt)

ingest/
  loader.py           # load JSONL knowledge base from disk

evals/
  datasets/
    golden.jsonl
  run_eval.py
  metrics.py

ui/
  app_streamlit.py    # Streamlit UI (calls API)

data/
  finance_guidelines.jsonl

logs/
  refusals.jsonl
  responses.jsonl

requirements.txt
README.md
```

---

## Configuration (env)

Create a `.env` in the repo root. All except `OPENAI_API_KEY` have defaults.

| Variable | Default | Description |
|----------|---------|-------------|
| `OPENAI_API_KEY` | — | **Required.** OpenAI API key. |
| `DATA_PATH` | `data/finance_guidelines.jsonl` | Path to JSONL knowledge base. |
| `LOG_DIR` | `logs` | Directory for refusals and responses JSONL. |
| `DOMAIN_CONF_THRESHOLD` | `0.6` | Min classifier confidence to treat query as in-scope. |
| `RETRIEVAL_SCORE_THRESHOLD` | `0.6` | Min top retrieval score to call the generator. |
| `TOP_K` | `2` | Number of chunks retrieved per query. |
| `API_BASE_URL` | `http://localhost:8000` | Used by Streamlit to call the API. |

---

## Setup

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Add `.env` with at least:

```bash
OPENAI_API_KEY=your_key_here
```

---

## Run locally

**Terminal A — API**

```bash
uvicorn api.app:app --reload
```

**Terminal B — UI**

```bash
streamlit run ui/app_streamlit.py
```

- **API**: `http://localhost:8000`  
  - `GET /ask?q=...` — main Q&A endpoint  
  - `GET /health` — liveness  
  - `GET /ready` — readiness (checks embeddings loaded at startup)
- **UI**: Uses `API_BASE_URL` (default above) to call `/ask`.

---

## Sample queries

- **In-scope (answer with citations):** “What should a crypto risk warning include for retail customers?”
- **Out-of-scope (refusal, no citations):** “What is earth?” or “What is the capital of France?”
- **In-scope but weak match (refusal before generation):** A finance question phrased so that no guideline chunk scores above the threshold.

---

## Why this project (portfolio / interview)

- **Layered guardrails**: Domain classifier → retrieval threshold → LLM refusal. Clear separation of “in scope?” vs “enough evidence?” vs “can answer from sources?”
- **Safe failure**: Timeout + retries + explicit “I can’t answer safely right now.” so the system never pretends to answer when the API is failing.
- **Auditability**: Every decision path is logged with refusal reason, classifier output, and retrieval scores—suitable for compliance and improving the knowledge base.
- **Production-style layout**: Settings, routes, schemas, logging, and RAG logic split into focused modules without changing behaviour.
