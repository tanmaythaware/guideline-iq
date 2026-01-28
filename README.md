# GuidelineIQ (Finance-first RAG Demo)

GuidelineIQ is a minimal, interview-ready Retrieval-Augmented Generation (RAG) system focused on grounded answers for finance/compliance questions. It retrieves relevant source text and generates answers constrained to those sources. If the sources are insufficient, it refuses with an explicit "I don't know based on the provided sources yet."

This repo is intentionally small and understandable: it demonstrates the full RAG loop end-to-end without unnecessary frameworks.

---

## What it does

**Query → Retrieve → Generate (grounded) → Return answer + sources**

- **Ingest**: Loads a small dataset of finance/compliance snippets (JSONL).
- **Embed**: Converts text into embeddings using OpenAI embeddings.
- **Retrieve**: Finds the most relevant sources via cosine similarity.
- **Generate**: Produces a concise answer using only retrieved sources.
- **Refuse**: If sources do not support the answer, responds:  
	"I don't know based on the provided sources yet."

---

## Why this exists (portfolio intent)

This project is designed to demonstrate:
- practical RAG architecture (retrieval separated from generation)
- grounding + refusal behavior (hallucination control)
- API design that returns auditable sources
- ability to ship a working demo quickly

---

## Tech stack

- Python
- FastAPI (API layer)
- OpenAI API:
	- embeddings: `text-embedding-3-small`
	- generation: `gpt-4o-mini`
- Streamlit (local demo UI)
- In-memory vector store (prototype backend)

---

## Project structure

- [app.py](app.py) — FastAPI app with `/ask`
- [rag_core.py](rag_core.py) — embeddings, similarity, vector store, generation
- [prompts.py](prompts.py) — grounding policy and prompt template
- [ingest.py](ingest.py) — loads dataset from JSONL into `docs`
- [data/finance_guidelines.jsonl](data/finance_guidelines.jsonl) — finance/compliance sources dataset
- [app_streamlit.py](app_streamlit.py) — Streamlit UI calling the FastAPI endpoint

---

## Auditing and refusals

- Every refusal (either no documents retrieved or the model responds with the refusal text) is logged to [logs/refusals.jsonl](logs/refusals.jsonl).
- Every /ask response (answer or refusal) is logged to [logs/responses.jsonl](logs/responses.jsonl) with request_id, query, decision, confidence, retrieval results (ids + scores), citations, answer, and timestamp.
- This makes it easy to review unanswered or low-confidence questions and decide which sources to add or update.
- The refusal text is consistent across prompts and API responses: "I don't know based on the provided sources yet."

---

## Setup

### 1) Create environment

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2) Configure secrets

Create a .env file in the repo root:

```bash
OPENAI_API_KEY=your_key_here
```

Important: Do not commit .env. Ensure it is listed in .gitignore.

---

## Run locally (demo mode)

Open two terminals.

Terminal A: start API

```bash
uvicorn app:app --reload
```

Terminal B: start UI

```bash
streamlit run app_streamlit.py
```

---

## Sample queries (for demo)

- In-scope answer: “What should a cryptoasset risk warning and promotion avoid or include for retail customers?”
- Out-of-scope refusal: “What is the exact FCA threshold for classifying a customer as high net worth?”