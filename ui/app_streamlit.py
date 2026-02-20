import os
import requests
import streamlit as st
from dotenv import load_dotenv

load_dotenv()
API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="GuidelineIQ", page_icon="🤖", layout="centered")

# ---------- Helpers ----------
def get_json(url, **kwargs):
    r = requests.get(url, **kwargs)
    r.raise_for_status()
    return r.json()

def health_ok():
    try:
        return get_json(f"{API_BASE_URL}/health", timeout=2).get("status") == "ok"
    except Exception:
        return False

REFUSAL_REASON_COPY = {
    "out_of_domain": "Out of scope (non-finance or low classifier confidence).",
    "no_relevant_docs": "No relevant guideline chunks were retrieved.",
    "low_retrieval_score_pre_generation": "Evidence was too weak (top score below threshold).",
    "llm_refusal": "Model refused because the provided sources were insufficient.",
    "classification_failure": "Temporary API failure during classification (safe failure).",
    "retrieval_failure": "Temporary API failure during retrieval (safe failure).",
    "generation_failure": "Temporary API failure during generation (safe failure).",
}

def chip(text: str):
    st.markdown(
        f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
        f"border:1px solid rgba(255,255,255,0.15);font-size:12px;'>{text}</span>",
        unsafe_allow_html=True,
    )

def fmt_pct(x: float) -> str:
    return f"{x*100:.0f}%"

# ---------- Header ----------
st.title("GuidelineIQ 🤖")
st.caption("Finance-first guardrailed RAG for regulated answers — grounded when evidence is strong, refusal when it isn’t.")

col1, col2, col3 = st.columns([1, 1, 3])
with col1:
    ok = health_ok()
    chip("API ✅ Online" if ok else "API ❌ Offline")
with col2:
    chip("Guardrails ✅ On")
with col3:
    st.caption("Citations appear only when an answer is supported by sources.")

with st.expander("How it works", expanded=False):
    st.markdown("""
**1) Domain gate** — finance/compliance only (classifier + threshold).  
**2) Evidence gate** — retrieval top score must clear a threshold.  
**3) Grounded generation** — answer must be supported by retrieved text.  
**4) Clean refusals** — no citations on refusals.  
""")

# ---------- Examples ----------
if "query_input" not in st.session_state:
    st.session_state.query_input = ""

EXAMPLE_QUERIES = [
    ("In-scope (answer)", "What should a crypto risk warning include for retail customers?"),
    ("Out-of-scope (refusal)", "What is the capital of France?"),
]
st.markdown("**Try these examples:**")
cols = st.columns(len(EXAMPLE_QUERIES))
for col, (label, query) in zip(cols, EXAMPLE_QUERIES):
    with col:
        if st.button(label, use_container_width=True):
            st.session_state.query_input = query
            st.rerun()

show_debug = st.toggle("Show debug / audit details", value=False)

# ---------- Form (Enter submits) ----------
with st.form("ask_form"):
    q = st.text_input(
        "Ask a finance or compliance question:",
        placeholder="e.g. What should a crypto risk warning include?",
        key="query_input",
    )
    submitted = st.form_submit_button("Ask")

# ---------- Call API ----------
if submitted and q.strip():
    if not ok:
        st.error("API looks offline. Check API_BASE_URL and that FastAPI is running.")
    else:
        try:
            with st.spinner("Running guardrails → retrieval → grounded generation…"):
                data = get_json(
                    f"{API_BASE_URL}/ask",
                    params={"q": q},
                    timeout=30,
                )

            decision = data.get("decision")
            confidence = data.get("confidence", "low")
            answer = data.get("answer", "")
            request_id = data.get("request_id", "")
            refusal_reason = data.get("refusal_reason")

            classifier = data.get("classifier") or {}
            policy = data.get("policy") or {}
            retrieval = data.get("retrieval") or {}

            domain_thr = float(policy.get("domain_conf_threshold", 0.0))
            retrieval_thr = float(policy.get("retrieval_score_threshold", 0.0))

            cls_label = classifier.get("label", "unsure")
            cls_conf = float(classifier.get("confidence", 0.0))

            results = retrieval.get("results") or []
            top_score = float(results[0]["score"]) if results else 0.0
            retrieval_ms = float(retrieval.get("duration_ms", 0.0))

            # ---------- Result header ----------
            st.markdown("### Result")
            a, b, c = st.columns([1.2, 1.2, 2.2])
            with a:
                is_answer = decision == "answer"
                if is_answer:
                    bg, border = "rgba(0,180,0,0.4)", "rgba(0,200,0,0.7)"
                else:
                    bg, border = "rgba(220,50,50,0.4)", "rgba(255,80,80,0.7)"
                label = "Answer" if is_answer else "Refusal"
                st.markdown(
                    f"<span style='display:inline-block;padding:4px 10px;border-radius:999px;"
                    f"background-color:{bg};border:1px solid {border};"
                    f"color:#fff;font-size:12px;font-weight:500;'>{label}</span>",
                    unsafe_allow_html=True,
                )
            with b:
                chip(f"Confidence: {confidence.upper()}")
            with c:
                st.caption(f"Request ID: `{request_id}`")

            # ---------- Refusal reason ----------
            if decision != "answer":
                friendly = REFUSAL_REASON_COPY.get(refusal_reason, refusal_reason or "Unknown refusal reason.")
                st.info(f"**Why refused:** {friendly}")

            # ---------- Answer ----------
            st.markdown("### Answer")
            st.write(answer if answer else "—")

            # ---------- Sources (only for answers) ----------
            if decision == "answer":
                citations = data.get("citations", [])
                st.markdown("### Sources")
                for s in citations:
                    idx = s.get("index")
                    sid = s.get("id", "source")
                    score = float(s.get("score", 0.0))
                    title = f"[{idx}] {sid} — score {score:.3f}"
                    with st.expander(title, expanded=False):
                        st.write(s.get("text", ""))

            # ---------- Debug / audit ----------
            if show_debug:
                st.markdown("### Debug / Audit")
                d1, d2, d3 = st.columns(3)

                with d1:
                    st.markdown("**Domain classifier**")
                    st.write(f"Label: `{cls_label}`")
                    st.write(f"Confidence: `{cls_conf:.2f}` (threshold `{domain_thr:.2f}`)")
                    passed = (cls_label == "finance" and cls_conf >= domain_thr)
                    st.write(f"Pass: {'✅' if passed else '❌'}")

                with d2:
                    st.markdown("**Retrieval**")
                    st.write(f"Top score: `{top_score:.3f}` (threshold `{retrieval_thr:.2f}`)")
                    st.write(f"Results returned: `{len(results)}`")
                    st.write(f"Duration: `{retrieval_ms:.1f} ms`")

                with d3:
                    st.markdown("**Policy**")
                    st.write(f"Domain confidence threshold = `{domain_thr}`")
                    st.write(f"Retrieval score threshold = `{retrieval_thr}`")
                    st.write(f"Number of chunks retrieved = `{retrieval.get('top_k', '—')}`")

                if results:
                    with st.expander("Raw retrieval results", expanded=False):
                        st.json(results)

        except Exception as e:
            st.error(f"API error: {e}")

st.divider()

# ---------- Eval metrics (latest run) ----------
LATEST_EVAL = {
    "total_cases": 30,
    "decision_accuracy": 1.0,
    "refusal_precision": 1.0,
    "refusal_reason_accuracy": 0.8333,
    "empty_retrieval_rate": 0.2,
    "answer_citation_compliance": 1.0,
    "recall_at_k": 1.0,
    "k": 5,
}

with st.expander("Evaluation (latest run)", expanded=False):
    st.caption(
        f"Golden set: {LATEST_EVAL['total_cases']} labeled cases. "
        f"Focus: validating guardrail + grounding behavior (not a production benchmark)."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Decision accuracy", f"{LATEST_EVAL['decision_accuracy']*100:.1f}%")
    c2.metric("Refusal precision", f"{LATEST_EVAL['refusal_precision']*100:.1f}%")
    c3.metric("Citation compliance", f"{LATEST_EVAL['answer_citation_compliance']*100:.1f}%")
    c4.metric(f"Recall@{LATEST_EVAL['k']}", f"{LATEST_EVAL['recall_at_k']*100:.1f}%")

    d1, d2 = st.columns(2)
    d1.metric("Refusal reason accuracy", f"{LATEST_EVAL['refusal_reason_accuracy']*100:.1f}%")
    d2.metric("Empty retrieval rate", f"{LATEST_EVAL['empty_retrieval_rate']*100:.1f}%")

    with st.expander("What these mean", expanded=False):
        st.markdown(
            f"""
- **Decision accuracy**: Correct answer vs refuse decisions on the labeled set.
- **Refusal precision**: When the system refuses, it was expected to refuse.
- **Refusal reason accuracy**: When refusing, the policy reason label matched the expected reason.
- **Citation compliance**: Answers always included citations; refusals never included citations.
- **Recall@{LATEST_EVAL['k']}**: At least one relevant chunk appears in the top-{LATEST_EVAL['k']} retrieval results.
- **Empty retrieval rate**: Fraction of queries where retrieval returned no chunks.
            """
        )

st.caption("Source: [GitHub](https://github.com/tanmaythaware/guideline-iq)")
