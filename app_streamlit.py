import streamlit as st
import requests

st.set_page_config(page_title="Guideline IQ", page_icon="🤖", layout = "centered")

st.title("Guideline IQ 🤖")
st.caption("Finance-focused RAG demo (grounded answers with refusals)")

q = st.text_input(
    "Ask a finance or compliance question:",
    placeholder="E.g., 'What should a crypto risk warning include?'",
)

if st.button("Ask") and q.strip():
    try:
        response = requests.get(
            "http://localhost:8000/ask",
            params={"q": q},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json()

        st.subheader("Answer:")
        st.write(data.get("answer", ""))

        st.caption(
            f"Decision: {data.get('decision', 'n/a')} | Confidence: {data.get('confidence', 'n/a')}"
        )

        st.subheader("Sources:")
        citations = data.get("citations", [])
        if not citations:
            st.write("No sources returned.")
        else:
            for s in citations:
                st.markdown(
                    f"**{s.get('id')}** (score: {round(s.get('score', 0.0), 3)})"
                )
                st.write(s.get("text", ""))
    
    except Exception as e:
        st.error(f"API error: {e}")