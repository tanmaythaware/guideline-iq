import streamlit as st
import requests
from string import capwords
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

        # add badge for decision and confidence
        if data.get('decision') == 'answer':
            st.markdown(
                f"**Decision: :green-badge[Answer] | Confidence: :blue-badge[{capwords(data.get('confidence'))}]**"
            )
        else:
            st.markdown(
                f"**Decision: :red-badge[Refusal] | Confidence: :blue-badge[{capwords(data.get('confidence'))}]**"
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