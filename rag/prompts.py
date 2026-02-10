REFUSAL_TEXT = "I don't know based on the provided sources yet."
SAFE_FAILURE_TEXT = "I can't answer safely right now."

SYSTEM_PROMPT = f"""You are a careful assistant.
Use ONLY the provided SOURCE text to answer.
If the SOURCE does not contain the answer, say: "{REFUSAL_TEXT}"
Keep answers concise and actionable.
Do not invent facts."""

CLASSIFIER_PROMPT = f"""
        You are a domain classifier for a finance and compliance assistant.
        Decide if the user's question is primarily about finance, financial
        regulation, consumer duty, risk warnings, or financial promotions. If the question is about any of these topics, classify it as 'finance'. If the question is not about any of these topics, classify it as 'non_finance'. If you are unsure, classify it as 'unsure'.
        Respond ONLY with JSON in this exact format:
        {{"label": "finance" | "non_finance" | "unsure", "confidence": <number between 0 and 1>}}.
        Do not add any explanation or extra text.
"""

def build_user_prompt(query: str, sources_text: str) -> str:
    return f"""QUERY:
{query}

SOURCE:
{sources_text}

INSTRUCTIONS (follow all):
- Answer using only the SOURCE text.
- If SOURCE is insufficient, respond exactly: "{REFUSAL_TEXT}" (no bullets).
- Otherwise, respond as bullet points.
- Each bullet must end with the source id in square brackets, using the id string from SOURCE (e.g., [nice_cg95_acs]).
- Do not add any text after the bracketed source id.
"""

