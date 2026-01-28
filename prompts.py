REFUSAL_TEXT = "I don't know based on the provided sources yet."

SYSTEM_PROMPT = f"""You are a careful assistant.
Use ONLY the provided SOURCE text to answer.
If the SOURCE does not contain the answer, say: "{REFUSAL_TEXT}"
Keep answers concise and actionable.
Do not invent facts."""

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
