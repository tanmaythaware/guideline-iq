SYSTEM_PROMPT = """You are a careful assistant.
Use ONLY the provided SOURCE text to answer.
If the SOURCE does not contain the answer, say: "I don't know based on the provided sources."
Keep answers concise and actionable.
Do not invent facts."""

def build_user_prompt(query: str, sources_text: str) -> str:
    return f"""QUERY:
{query}

SOURCE:
{sources_text}

INSTRUCTIONS:
- Answer the QUERY using only SOURCE.
- If SOURCE is insufficient, say you don't know based on the sources.
"""
