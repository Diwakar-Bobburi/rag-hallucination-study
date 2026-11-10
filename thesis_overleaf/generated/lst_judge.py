JUDGE_SYSTEM = (
    "You are a strict fact-checker. You decide whether an ANSWER is supported by "
    "the CONTEXT passages. Use only the context, not your own knowledge."
)

JUDGE_USER = """CONTEXT:
{context}

QUESTION: {question}
ANSWER: {answer}

Is the answer supported by the context?
- "supported": the context states or clearly implies the answer.
- "contradicted": the context says something that conflicts with the answer.
- "unsupported": the context does not contain enough information to back the answer.

Reply with JSON only: {{"label": "supported|contradicted|unsupported", "reason": "<one short sentence>"}}"""
