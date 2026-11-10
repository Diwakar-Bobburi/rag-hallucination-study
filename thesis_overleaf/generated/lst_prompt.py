SYSTEM_PROMPT = (
    "You answer questions using only the provided context passages. "
    "Reply with a short answer only (a few words, no explanation)."
)

CLOSED_BOOK_PROMPT = (
    "Answer the question. Reply with a short answer only (a few words, no explanation)."
)

def build_messages(question, passages):
    """passages: list of passage texts, already in retrieval order. Empty = closed-book."""
    if not passages:
        return [{"role": "system", "content": CLOSED_BOOK_PROMPT},
                {"role": "user", "content": f"Question: {question}\nAnswer:"}]
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}]
