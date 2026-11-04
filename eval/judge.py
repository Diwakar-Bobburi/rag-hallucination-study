"""Faithfulness judge: is the answer supported by the passages the model was given?

Adapted from RAGAS faithfulness (Es et al., 2023). Answers here are short, so
instead of splitting into statements we judge the whole answer in one call, and
we use RAGTruth-style labels (Niu et al., 2024):
  supported     every claim is backed by the context
  contradicted  the answer conflicts with the context
  unsupported   the answer is not backed by the context (baseless / from memory)

Note: an answer that merely APPEARS in the context is not necessarily supported
(e.g. the model picks the wrong name from a passage), so every answer is judged.
"""
import json
import re

from eval.metrics import normalize

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

LABELS = ("supported", "contradicted", "unsupported")
_LABEL_RE = re.compile(r"\b(supported|contradicted|unsupported)\b", re.IGNORECASE)


def string_match(answer, passages):
    a = normalize(answer)
    if not a or a in {"yes", "no"}:
        return False
    ctx = normalize(" ".join(passages))
    return f" {a} " in f" {ctx} "


def build_judge_messages(question, answer, passages):
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, 1))
    return [{"role": "system", "content": JUDGE_SYSTEM},
            {"role": "user", "content": JUDGE_USER.format(
                context=context, question=question, answer=answer)}]


def parse_label(text):
    """Returns (label, reason). label is None if the judge reply can't be read."""
    text = text or ""
    m = re.search(r"\{.*\}", text, re.DOTALL)
    if m:
        try:
            obj = json.loads(m.group(0))
            label = str(obj.get("label", "")).lower().strip()
            if label in LABELS:
                return label, str(obj.get("reason", ""))[:300]
        except json.JSONDecodeError:
            pass
    # Fallback: first label word found. Check "unsupported" before "supported".
    lower = text.lower()
    for label in ("unsupported", "contradicted", "supported"):
        if re.search(rf"\b{label}\b", lower):
            return label, text[:300]
    return None, text[:300]