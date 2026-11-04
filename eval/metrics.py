"""Answer-accuracy metrics. Exact Match and F1 follow the official HotpotQA script."""
import re
import string
from collections import Counter

_ARTICLES = re.compile(r"\b(a|an|the)\b")
_PUNCT = set(string.punctuation)

# Phrases that mean the model declined to answer.
_ABSTAIN = re.compile(
    r"\b(i don'?t know|i do not know|unknown|cannot (be )?(determine|answer|tell)|"
    r"can'?t (determine|answer|tell)|not (mention(ed)?|provide(d)?|state(d)?|specif(y|ied)|"
    r"given|enough|available|contain)|no (information|answer|mention)|insufficient|"
    r"unable to)\b",
    re.IGNORECASE)


def normalize(s):
    s = (s or "").lower()
    s = "".join(ch for ch in s if ch not in _PUNCT)
    s = _ARTICLES.sub(" ", s)
    return " ".join(s.split())


def exact_match(pred, gold):
    return float(normalize(pred) == normalize(gold))


def f1(pred, gold):
    p, g = normalize(pred), normalize(gold)
    special = {"yes", "no", "noanswer"}
    if (p in special or g in special) and p != g:
        return 0.0
    p_tok, g_tok = p.split(), g.split()
    common = sum((Counter(p_tok) & Counter(g_tok)).values())
    if common == 0:
        return 0.0
    precision, recall = common / len(p_tok), common / len(g_tok)
    return 2 * precision * recall / (precision + recall)


def is_abstain(pred):
    return not (pred or "").strip() or bool(_ABSTAIN.search(pred))
