"""Prompt building and model calls (Groq or Google Gemini), with retries for rate limits.

Use make_llm(model_name): names starting with "gemini" or "gemma" go to Google,
everything else to Groq."""
import os
import re
import time

from dotenv import load_dotenv

import config as C

load_dotenv()

SYSTEM_PROMPT = (
    "You answer questions using only the provided context passages. "
    "Reply with a short answer only (a few words, no explanation)."
)
CLOSED_BOOK_PROMPT = (
    "Answer the question. Reply with a short answer only (a few words, no explanation)."
)
_THINK = re.compile(r"<think>.*?</think>", re.DOTALL)


def build_messages(question, passages):
    """passages: list of passage texts, already in retrieval order. Empty = closed-book."""
    if not passages:
        return [{"role": "system", "content": CLOSED_BOOK_PROMPT},
                {"role": "user", "content": f"Question: {question}\nAnswer:"}]
    context = "\n\n".join(f"[{i}] {t}" for i, t in enumerate(passages, 1))
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}\nAnswer:"}]


def clean(text):
    """Remove any visible reasoning (<think>...</think>) and surrounding whitespace."""
    return _THINK.sub("", text or "").strip()


class LLM:
    def __init__(self, model, client=None):
        from groq import Groq
        self.model = model
        self.client = client or Groq(api_key=os.getenv("GROQ_API_KEY"))
        self.extra = dict(C.MODEL_PARAMS.get(model, {}))

    def ask(self, messages, max_retries=20):
        import groq
        for attempt in range(max_retries):
            if getattr(self, "stop", False):
                raise RuntimeError(f"{self.model}: stopped")
            try:
                t0 = time.time()
                r = self.client.chat.completions.create(
                    model=self.model, messages=messages,
                    temperature=C.TEMPERATURE, max_tokens=C.MAX_TOKENS, **self.extra)
                choice = r.choices[0]
                return {"answer": clean(choice.message.content),
                        "finish_reason": choice.finish_reason,
                        "latency_s": round(time.time() - t0, 2),
                        "tokens": getattr(r.usage, "total_tokens", None)}
            except groq.BadRequestError as e:
                if not self.extra:
                    raise
                print(f"    {self.model} rejected {self.extra} ({e}); continuing without it")
                self.extra = {}
            except groq.RateLimitError as e:
                wait = min(_retry_after(e) or 2 ** attempt * 5, 600)
                print(f"    rate limit, waiting {wait:.0f}s...")
                time.sleep(wait)
            except (groq.APIConnectionError, groq.InternalServerError) as e:
                wait = 2 ** attempt * 2
                print(f"    {type(e).__name__}, retrying in {wait}s...")
                time.sleep(wait)
        raise RuntimeError(f"{self.model}: gave up after {max_retries} retries")


class GeminiLLM:
    """Same interface as LLM, but calls Google's Gemini API (key: GOOGLE_API_KEY)."""

    def __init__(self, model, client=None):
        from google import genai
        self.model = model
        # 3-minute timeout per request: without it, a stalled connection can hang forever.
        self.client = client or genai.Client(
            api_key=os.getenv("GOOGLE_API_KEY"),
            http_options=genai.types.HttpOptions(timeout=180_000))
        # Short thinking = faster, cheaper calls. Dropped automatically if unsupported.
        # Gemma models on this API take no system instruction or thinking setting.
        self.is_gemma = model.lower().startswith("gemma")
        self.thinking = None if self.is_gemma else "LOW"

    def ask(self, messages, max_retries=20):
        import httpx
        from google.genai import errors, types
        system = "\n".join(m["content"] for m in messages if m["role"] == "system")
        user = "\n\n".join(m["content"] for m in messages if m["role"] != "system")
        if self.is_gemma and system:
            user, system = f"{system}\n\n{user}", ""
        for attempt in range(max_retries):
            if getattr(self, "stop", False):
                raise RuntimeError(f"{self.model}: stopped")
            cfg = types.GenerateContentConfig(
                system_instruction=system or None, temperature=C.TEMPERATURE,
                max_output_tokens=C.MAX_TOKENS,
                thinking_config=(types.ThinkingConfig(thinking_level=self.thinking)
                                 if self.thinking else None))
            try:
                t0 = time.time()
                r = self.client.models.generate_content(model=self.model, contents=user,
                                                        config=cfg)
                usage = getattr(r, "usage_metadata", None)
                cand = (r.candidates or [None])[0]
                return {"answer": clean(r.text),
                        "finish_reason": str(getattr(cand, "finish_reason", "")),
                        "latency_s": round(time.time() - t0, 2),
                        "tokens": getattr(usage, "total_token_count", None)}
            except errors.APIError as e:
                if e.code == 400 and self.thinking and "think" in str(e).lower():
                    print(f"    {self.model} doesn't accept thinking_level; continuing without it")
                    self.thinking = None
                elif e.code == 429 and "perday" in str(e).lower().replace("_", ""):
                    raise RuntimeError(f"{self.model}: daily quota used up") from e
                elif e.code == 429 or (e.code or 0) >= 500:
                    # Server errors (500/503) are random blips: retry often, never wait > 60s.
                    cap = 600 if e.code == 429 else 60
                    wait = min(_gemini_retry_delay(e) or 2 ** attempt * 5, cap)
                    label = "rate limit" if e.code == 429 else f"server error {e.code}"
                    print(f"    {label}, waiting {wait:.0f}s...")
                    time.sleep(wait)
                else:
                    raise
            except (httpx.HTTPError, ConnectionError, TimeoutError) as e:
                # dropped connection / timeout: retry like a server error
                wait = min(2 ** attempt * 5, 60)
                print(f"    connection error ({type(e).__name__}), waiting {wait:.0f}s...")
                time.sleep(wait)
        raise RuntimeError(f"{self.model}: gave up after {max_retries} retries")


def make_llm(model):
    if model.lower().startswith(("gemini", "gemma")):
        return GeminiLLM(model)
    return LLM(model)


def _gemini_retry_delay(err):
    """Google sends e.g. "retryDelay": "37s" inside the error details."""
    m = re.search(r"retryDelay\W+(\d+(?:\.\d+)?)s", f"{getattr(err, 'details', '')} {err}")
    return float(m.group(1)) + 1 if m else None


def _retry_after(err):
    try:
        return float(err.response.headers.get("retry-after"))
    except Exception:
        return None