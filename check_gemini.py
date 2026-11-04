"""Run once after adding GOOGLE_API_KEY to .env:  python check_gemini.py
Lists the Gemini/Gemma models your key can use and tests the judge model."""
import os
import sys

from dotenv import load_dotenv

load_dotenv()
key = os.getenv("GOOGLE_API_KEY")
if not key:
    sys.exit("GOOGLE_API_KEY missing from .env")

from google import genai

client = genai.Client(api_key=key)
names = []
for m in client.models.list():
    actions = getattr(m, "supported_actions", None) or []
    if "generateContent" in actions:
        names.append(m.name.removeprefix("models/"))
print("Text models your key can use:")
for n in sorted(names):
    print("  ", n)

import config as C
from generation.llm import make_llm

model = sys.argv[1] if len(sys.argv) > 1 else C.JUDGE_MODEL
if not model.startswith(("gemini", "gemma")):
    sys.exit(f"\nJUDGE_MODEL is {model}. Pick one from the list above, e.g.\n"
             f"  python check_gemini.py <model-name>\nthen put it in config.py.")
r = make_llm(model).ask([{"role": "user", "content": "Reply with one word: ready"}])
print(f"\nOK  {model} replied: {r['answer']}")