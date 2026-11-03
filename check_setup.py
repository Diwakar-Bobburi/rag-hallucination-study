"""Run once after installing: python check_setup.py
Checks packages, your Groq key, and that the dataset downloads."""
import importlib
import os
import sys
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

PACKAGES = ["datasets", "rank_bm25", "sentence_transformers", "faiss",
            "groq", "dotenv", "pandas", "matplotlib"]

ok = True
print("1. Packages")
for p in PACKAGES:
    try:
        importlib.import_module(p)
        print(f"   OK   {p}")
    except Exception:
        print(f"   MISSING {p}  ->  pip install -r requirements.txt")
        ok = False

print("2. Groq API key")
from dotenv import load_dotenv
load_dotenv()
key = os.getenv("GROQ_API_KEY")
if not key or key.startswith("your_"):
    print("   MISSING  ->  copy .env.example to .env and add your key")
    ok = False
else:
    try:
        from groq import Groq
        client = Groq(api_key=key)
        available = sorted(m.id for m in client.models.list().data)
        print(f"   Key works. Models your key can use: {', '.join(available)}")
        from config import LLM_MODELS
        for m in LLM_MODELS:
            if m not in available:
                print(f"   MISSING model {m}  ->  change it in config.py")
                ok = False
                continue
            r = client.chat.completions.create(
                model=m,
                messages=[{"role": "user", "content": "Reply with one word: ready"}],
                max_tokens=300)
            print(f"   OK   {m} replied: {(r.choices[0].message.content or '').strip()}")
    except Exception as e:
        print(f"   FAILED  {e}")
        ok = False

print("3. Dataset")
try:
    from datasets import load_dataset
    ds = load_dataset("hotpotqa/hotpot_qa", "distractor", split="validation[:3]")
    print(f"   OK   sample question: {ds[0]['question']}")
except Exception as e:
    print(f"   FAILED  {e}")
    ok = False

print("\nAll good, ready for Step 2." if ok else "\nFix the items above, then rerun.")
sys.exit(0 if ok else 1)