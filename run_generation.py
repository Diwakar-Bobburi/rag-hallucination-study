"""Step 5: generate an answer for every (retriever, model, question).

Conditions (retriever names):
  none            closed-book: no passages (shows what the model "knows" alone)
  bm25 / dense / hybrid / hybrid_rerank   top-TOP_K retrieved passages
  oracle          only the 2 gold passages (upper bound: perfect retrieval)

Run a small pilot first, then the full run:
  python run_generation.py --limit 20
  python run_generation.py

Resumable: rerunning skips questions already answered, so you can stop and
continue later (useful when you hit Groq's daily limits).

Output: results/gen_<retriever>__<model>.jsonl  (one line per question)
"""
import argparse
import json

from tqdm import tqdm

import config as C
from generation.llm import build_messages, make_llm

RETRIEVERS = ["none", "bm25", "dense", "hybrid", "hybrid_rerank", "oracle"]


def out_path(retriever, model):
    return C.RESULTS_DIR / f"gen_{retriever}__{model.replace('/', '_')}.jsonl"


def done_qids(path):
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["qid"] for line in f if line.strip()}


def passages_for(retriever, q, retrieved, text):
    if retriever == "none":
        return []
    if retriever == "oracle":
        return list(q["gold_pids"])
    return [h["pid"] for h in retrieved[retriever][q["qid"]][:C.TOP_K]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--retrievers", nargs="+", default=RETRIEVERS, choices=RETRIEVERS)
    ap.add_argument("--models", nargs="+", default=C.LLM_MODELS)
    ap.add_argument("--limit", type=int, default=None, help="only the first N questions")
    args = ap.parse_args()

    questions = json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))[:args.limit]
    corpus = json.loads(C.CORPUS_FILE.read_text(encoding="utf-8"))
    text = {p["pid"]: p["text"] for p in corpus}

    retrieved = {}
    for r in args.retrievers:
        if r in ("none", "oracle"):
            continue
        f = C.RESULTS_DIR / f"retrieved_{r}.json"
        if not f.exists():
            raise SystemExit(f"Missing {f.name}. Run python run_retrieval.py first.")
        retrieved[r] = json.loads(f.read_text(encoding="utf-8"))

    for model in args.models:
        llm = make_llm(model)
        for r in args.retrievers:
            path = out_path(r, model)
            skip = done_qids(path)
            todo = [q for q in questions if q["qid"] not in skip]
            if not todo:
                print(f"{r} / {model}: already done")
                continue
            with path.open("a", encoding="utf-8") as f:
                for q in tqdm(todo, desc=f"{r} / {model}"):
                    pids = passages_for(r, q, retrieved, text)
                    msgs = build_messages(q["question"], [text[p] for p in pids])
                    try:
                        res = llm.ask(msgs)
                    except RuntimeError:
                        print(f"\n{model}: daily limit reached. All answers so far are saved.\n"
                              f"Rerun the same command in a few hours to continue.")
                        return
                    f.write(json.dumps({"qid": q["qid"], "retriever": r, "model": model,
                                        "pids": pids, "gold": q["answer"], **res},
                                       ensure_ascii=False) + "\n")
                    f.flush()

    print("\nDone. Files are in results/ (gen_*.jsonl).")


if __name__ == "__main__":
    main()