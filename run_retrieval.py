"""Step 4: run all four retrievers on the 500 questions and measure retrieval quality.

Run:  python run_retrieval.py

Creates in results/:
  retrieved_<retriever>.json  top-10 passages (+ scores) per question, used in Step 5
  retrieval_metrics.csv       the main retrieval table for the paper
"""
import json

import pandas as pd
from tqdm import tqdm

import config as C
from retrieval.retrievers import (BM25Retriever, DenseRetriever, HybridRetriever,
                                  RerankRetriever)

KS = [2, 5, 10]
SAVE_TOP = 10


def metrics_for(ranked_pids, gold):
    """Retrieval metrics for one question. gold = set of 2 gold passage ids."""
    out = {}
    for k in KS:
        found = len(gold & set(ranked_pids[:k]))
        out[f"recall@{k}"] = found / len(gold)          # share of gold passages found
        out[f"both@{k}"] = float(found == len(gold))    # found ALL gold passages
    first = next((r for r, pid in enumerate(ranked_pids, 1) if pid in gold), None)
    out["mrr"] = 1.0 / first if first else 0.0           # how high the first gold one is
    return out


def evaluate(retriever, questions):
    rows, saved = [], {}
    for q in tqdm(questions, desc=retriever.name):
        hits = retriever.retrieve(q["question"], SAVE_TOP)
        saved[q["qid"]] = [{"pid": pid, "score": round(s, 5)} for pid, s in hits]
        m = metrics_for([pid for pid, _ in hits], set(q["gold_pids"]))
        rows.append({"qid": q["qid"], "type": q["type"], **m})
    return pd.DataFrame(rows), saved


def main():
    questions = json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))
    corpus = json.loads(C.CORPUS_FILE.read_text(encoding="utf-8"))
    C.RESULTS_DIR.mkdir(exist_ok=True)
    print(f"{len(questions)} questions, {len(corpus)} passages\n")

    print("Building retrievers...")
    bm25 = BM25Retriever(corpus)
    dense = DenseRetriever(corpus)
    hybrid = HybridRetriever(bm25, dense)
    rerank = RerankRetriever(hybrid, corpus)

    summary, per_question = [], []
    for r in [bm25, dense, hybrid, rerank]:
        df, saved = evaluate(r, questions)
        out = C.RESULTS_DIR / f"retrieved_{r.name}.json"
        out.write_text(json.dumps(saved, indent=1), encoding="utf-8")
        df.insert(0, "retriever", r.name)
        per_question.append(df)
        summary.append({"retriever": r.name,
                        **df.drop(columns=["qid", "type", "retriever"]).mean().round(3)})

    table = pd.DataFrame(summary)
    table.to_csv(C.RESULTS_DIR / "retrieval_metrics.csv", index=False)
    pd.concat(per_question).to_csv(C.RESULTS_DIR / "retrieval_per_question.csv", index=False)

    print("\nRetrieval results (higher is better):")
    print(table.to_string(index=False))
    print(f"\nKey column for later steps: both@{C.TOP_K} = share of questions where the "
          f"LLM will see BOTH gold passages.")


if __name__ == "__main__":
    main()