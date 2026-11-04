"""Step 7: training-free abstention based on retrieval confidence.

Idea: the cross-encoder reranker already scores how well each retrieved passage
matches the question. If even the best passages score low, retrieval probably
failed, so the system should say "I don't know" instead of letting the model guess.

This is applied AFTER generation, to the hybrid_rerank answers, so it needs no
new API calls. For every confidence threshold we measure what is gained (fewer
hallucinations) and what is lost (correct answers withheld).

Two confidence signals are compared:
  top1   score of the best passage
  top2   score of the 2nd-best passage. HotpotQA needs TWO passages, so a weak
         2nd passage suggests the second "hop" is missing.

To avoid tuning the threshold on the same questions we report on, questions are
split 50/50 into dev and test: the threshold is chosen on dev, reported on test.

Definitions (per answer):
  correct       F1 >= 0.5 against the gold answer
  hallucinated  judge label is "contradicted" or "unsupported"
  abstained     the model said "I don't know" OR the threshold withheld the answer
Score used to pick the threshold: (#correct - #hallucinated) / n  (abstaining = 0)

Run:  python run_abstention.py            (uses JUDGE_MODEL from config.py)
      python run_abstention.py --judge gemma-4-31b-it
Outputs in results/: abstention_curve.csv, abstention_summary.csv, abstention_curve.png
"""
import argparse
import json
import random

import numpy as np
import pandas as pd

import config as C
from eval.metrics import exact_match, f1, is_abstain

RETRIEVER = "hybrid_rerank"
SIGNALS = ["top1", "top2"]


def load_jsonl(path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def auroc(pos_scores, neg_scores):
    """Probability a random non-hallucinated answer has HIGHER confidence than a
    random hallucinated one (0.5 = signal is useless, 1.0 = perfect)."""
    pos, neg = np.asarray(pos_scores), np.asarray(neg_scores)
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    greater = (pos[:, None] > neg[None, :]).sum()
    ties = (pos[:, None] == neg[None, :]).sum()
    return float((greater + 0.5 * ties) / (len(pos) * len(neg)))


def build_frame(model, judge, retrieved):
    tag = model.replace("/", "_")
    gen = load_jsonl(C.RESULTS_DIR / f"gen_{RETRIEVER}__{tag}.jsonl")
    verdicts = {v["qid"]: v["label"] for v in load_jsonl(
        C.RESULTS_DIR / "judges" / judge.replace("/", "_") / f"{RETRIEVER}__{tag}.jsonl")}
    rows = []
    for r in gen:
        scores = sorted((h["score"] for h in retrieved[r["qid"]]), reverse=True)
        abst = is_abstain(r["answer"])
        rows.append({
            "qid": r["qid"], "model": model,
            "top1": scores[0], "top2": scores[1] if len(scores) > 1 else scores[0],
            "model_abstained": abst,
            "em": exact_match(r["answer"], r["gold"]),
            "correct": f1(r["answer"], r["gold"]) >= 0.5 and not abst,
            "judge": verdicts.get(r["qid"]),
        })
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["hallucinated"] = df["judge"].isin(["contradicted", "unsupported"]) & ~df["model_abstained"]
    return df


def at_threshold(df, signal, tau):
    """Metrics if we withhold every answer whose confidence is below tau."""
    keep = (df[signal] >= tau) & ~df["model_abstained"]
    n = len(df)
    correct = int((df["correct"] & keep).sum())
    halluc = int((df["hallucinated"] & keep).sum())
    answered = int(keep.sum())
    return {
        "threshold": tau, "coverage": answered / n,
        "accuracy": correct / n,                          # abstaining counts as wrong
        "selective_accuracy": correct / answered if answered else float("nan"),
        "hallucination_rate": halluc / answered if answered else 0.0,
        "hallucinations": halluc, "correct": correct,
        "score": (correct - halluc) / n,
    }


def thresholds_for(values):
    """Candidate thresholds: below every score (= never abstain) plus each unique score."""
    vals = np.unique(values)
    return np.concatenate([[vals.min() - 1.0], vals])


def choose_threshold(df, signal):
    best = max((at_threshold(df, signal, t) for t in thresholds_for(df[signal])),
               key=lambda m: (m["score"], m["coverage"]))
    return best["threshold"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judge", default=C.JUDGE_MODEL)
    args = ap.parse_args()

    retrieved = json.loads((C.RESULTS_DIR / f"retrieved_{RETRIEVER}.json").read_text(encoding="utf-8"))
    curves, summary = [], []

    for model in C.LLM_MODELS:
        df = build_frame(model, args.judge, retrieved)
        if df.empty:
            print(f"{model}: no {RETRIEVER} answers yet, skipping")
            continue
        missing = df["judge"].isna() & ~df["model_abstained"]
        if missing.any():
            print(f"{model}: {missing.sum()} answers not judged yet by {args.judge}; "
                  f"they are left out. Run run_eval.py --judge {args.judge} first.")
            df = df[~missing]
        if len(df) < 20:
            print(f"{model}: only {len(df)} usable answers, too few, skipping")
            continue

        qids = sorted(df["qid"])
        random.Random(C.SEED).shuffle(qids)
        dev_q = set(qids[: len(qids) // 2])
        dev, test = df[df["qid"].isin(dev_q)], df[~df["qid"].isin(dev_q)]

        answered = df[~df["model_abstained"]]
        for signal in SIGNALS:
            for t in thresholds_for(df[signal]):
                curves.append({"model": model, "signal": signal, **at_threshold(df, signal, t)})

            tau = choose_threshold(dev, signal)
            base = at_threshold(test, signal, thresholds_for(test[signal])[0])
            fixed = at_threshold(test, signal, tau)
            summary.append({
                "model": model, "signal": signal, "n_test": len(test),
                "auroc": auroc(answered.loc[~answered["hallucinated"], signal],
                               answered.loc[answered["hallucinated"], signal]),
                "threshold": round(tau, 3),
                "coverage": fixed["coverage"],
                "halluc_rate_before": base["hallucination_rate"],
                "halluc_rate_after": fixed["hallucination_rate"],
                "hallucinations_removed": base["hallucinations"] - fixed["hallucinations"],
                "correct_lost": base["correct"] - fixed["correct"],
                "accuracy_before": base["accuracy"],
                "accuracy_after": fixed["accuracy"],
            })

    if not summary:
        raise SystemExit("Nothing to report yet: finish hybrid_rerank generation + judging first.")

    pd.DataFrame(curves).to_csv(C.RESULTS_DIR / "abstention_curve.csv", index=False)
    table = pd.DataFrame(summary)
    table.to_csv(C.RESULTS_DIR / "abstention_summary.csv", index=False)
    pd.set_option("display.width", 220)
    print(f"\nAbstention results on the TEST half (judge: {args.judge}):")
    print(table.round(3).to_string(index=False))
    print("\nauroc: how well the signal separates good from hallucinated answers (0.5 = no better than chance)")
    plot(pd.DataFrame(curves))


def plot(curves):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    styles = {"top1": "-", "top2": "--"}
    for (model, signal), g in curves.groupby(["model", "signal"]):
        g = g.sort_values("coverage")
        ax.plot(g["coverage"], g["hallucination_rate"], styles[signal],
                label=f"{model.split('/')[-1]} · {signal}")
    ax.set_xlabel("Coverage (share of questions answered)")
    ax.set_ylabel("Hallucination rate (among answered)")
    ax.set_title("Abstaining on low retrieval confidence")
    ax.invert_xaxis()
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = C.RESULTS_DIR / "abstention_curve.png"
    fig.savefig(out, dpi=200)
    print(f"Saved plot: {out.relative_to(C.ROOT)}")


if __name__ == "__main__":
    main()