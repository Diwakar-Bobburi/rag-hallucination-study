"""Step 8b: checks on the gold-passages result (RQ1). No API calls; changes no existing file.

Table 3 pools four retrievers, so one question can appear up to four times. This script
checks how much that, and differences between questions, affect the result.

Output in results/analysis/:
  gold_found_cluster_ci.csv    95% CI that resamples whole questions, not single answers
  gold_found_oracle_check.csv  oracle hallucination on the questions in each gold-found group
                               (are the questions with a missed hop simply harder?)
  gold_found_paired.csv        questions where retrievers found different numbers of gold
                               passages, so each question is compared with itself
  question_type_pooled.csv     hallucination by question type, all conditions pooled
  answer_counts.csv            answers, model abstentions, unparsed verdicts, label totals
"""
import numpy as np
import pandas as pd

import config as C
from run_analysis import OUT, SEED, load_answers

RETRIEVERS = ["bm25", "dense", "hybrid", "hybrid_rerank"]
N_BOOT = 2000


def cluster_ci(group, rng):
    """Percentile bootstrap over questions, for the 1-of-2 and 2-of-2 groups."""
    qids = group["qid"].unique()
    by_q = {q: (g["gold_found"].to_numpy(), g["halluc"].to_numpy(float))
            for q, g in group.groupby("qid")}
    draws = {1: [], 2: []}
    for _ in range(N_BOOT):
        pick = rng.choice(qids, len(qids))
        found = np.concatenate([by_q[q][0] for q in pick])
        halluc = np.concatenate([by_q[q][1] for q in pick])
        for k in draws:
            draws[k].append(halluc[found == k].mean())
    return {k: np.percentile(v, [2.5, 97.5]) for k, v in draws.items()}


def main():
    df = load_answers()
    pooled = df[df["judged"] & df["retriever"].isin(RETRIEVERS) & df["gold_found"].notna()]
    oracle = df[df["judged"] & (df["retriever"] == "oracle")].set_index(["model", "qid"])["halluc"]
    rng = np.random.default_rng(SEED)
    cluster, check, paired = [], [], []

    for model in C.LLM_MODELS:
        g = pooled[pooled["model"] == model]
        ci = cluster_ci(g, rng)
        for k in (1, 2):
            part = g[g["gold_found"] == k]
            cluster.append({"model": model, "gold_found_top5": k, "n_answers": len(part),
                            "n_questions": part["qid"].nunique(), "halluc": part["halluc"].mean(),
                            "ci_low": ci[k][0], "ci_high": ci[k][1]})
            rates = [oracle[(model, q)] for q in part["qid"].unique() if (model, q) in oracle.index]
            check.append({"model": model, "gold_found_top5": k, "n_questions": len(rates),
                          "oracle_halluc": float(np.mean(rates))})

        one, two = [], []
        for _, q in g.groupby("qid"):
            if (q["gold_found"] == 1).any() and (q["gold_found"] == 2).any():
                one.append(q.loc[q["gold_found"] == 1, "halluc"].mean())
                two.append(q.loc[q["gold_found"] == 2, "halluc"].mean())
        paired.append({"model": model, "n_questions": len(one),
                       "halluc_one_gold": float(np.mean(one)), "halluc_two_gold": float(np.mean(two))})

    with_context = df[df["retriever"] != "none"]
    judged = with_context[with_context["judged"]]
    qtype = [{"model": m, "qtype": t, "n_judged": len(g), "halluc": g["halluc"].mean()}
             for (m, t), g in judged.groupby(["model", "qtype"])]
    counts = [{"answers_with_context": len(with_context),
               "model_abstained": int(with_context["abstain"].sum()),
               "unparsed_verdicts": int((~with_context["abstain"] & ~with_context["judged"]).sum()),
               "judged": len(judged), "hallucinated": int(judged["halluc"].sum()),
               "contradicted": int((judged["label"] == "contradicted").sum()),
               "unsupported": int((judged["label"] == "unsupported").sum())}]

    for name, rows in [("gold_found_cluster_ci", cluster), ("gold_found_oracle_check", check),
                       ("gold_found_paired", paired), ("question_type_pooled", qtype),
                       ("answer_counts", counts)]:
        table = pd.DataFrame(rows)
        table.to_csv(OUT / f"{name}.csv", index=False)
        print(f"\n{name}\n{table.round(3).to_string(index=False)}")


if __name__ == "__main__":
    main()
