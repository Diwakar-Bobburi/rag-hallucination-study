"""Compare the LLM reader's labels (Claude Opus 5.5) with the two judges. No API calls.

This is a check by a THIRD MODEL, not a human check. Don't report it as one.

Input:   results/analysis/reader_labels_claude.csv, error_types_claude.csv, manual_labels_key.csv,
         error_samples.csv, answer_counts.csv (from run_robustness.py)
Output:  results/analysis/reader_agreement.csv, reader_disagreements.csv, reader_reweighted.csv,
         error_type_counts.csv (author and model), error_type_agreement.csv
"""
import numpy as np
import pandas as pd

from run_analysis import HALLUC, MAIN_JUDGE, OUT, SECOND_JUDGE, cohen_kappa, load_answers

LABELS = ["supported", "contradicted", "unsupported"]


def main():
    reader = pd.read_csv(OUT / "reader_labels_claude.csv", dtype=str, keep_default_na=False)
    if not reader["reader_label"].isin(LABELS).all() or reader["id"].nunique() != len(reader):
        raise SystemExit("reader_labels_claude.csv has a bad label or a repeated id.")
    key = pd.read_csv(OUT / "manual_labels_key.csv", encoding="utf-8-sig", dtype=str)
    merged = reader.merge(key, on="id", validate="one_to_one")

    rows = []
    for col, name in [("label", MAIN_JUDGE), ("label2", SECOND_JUDGE)]:
        x = merged[merged[col].isin(LABELS)]
        rb = ["h" if v in HALLUC else "ok" for v in x["reader_label"]]
        jb = ["h" if v in HALLUC else "ok" for v in x[col]]
        rows.append({"judge": name, "n": len(x),
                     "agree_3label": (x["reader_label"] == x[col]).mean(),
                     "kappa_3label": cohen_kappa(x["reader_label"], x[col]),
                     "agree_binary": np.mean([a == b for a, b in zip(rb, jb)]),
                     "kappa_binary": cohen_kappa(rb, jb),
                     "reader_halluc": np.mean([v == "h" for v in rb]),
                     "judge_halluc": np.mean([v == "h" for v in jb])})
    table = pd.DataFrame(rows)
    table.to_csv(OUT / "reader_agreement.csv", index=False)
    print(table.round(3).to_string(index=False))

    main_rb = merged["reader_label"].isin(HALLUC)
    main_jb = merged["label"].isin(HALLUC)
    differ = merged[main_rb != main_jb]
    differ[["id", "reader_label", "label", "label2", "note"]].to_csv(
        OUT / "reader_disagreements.csv", index=False)
    print(f"\nBinary disagreements with {MAIN_JUDGE}: {len(differ)}")
    print(pd.crosstab(merged["reader_label"], merged["label"]).to_string())

    # The sheet is half judge-hallucinated and half judge-supported, so reweight to the real rate.
    counts = pd.read_csv(OUT / "answer_counts.csv").iloc[0]
    judge_rate = counts["hallucinated"] / counts["judged"]
    keep = main_rb[main_jb].mean()    # reader also says hallucinated, when the judge does
    extra = main_rb[~main_jb].mean()  # reader says hallucinated, when the judge says supported
    weighted = pd.DataFrame([{"judge_rate": judge_rate, "reader_agrees_on_halluc": keep,
                              "reader_flags_supported": extra,
                              "reweighted_rate": judge_rate * keep + (1 - judge_rate) * extra}])
    weighted.to_csv(OUT / "reader_reweighted.csv", index=False)
    print("\n" + weighted.round(3).to_string(index=False))

    # Error causes: the author's from the sheet; the model's followed one rule, in order (see the paper).
    model = pd.read_csv(OUT / "error_types_claude.csv", dtype=str, keep_default_na=False)
    sheet = pd.read_csv(OUT / "error_samples.csv", encoding="utf-8-sig", dtype=str)
    errors = sheet[["id", "error_type", "gold_passages_in_top5"]].rename(columns={"error_type": "author"}).merge(
        model[["id", "error_type"]].rename(columns={"error_type": "model"}), on="id", validate="one_to_one")
    table = (pd.DataFrame({"author": errors["author"].value_counts(), "model": errors["model"].value_counts()})
             .fillna(0).astype(int).rename_axis("error_type").reset_index())
    table.to_csv(OUT / "error_type_counts.csv", index=False)
    agreement = pd.DataFrame([{"n": len(errors), "same_cause": int((errors["author"] == errors["model"]).sum()),
                               "kappa": cohen_kappa(errors["author"], errors["model"])}])
    agreement.to_csv(OUT / "error_type_agreement.csv", index=False)
    print("\n" + table.to_string(index=False))
    print("\n" + agreement.round(3).to_string(index=False))
    print("\nRows: author; columns: model")
    print(pd.crosstab(errors["author"], errors["model"]).to_string())
    print("\nGold passages in top 5, by the author's cause")
    print(pd.crosstab(errors["author"], errors["gold_passages_in_top5"]).to_string())
    human_summary(key, reader, counts)


def human_summary(key, reader, counts):
    """The author's labels: overall rate and the RQ1 gap, using human labels instead of the judge's."""
    sheet = pd.read_csv(OUT / "manual_labels.csv", encoding="utf-8-sig", dtype=str).fillna("")
    sheet["human_label"] = sheet["human_label"].str.strip().str.lower()
    done = sheet.loc[sheet["human_label"].isin(LABELS), ["id", "human_label"]]
    m = done.merge(key, on="id", validate="one_to_one").merge(reader[["id", "reader_label"]], on="id")
    answers = load_answers()[["qid", "model", "retriever", "gold_found"]]
    m = m.merge(answers, on=["qid", "model", "retriever"], validate="one_to_one")
    m["human_h"] = m["human_label"].isin(HALLUC)
    m["judge_h"] = m["label"].isin(HALLUC)
    m["reader_h"] = m["reader_label"].isin(HALLUC)

    judge_rate = counts["hallucinated"] / counts["judged"]
    keep = m.loc[m["judge_h"], "human_h"].mean()
    extra = m.loc[~m["judge_h"], "human_h"].mean()
    overall = pd.DataFrame([{
        "n": len(m), "n_questions": m["qid"].nunique(),
        "judge_halluc_in_sample": int(m["judge_h"].sum()), "human_halluc_in_sample": int(m["human_h"].sum()),
        "reader_halluc_in_sample": int(m["reader_h"].sum()),
        "judge_h_human_ok": int((m["judge_h"] & ~m["human_h"]).sum()),
        "judge_ok_human_h": int((~m["judge_h"] & m["human_h"]).sum()),
        "human_ok_all_three_models_h": int((~m["human_h"] & m["judge_h"] & m["reader_h"]
                                            & m["label2"].isin(HALLUC)).sum()),
        "human_agrees_on_judge_halluc": keep, "human_flags_judge_supported": extra,
        "judge_rate": judge_rate, "reweighted_rate": judge_rate * keep + (1 - judge_rate) * extra,
        "human_reader_agree_binary": (m["human_h"] == m["reader_h"]).mean(),
        "human_reader_kappa_binary": cohen_kappa(m["human_h"], m["reader_h"])}])
    overall.to_csv(OUT / "human_summary.csv", index=False)
    print("\nHuman labels\n" + overall.round(3).T.to_string())

    # RQ1 on human labels. The sample is balanced by judge verdict, so raw rates are not population
    # rates; each group is also reweighted to its own judge rate from all judged answers.
    full = load_answers()
    full = full[full["judged"] & (full["retriever"] != "none")]
    rows = []
    for group, part, pool in [
        ("one gold", m[(m["retriever"] != "oracle") & (m["gold_found"] == 1)],
         full[(full["retriever"] != "oracle") & (full["gold_found"] == 1)]),
        ("two gold", m[(m["retriever"] != "oracle") & (m["gold_found"] == 2)],
         full[(full["retriever"] != "oracle") & (full["gold_found"] == 2)]),
        ("oracle", m[m["retriever"] == "oracle"], full[full["retriever"] == "oracle"])]:
        rate = pool["halluc"].mean()
        h, ok = part[part["judge_h"]], part[~part["judge_h"]]
        keep_g = h["human_h"].mean() if len(h) else np.nan
        extra_g = ok["human_h"].mean() if len(ok) else np.nan
        rows.append({"group": group, "n": len(part), "human_halluc": int(part["human_h"].sum()),
                     "judge_halluc": int(part["judge_h"].sum()), "n_judge_h": len(h), "n_judge_ok": len(ok),
                     "judge_rate_all": rate, "human_reweighted": rate * keep_g + (1 - rate) * extra_g})
    gold = pd.DataFrame(rows)
    gold.to_csv(OUT / "human_gold_found.csv", index=False)
    print("\nRQ1 on human labels\n" + gold.round(3).to_string(index=False))


if __name__ == "__main__":
    main()
