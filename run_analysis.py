"""Step 8: analysis of the finished results. No API calls; safe to rerun any time.

Run:  python run_analysis.py
Outputs in results/analysis/:
  main_with_ci.csv          hallucination rate per condition + 95% bootstrap CI
  judge_agreement.csv       Gemma vs gpt-oss-120b: % agreement + Cohen's kappa
  judge_confusion.csv       3x3 label confusion matrix between the two judges
  gold_found.csv            hallucination rate by how many gold passages were retrieved (0/1/2)
  correct_vs_faithful.csv   2x2 counts: correct/incorrect x supported/hallucinated
  question_type.csv         EM and hallucination for bridge vs comparison questions
  error_profile.csv         hallucinations split into "answer in context" vs "not in context"
  abstention_repeated.csv   abstention averaged over 200 random dev/test splits (mean, sd)
  manual_labels.csv         100 answers for YOU to label (blind: no judge labels shown)
  manual_labels_key.csv     hidden key for that sheet (don't open before labelling)
  error_samples.csv         30 hallucinations for YOU to categorise
  manual_agreement.csv      human vs judges (only once manual_labels.csv is filled in)
  fig_main.png, fig_gold_found.png, fig_abstention.png
"""
import glob
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd

import config as C
from eval.metrics import exact_match, f1, is_abstain
from run_abstention import answer_overlap, auroc

OUT = C.RESULTS_DIR / "analysis"
MAIN_JUDGE = C.JUDGE_MODEL
SECOND_JUDGE = "openai/gpt-oss-120b"
RETRIEVERS = ["none", "bm25", "dense", "hybrid", "hybrid_rerank", "oracle"]
NICE = {"none": "Closed-book", "bm25": "BM25", "dense": "Dense", "hybrid": "Hybrid",
        "hybrid_rerank": "Hybrid+rerank", "oracle": "Oracle"}
HALLUC = {"contradicted", "unsupported"}
SEED = 42
COLORS = ["#2a78d6", "#eb6834"]  # validated categorical slots 1-2 (light surface)
INK, MUTED, SURFACE = "#0b0b0b", "#52514e", "#fcfcfb"


# --------------------------------------------------------------------------- loading
def load_jsonl(path):
    p = Path(path)
    if not p.exists():
        return []
    with p.open(encoding="utf-8") as f:
        return [json.loads(l) for l in f if l.strip()]


def judge_labels(judge):
    """{(retriever, model, qid): label} for one judge (last verdict wins)."""
    out = {}
    d = C.RESULTS_DIR / "judges" / judge.replace("/", "_")
    for f in d.glob("*.jsonl"):
        retriever, tag = f.stem.split("__", 1)
        for v in load_jsonl(f):
            out[(retriever, tag, v["qid"])] = v.get("label")
    return out


def load_answers():
    questions = {q["qid"]: q for q in json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))}
    text = {p["pid"]: p["text"] for p in json.loads(C.CORPUS_FILE.read_text(encoding="utf-8"))}
    main, second = judge_labels(MAIN_JUDGE), judge_labels(SECOND_JUDGE)

    gold_found = {}
    rpq = C.RESULTS_DIR / "retrieval_per_question.csv"
    if rpq.exists():
        t = pd.read_csv(rpq)
        for ret, qid, rec in zip(t["retriever"], t["qid"], t["recall@5"]):
            gold_found[(ret, qid)] = int(round(rec * 2))
    rows = []
    for gf in sorted(glob.glob(str(C.RESULTS_DIR / "gen_*.jsonl"))):
        for r in load_jsonl(gf):
            tag = r["model"].replace("/", "_")
            key = (r["retriever"], tag, r["qid"])
            passages = [text[p] for p in r["pids"]]
            abst = is_abstain(r["answer"])
            q = questions[r["qid"]]
            gfound = 2 if r["retriever"] == "oracle" else gold_found.get((r["retriever"], r["qid"]))
            rows.append({
                "qid": r["qid"], "model": r["model"], "retriever": r["retriever"],
                "qtype": q["type"], "question": q["question"], "gold": r["gold"],
                "answer": r["answer"], "passages": passages,
                "em": exact_match(r["answer"], r["gold"]), "f1": f1(r["answer"], r["gold"]),
                "abstain": abst, "correct": (f1(r["answer"], r["gold"]) >= 0.5) and not abst,
                "label": main.get(key), "label2": second.get(key),
                "overlap": answer_overlap(r["answer"], passages) if passages else np.nan,
                "gold_found": gfound,
            })
    df = pd.DataFrame(rows)
    df["judged"] = df["label"].isin(["supported", *HALLUC]) & ~df["abstain"]
    df["halluc"] = df["label"].isin(HALLUC)
    return df


# ------------------------------------------------------------------------- statistics
def boot_ci(values, n=2000, seed=SEED):
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(v, size=(n, len(v)), replace=True).mean(axis=1)
    return tuple(np.percentile(means, [2.5, 97.5]))


def cohen_kappa(a, b):
    a, b = list(a), list(b)
    cats = sorted(set(a) | set(b))
    n = len(a)
    if n == 0:
        return np.nan
    po = sum(x == y for x, y in zip(a, b)) / n
    pe = sum((a.count(c) / n) * (b.count(c) / n) for c in cats)
    return (po - pe) / (1 - pe) if pe < 1 else 1.0


# --------------------------------------------------------------------------- analyses
def main_with_ci(df):
    rows = []
    for (ret, model), g in df.groupby(["retriever", "model"]):
        j = g[g["judged"]]
        lo, hi = boot_ci(j["halluc"]) if ret != "none" else (np.nan, np.nan)
        rows.append({"retriever": ret, "model": model, "n": len(g),
                     "EM": g["em"].mean(), "F1": g["f1"].mean(), "abstain": g["abstain"].mean(),
                     "judged": len(j),
                     "halluc": j["halluc"].mean() if ret != "none" and len(j) else np.nan,
                     "ci_low": lo, "ci_high": hi,
                     "contradicted": (j["label"] == "contradicted").mean() if ret != "none" and len(j) else np.nan,
                     "unsupported": (j["label"] == "unsupported").mean() if ret != "none" and len(j) else np.nan})
    t = pd.DataFrame(rows)
    t["_o"] = t["retriever"].map({r: i for i, r in enumerate(RETRIEVERS)})
    return t.sort_values(["model", "_o"]).drop(columns="_o")


def judge_agreement(df):
    both = df[df["judged"] & df["label2"].isin(["supported", *HALLUC])]
    a, b = both["label"].tolist(), both["label2"].tolist()
    bin_a = ["halluc" if x in HALLUC else "ok" for x in a]
    bin_b = ["halluc" if x in HALLUC else "ok" for x in b]
    summary = pd.DataFrame([{
        "n_overlap": len(both),
        "agree_3label": np.mean([x == y for x, y in zip(a, b)]) if a else np.nan,
        "kappa_3label": cohen_kappa(a, b),
        "agree_binary": np.mean([x == y for x, y in zip(bin_a, bin_b)]) if a else np.nan,
        "kappa_binary": cohen_kappa(bin_a, bin_b),
        "halluc_rate_main": np.mean([x == "halluc" for x in bin_a]) if a else np.nan,
        "halluc_rate_second": np.mean([x == "halluc" for x in bin_b]) if a else np.nan,
    }])
    labels = ["supported", "contradicted", "unsupported"]
    conf = pd.crosstab(pd.Categorical(a, labels), pd.Categorical(b, labels),
                       rownames=[f"main: {MAIN_JUDGE}"], colnames=[f"second: {SECOND_JUDGE}"],
                       dropna=False)
    return summary, conf


def by_gold_found(df):
    d = df[df["judged"] & df["retriever"].isin(["bm25", "dense", "hybrid", "hybrid_rerank"])
           & df["gold_found"].notna()]
    rows = []
    for (model, gfound), g in d.groupby(["model", "gold_found"]):
        lo, hi = boot_ci(g["halluc"])
        rows.append({"model": model, "gold_found_top5": int(gfound), "n": len(g),
                     "halluc": g["halluc"].mean(), "ci_low": lo, "ci_high": hi,
                     "EM": g["em"].mean()})
    o = df[df["judged"] & (df["retriever"] == "oracle")]
    for model, g in o.groupby("model"):
        lo, hi = boot_ci(g["halluc"])
        rows.append({"model": model, "gold_found_top5": "oracle", "n": len(g),
                     "halluc": g["halluc"].mean(), "ci_low": lo, "ci_high": hi, "EM": g["em"].mean()})
    return pd.DataFrame(rows)


def correct_vs_faithful(df):
    d = df[df["judged"] & (df["retriever"] != "none")]
    t = d.groupby(["model", "retriever", "correct", "halluc"]).size().unstack(["correct", "halluc"], fill_value=0)
    t.columns = [f"{'correct' if c else 'incorrect'}_{'halluc' if h else 'supported'}" for c, h in t.columns]
    return t.reset_index()


def question_type(df):
    d = df[df["retriever"] != "none"]
    rows = []
    for (model, ret, qt), g in d.groupby(["model", "retriever", "qtype"]):
        j = g[g["judged"]]
        rows.append({"model": model, "retriever": ret, "qtype": qt, "n": len(g),
                     "EM": g["em"].mean(), "halluc": j["halluc"].mean() if len(j) else np.nan})
    return pd.DataFrame(rows)


def error_profile(df):
    d = df[df["halluc"] & df["judged"] & (df["retriever"] != "none")].copy()
    yn = d["answer"].str.strip().str.lower().str.rstrip(".").isin(["yes", "no"])
    d["where"] = np.where(yn, "yes/no answer",
                          np.where(d["overlap"] >= 1.0, "answer words all in context (wrong span?)",
                                   "answer words not all in context (from memory?)"))
    return d.groupby(["model", "where"]).size().unstack(fill_value=0).reset_index()


# ------------------------------------------------------------------ repeated abstention
def _metrics(sig, correct, halluc, abst, tau):
    keep = (sig >= tau) & ~abst
    n, answered = len(sig), keep.sum()
    c, h = (correct & keep).sum(), (halluc & keep).sum()
    return {"coverage": answered / n, "accuracy": c / n,
            "halluc_rate": h / answered if answered else 0.0,
            "halluc": h, "correct": c, "score": (c - h) / n}


def _choose(sig, correct, halluc, abst):
    cands = np.concatenate([[sig.min() - 1.0], np.unique(sig)])
    best = max(cands, key=lambda t: (_metrics(sig, correct, halluc, abst, t)["score"],
                                     _metrics(sig, correct, halluc, abst, t)["coverage"]))
    return best


def repeated_abstention(df, retrievers=("hybrid_rerank", "dense"), n_splits=200):
    import run_abstention as A
    rows = []
    for ret in retrievers:
        A.RETRIEVER = ret
        rfile = C.RESULTS_DIR / f"retrieved_{ret}.json"
        if not rfile.exists():
            continue
        retrieved = json.loads(rfile.read_text(encoding="utf-8"))
        text = {p["pid"]: p["text"] for p in json.loads(C.CORPUS_FILE.read_text(encoding="utf-8"))}
        for model in C.LLM_MODELS:
            fr = A.build_frame(model, MAIN_JUDGE, retrieved, text)
            if fr.empty:
                continue
            fr = fr[~(fr["judge"].isna() & ~fr["model_abstained"])].reset_index(drop=True)
            correct = fr["correct"].to_numpy(bool)
            halluc = fr["hallucinated"].to_numpy(bool)
            abst = fr["model_abstained"].to_numpy(bool)
            idx = np.arange(len(fr))
            for signal in A.SIGNALS:
                sig = fr[signal].to_numpy(float)
                full_auc = auroc(sig[~abst & ~halluc], sig[~abst & halluc])
                res = []
                for s in range(n_splits):
                    rng = np.random.default_rng(s)
                    perm = rng.permutation(idx)
                    dev, test = perm[: len(idx) // 2], perm[len(idx) // 2:]
                    tau = _choose(sig[dev], correct[dev], halluc[dev], abst[dev])
                    b = _metrics(sig[test], correct[test], halluc[test], abst[test], -np.inf)
                    a = _metrics(sig[test], correct[test], halluc[test], abst[test], tau)
                    res.append({"coverage": a["coverage"], "halluc_before": b["halluc_rate"],
                                "halluc_after": a["halluc_rate"],
                                "removed": b["halluc"] - a["halluc"], "lost": b["correct"] - a["correct"],
                                "acc_before": b["accuracy"], "acc_after": a["accuracy"]})
                r = pd.DataFrame(res)
                row = {"retriever": ret, "model": model, "signal": signal, "n": len(fr),
                       "auroc_all": full_auc}
                for col in r.columns:
                    row[f"{col}_mean"] = r[col].mean()
                    row[f"{col}_sd"] = r[col].std()
                rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------- manual sheets
def manual_sheet(df):
    sheet, key = OUT / "manual_labels.csv", OUT / "manual_labels_key.csv"
    if sheet.exists():
        return False
    pool = df[df["judged"] & df["retriever"].isin(["bm25", "dense", "hybrid", "hybrid_rerank", "oracle"])]
    h, s = pool[pool["halluc"]], pool[~pool["halluc"]]
    pick = pd.concat([h.sample(min(50, len(h)), random_state=SEED),
                      s.sample(min(50, len(s)), random_state=SEED)]).sample(frac=1, random_state=SEED)
    pick = pick.reset_index(drop=True)
    pick["id"] = [f"m{i+1:03d}" for i in range(len(pick))]
    out = pd.DataFrame({
        "id": pick["id"], "question": pick["question"], "answer": pick["answer"],
        "passages": ["\n\n".join(f"[{i}] {t}" for i, t in enumerate(p, 1)) for p in pick["passages"]],
        "human_label": "",
    })
    out.to_csv(sheet, index=False, encoding="utf-8-sig")
    pick[["id", "qid", "model", "retriever", "label", "label2"]].to_csv(key, index=False, encoding="utf-8-sig")
    return True


def manual_agreement():
    sheet, key = OUT / "manual_labels.csv", OUT / "manual_labels_key.csv"
    if not (sheet.exists() and key.exists()):
        return None
    s = pd.read_csv(sheet, encoding="utf-8-sig", dtype=str).fillna("")
    s["human_label"] = s["human_label"].str.strip().str.lower()
    done = s[s["human_label"].isin(["supported", "contradicted", "unsupported"])]
    if done.empty:
        return None
    m = done.merge(pd.read_csv(key, encoding="utf-8-sig", dtype=str), on="id")
    rows = []
    for col, name in [("label", MAIN_JUDGE), ("label2", SECOND_JUDGE)]:
        x = m[m[col].isin(["supported", "contradicted", "unsupported"])]
        if x.empty:
            continue
        hb = ["h" if v in HALLUC else "ok" for v in x["human_label"]]
        jb = ["h" if v in HALLUC else "ok" for v in x[col]]
        rows.append({"judge": name, "n": len(x),
                     "agree_3label": (x["human_label"] == x[col]).mean(),
                     "kappa_3label": cohen_kappa(x["human_label"], x[col]),
                     "agree_binary": np.mean([a == b for a, b in zip(hb, jb)]),
                     "kappa_binary": cohen_kappa(hb, jb)})
    return pd.DataFrame(rows)


def error_sheet(df):
    path = OUT / "error_samples.csv"
    if path.exists():
        return False
    d = df[df["halluc"] & df["judged"] & df["retriever"].isin(["bm25", "dense"])]
    pick = d.sample(min(30, len(d)), random_state=SEED).reset_index(drop=True)
    main_reasons = {}
    for f in (C.RESULTS_DIR / "judges" / MAIN_JUDGE.replace("/", "_")).glob("*.jsonl"):
        ret, tag = f.stem.split("__", 1)
        for v in load_jsonl(f):
            main_reasons[(ret, tag, v["qid"])] = v.get("reason", "")
    pd.DataFrame({
        "id": [f"e{i+1:02d}" for i in range(len(pick))],
        "model": pick["model"], "retriever": pick["retriever"],
        "question": pick["question"], "gold": pick["gold"], "answer": pick["answer"],
        "judge_label": pick["label"],
        "judge_reason": [main_reasons.get((r, m.replace("/", "_"), q), "")
                         for r, m, q in zip(pick["retriever"], pick["model"], pick["qid"])],
        "gold_passages_in_top5": pick["gold_found"],
        "answer_in_context": pick["overlap"] >= 1.0,
        "passages": ["\n\n".join(f"[{i}] {t}" for i, t in enumerate(p, 1)) for p in pick["passages"]],
        "error_type": "",
    }).to_csv(path, index=False, encoding="utf-8-sig")
    return True


# --------------------------------------------------------------------------- figures
def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.yaxis.grid(True, color="#e6e5e1", linewidth=0.8)
    ax.set_axisbelow(True)


def figures(main, gold, rep):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "text.color": INK, "axes.labelcolor": INK,
                         "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    models = list(C.LLM_MODELS)
    hatches = ["", "////"]

    # 1. hallucination rate by retriever
    rets = ["bm25", "dense", "hybrid", "hybrid_rerank", "oracle"]
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    w = 0.38
    x = np.arange(len(rets))
    for i, m in enumerate(models):
        t = main[main["model"] == m].set_index("retriever").reindex(rets)
        y = t["halluc"].to_numpy() * 100
        err = np.vstack([y - t["ci_low"].to_numpy() * 100, t["ci_high"].to_numpy() * 100 - y])
        ax.bar(x + (i - 0.5) * w, y, w - 0.02, color=COLORS[i], hatch=hatches[i],
               edgecolor=SURFACE, linewidth=0.6, label=m.split("/")[-1])
        ax.errorbar(x + (i - 0.5) * w, y, yerr=err, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
    both = main.drop_duplicates("retriever").set_index("retriever")
    labels = [f"{NICE[r]}\nBoth@5={1.0 if r == 'oracle' else both.loc[r, 'both5']:.2f}"
              if "both5" in both.columns else NICE[r] for r in rets]
    ax.set_xticks(x, labels)
    ax.set_ylabel("Hallucination rate (%)")
    _style(ax)
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_main.png", dpi=200)
    plt.close(fig)

    # 2. hallucination by gold passages found
    if not gold.empty:
        cats = [0, 1, 2, "oracle"]
        fig, ax = plt.subplots(figsize=(5.4, 3.4))
        x = np.arange(len(cats))
        for i, m in enumerate(models):
            t = gold[gold["model"] == m].set_index("gold_found_top5")
            t.index = [c if c == "oracle" else int(c) for c in t.index]
            t = t.reindex(cats)
            y = t["halluc"].to_numpy(float) * 100
            err = np.vstack([y - t["ci_low"].to_numpy(float) * 100, t["ci_high"].to_numpy(float) * 100 - y])
            ax.bar(x + (i - 0.5) * w, y, w - 0.02, color=COLORS[i], hatch=hatches[i],
                   edgecolor=SURFACE, linewidth=0.6, label=m.split("/")[-1])
            ax.errorbar(x + (i - 0.5) * w, y, yerr=err, fmt="none", ecolor=MUTED, elinewidth=1, capsize=2)
        ax.set_xticks(x, ["0 of 2", "1 of 2", "2 of 2", "Oracle"])
        ax.set_xlabel("Gold passages in the top 5")
        ax.set_ylabel("Hallucination rate (%)")
        _style(ax)
        ax.legend(frameon=False, fontsize=9)
        fig.tight_layout()
        fig.savefig(OUT / "fig_gold_found.png", dpi=200)
        plt.close(fig)

    # 3. AUROC by signal
    if not rep.empty:
        sigs = ["top1", "top2", "answer_overlap"]
        names = {"top1": "Top-1\nscore", "top2": "Top-2\nscore", "answer_overlap": "Answer\ngrounding"}
        fig, axes = plt.subplots(1, len(rep["retriever"].unique()), figsize=(7.2, 3.4), sharey=True)
        axes = np.atleast_1d(axes)
        for ax, ret in zip(axes, rep["retriever"].unique()):
            x = np.arange(len(sigs))
            for i, m in enumerate(models):
                t = rep[(rep["retriever"] == ret) & (rep["model"] == m)].set_index("signal").reindex(sigs)
                ax.bar(x + (i - 0.5) * w, t["auroc_all"], w - 0.02, color=COLORS[i], hatch=hatches[i],
                       edgecolor=SURFACE, linewidth=0.6, label=m.split("/")[-1])
            ax.axhline(0.5, color=MUTED, linestyle="--", linewidth=1)
            ax.set_xticks(x, [names[s] for s in sigs], fontsize=9)
            ax.set_title(NICE[ret], fontsize=10)
            ax.set_ylim(0.3, 1.0)
            _style(ax)
        axes[0].set_ylabel("AUROC (hallucination detection)")
        axes[0].legend(frameon=False, fontsize=8, loc="upper left")
        fig.tight_layout()
        fig.savefig(OUT / "fig_abstention.png", dpi=200)
        plt.close(fig)


# ------------------------------------------------------------------------------ main
def main():
    OUT.mkdir(parents=True, exist_ok=True)
    pd.set_option("display.width", 220)
    df = load_answers()

    main_t = main_with_ci(df)
    rpq = C.RESULTS_DIR / "retrieval_per_question.csv"
    if rpq.exists():  # Both@5 on the questions these answers are about, not on all questions
        t = pd.read_csv(rpq)
        both5 = t[t["qid"].isin(df["qid"])].groupby("retriever", as_index=False)["both@5"].mean()
        main_t = main_t.merge(both5.rename(columns={"both@5": "both5"}), on="retriever", how="left")
    main_t.to_csv(OUT / "main_with_ci.csv", index=False)
    print("\n1. Hallucination with 95% bootstrap CI")
    print(main_t[["retriever", "model", "EM", "abstain", "judged", "halluc", "ci_low", "ci_high"]]
          .round(3).to_string(index=False))

    agree, conf = judge_agreement(df)
    agree.to_csv(OUT / "judge_agreement.csv", index=False)
    conf.to_csv(OUT / "judge_confusion.csv")
    print("\n2. Judge agreement (main vs second)")
    print(agree.round(3).to_string(index=False))
    print(conf)

    gold = by_gold_found(df)
    gold.to_csv(OUT / "gold_found.csv", index=False)
    print("\n3. Hallucination by gold passages found in top 5")
    print(gold.round(3).to_string(index=False))

    cvf = correct_vs_faithful(df)
    cvf.to_csv(OUT / "correct_vs_faithful.csv", index=False)
    qt = question_type(df)
    qt.to_csv(OUT / "question_type.csv", index=False)
    print("\n4. Bridge vs comparison (averaged over retrieval conditions)")
    print(qt.groupby(["model", "qtype"])[["EM", "halluc"]].mean().round(3))

    ep = error_profile(df)
    ep.to_csv(OUT / "error_profile.csv", index=False)
    print("\n5. Where hallucinated answers come from (counts)")
    print(ep.to_string(index=False))

    print("\n6. Abstention over 200 random dev/test splits (this takes ~1 minute)...")
    rep = repeated_abstention(df)
    rep.to_csv(OUT / "abstention_repeated.csv", index=False)
    cols = ["retriever", "model", "signal", "auroc_all", "coverage_mean", "halluc_before_mean",
            "halluc_after_mean", "removed_mean", "lost_mean"]
    print(rep[cols].round(3).to_string(index=False))

    figures(main_t, gold, rep)
    print(f"\nFigures saved in {OUT.relative_to(C.ROOT)}")

    if manual_sheet(df):
        print("\nCreated manual_labels.csv: label all 100 rows (supported / contradicted / unsupported).")
    if error_sheet(df):
        print("Created error_samples.csv: fill error_type for all 30 rows.")
    ma = manual_agreement()
    if ma is not None:
        ma.to_csv(OUT / "manual_agreement.csv", index=False)
        print("\n7. Human vs judges")
        print(ma.round(3).to_string(index=False))
    else:
        print("\n7. Human vs judges: waiting for you to fill human_label in manual_labels.csv")


if __name__ == "__main__":
    main()