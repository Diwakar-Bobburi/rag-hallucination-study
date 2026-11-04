"""Step 6: score every generated answer.

  python run_eval.py --no-judge     accuracy only (Exact Match, F1, abstain rate), no API calls
  python run_eval.py                + faithfulness judge (JUDGE_MODEL in config.py)
  python run_eval.py --judge gemini-...   use a different judge for this run

The judge is resumable: verdicts are cached per judge in results/judges/<judge>/,
so rerun the same command after a daily limit and it continues. Verdicts from
different judges are kept apart, so two judges can later be compared.

Outputs:
  results/eval_per_answer.csv   one row per answer (for analysis and manual checks)
  results/main_table.csv        the main results table for the paper

Hallucination rate = share of non-abstained answers the judge labels
contradicted or unsupported. Not computed for "none" (no context to check).
"""
import argparse
import glob
import json
from concurrent.futures import CancelledError, ThreadPoolExecutor, as_completed
from pathlib import Path

import pandas as pd
from tqdm import tqdm

import config as C
from eval.judge import build_judge_messages, parse_label
from eval.metrics import exact_match, f1, is_abstain


def load_jsonl(path):
    if not Path(path).exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def judge_dir(judge_model):
    return C.RESULTS_DIR / "judges" / judge_model.replace("/", "_")


def judge_path(gen_file, judge_model):
    return judge_dir(judge_model) / Path(gen_file).name.replace("gen_", "", 1)


def migrate_old_verdicts():
    """Earlier versions saved verdicts as results/judge_*.jsonl (all by gpt-oss-120b).
    Move them into results/judges/<judge>/ so they aren't lost or mixed up."""
    for old in C.RESULTS_DIR.glob("judge_*.jsonl"):
        rows = load_jsonl(old)
        judge = next((r.get("judged_by") for r in rows if r.get("judged_by")), "openai/gpt-oss-120b")
        new = judge_path(old.name.replace("judge_", "gen_", 1), judge)
        new.parent.mkdir(parents=True, exist_ok=True)
        if new.exists():
            with new.open("a", encoding="utf-8") as f:
                f.write(old.read_text(encoding="utf-8"))
            old.unlink()
        else:
            old.rename(new)
        print(f"  moved {old.name} -> {new.relative_to(C.ROOT)}")


def run_judge(rows, gen_file, questions, text, llm, judge_model, workers=1):
    """Adds judge labels to rows (in place). With llm=None, only reads cached verdicts."""
    jpath = judge_path(gen_file, judge_model)
    jpath.parent.mkdir(parents=True, exist_ok=True)
    cache = {r["qid"]: r for r in load_jsonl(jpath)}
    todo = [r for r in rows if r["qid"] not in cache
            and r["retriever"] != "none" and not r["abstain"]]
    stopped = False
    if todo and llm is not None:
        def judge_one(r):
            passages = [text[p] for p in r["pids"]]
            msgs = build_judge_messages(questions[r["qid"]]["question"], r["answer"], passages)
            label, reason = parse_label(llm.ask(msgs)["answer"])
            return {"qid": r["qid"], "label": label, "reason": reason, "judged_by": llm.model}

        # Several requests at once; results are written here, in the main thread only.
        with jpath.open("a", encoding="utf-8") as f, \
                ThreadPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(judge_one, r) for r in todo]
            for fut in tqdm(as_completed(futures), total=len(futures),
                            desc=f"judge {Path(gen_file).stem[4:]}"):
                try:
                    v = fut.result()
                except RuntimeError:
                    if not stopped:
                        print(f"\nJudge ({llm.model}) hit its daily limit. Verdicts so far "
                              "are saved; rerun later to continue.")
                        stopped = True
                        llm.stop = True  # tells in-flight requests to give up instead of waiting
                        for other in futures:
                            other.cancel()
                    continue
                except CancelledError:
                    continue
                cache[v["qid"]] = v
                f.write(json.dumps(v, ensure_ascii=False) + "\n")
                f.flush()
    for r in rows:
        v = cache.get(r["qid"])
        r["judge_label"] = v["label"] if v else None
        r["judge_reason"] = v["reason"] if v else None
    return stopped


def summarise(df):
    nan = float("nan")
    rows = []
    for (retriever, model), g in df.groupby(["retriever", "model"]):
        answered = g[~g["abstain"]]
        judged = answered[answered["judge_label"].notna()]
        has_judge = retriever != "none" and len(judged) > 0
        rows.append({
            "retriever": retriever, "model": model, "n": len(g),
            "EM": g["em"].mean(), "F1": g["f1"].mean(),
            "abstain_rate": g["abstain"].mean(),
            "judged": "-" if retriever == "none" else f"{len(judged)}/{len(answered)}",
            "hallucination_rate": (judged["judge_label"] != "supported").mean() if has_judge else nan,
            "contradicted": (judged["judge_label"] == "contradicted").mean() if has_judge else nan,
            "unsupported": (judged["judge_label"] == "unsupported").mean() if has_judge else nan,
        })
    table = pd.DataFrame(rows)

    ret_file = C.RESULTS_DIR / "retrieval_metrics.csv"
    if ret_file.exists():
        ret = pd.read_csv(ret_file)[["retriever", f"both@{C.TOP_K}"]]
        table = table.merge(ret, on="retriever", how="left")

    order = {r: i for i, r in enumerate(
        ["none", "bm25", "dense", "hybrid", "hybrid_rerank", "oracle"])}
    table["_o"] = table["retriever"].map(order)
    return table.sort_values(["model", "_o"]).drop(columns="_o")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-judge", action="store_true", help="skip API calls (accuracy only)")
    ap.add_argument("--judge", default=C.JUDGE_MODEL, help="judge model (default: config.py)")
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel judge requests (default 4 for Gemini, 1 for Groq)")
    args = ap.parse_args()
    migrate_old_verdicts()

    questions = {q["qid"]: q for q in json.loads(C.QUESTIONS_FILE.read_text(encoding="utf-8"))}
    text = {p["pid"]: p["text"] for p in json.loads(C.CORPUS_FILE.read_text(encoding="utf-8"))}

    llm = None
    if not args.no_judge:
        from generation.llm import make_llm
        llm = make_llm(args.judge)
        if args.workers is None:
            args.workers = 4 if args.judge.lower().startswith(("gemini", "gemma")) else 1
        print(f"Judge: {args.judge}  ({args.workers} parallel requests)")

    all_rows = []
    gen_files = sorted(glob.glob(str(C.RESULTS_DIR / "gen_*.jsonl")))
    if not gen_files:
        raise SystemExit("No gen_*.jsonl files in results/. Run run_generation.py first.")

    for gf in gen_files:
        rows = load_jsonl(gf)
        for r in rows:
            r["em"] = exact_match(r["answer"], r["gold"])
            r["f1"] = f1(r["answer"], r["gold"])
            r["abstain"] = is_abstain(r["answer"])
        if run_judge(rows, gf, questions, text, llm, args.judge, args.workers or 1):
            llm = None  # daily limit reached: stop calling the API, just read saved verdicts
        all_rows += rows

    df = pd.DataFrame(all_rows)
    df["qtype"] = df["qid"].map(lambda q: questions[q]["type"])
    keep = ["qid", "qtype", "retriever", "model", "gold", "answer", "em", "f1",
            "abstain", "judge_label", "judge_reason", "finish_reason"]
    df[keep].to_csv(C.RESULTS_DIR / "eval_per_answer.csv", index=False, encoding="utf-8")

    table = summarise(df)
    if args.no_judge:
        table = table.drop(columns=["judged", "hallucination_rate", "contradicted", "unsupported"])
    table.to_csv(C.RESULTS_DIR / "main_table.csv", index=False)
    pd.set_option("display.width", 200)
    print("\nMain results" + ("" if args.no_judge else f" (judge: {args.judge})") + ":")
    print(table.round(3).to_string(index=False))


if __name__ == "__main__":
    main()