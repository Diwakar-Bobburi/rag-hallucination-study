"""Step 3: build the question set and passage corpus.

Run:  python prepare_data.py

Creates in data/:
  questions.json  500 questions with gold answers and the ids of their gold passages
  corpus.json     every unique passage (the "knowledge base" the retrievers search)
  stats.json      summary numbers for the paper's data section
"""
import json
import random
from collections import Counter

from datasets import load_dataset

import config as C


def passage_text(title, sentences):
    """One passage = title + its sentences joined into a paragraph."""
    return f"{title}: {''.join(sentences).strip()}"


def build(examples, n_questions, seed):
    """examples: list of HotpotQA rows. Returns (questions, corpus, stats)."""
    rng = random.Random(seed)
    idx = list(range(len(examples)))
    rng.shuffle(idx)
    q_idx = idx[:n_questions]
    noise_idx = idx[n_questions:n_questions + C.N_NOISE_QUESTIONS]

    corpus, title_to_pid = [], {}

    def add_paragraphs(ex):
        for title, sents in zip(ex["context"]["title"], ex["context"]["sentences"]):
            if title not in title_to_pid:
                pid = f"p{len(corpus)}"
                title_to_pid[title] = pid
                corpus.append({"pid": pid, "title": title,
                               "text": passage_text(title, sents)})

    questions = []
    for i in q_idx:
        ex = examples[i]
        add_paragraphs(ex)
        gold_titles = sorted(set(ex["supporting_facts"]["title"]))
        gold_ids = [title_to_pid[t] for t in gold_titles]
        questions.append({
            "qid": ex["id"],
            "question": ex["question"],
            "answer": ex["answer"],
            "type": ex["type"],        # "bridge" or "comparison"
            "level": ex["level"],      # "easy" / "medium" / "hard"
            "gold_pids": gold_ids,
        })

    for i in noise_idx:
        add_paragraphs(examples[i])

    stats = {
        "n_questions": len(questions),
        "n_noise_questions": len(noise_idx),
        "n_passages": len(corpus),
        "avg_passage_words": round(
            sum(len(p["text"].split()) for p in corpus) / len(corpus), 1),
        "question_types": dict(Counter(q["type"] for q in questions)),
        "question_levels": dict(Counter(q["level"] for q in questions)),
        "avg_gold_passages_per_question": round(
            sum(len(q["gold_pids"]) for q in questions) / len(questions), 2),
    }
    return questions, corpus, stats


def main():
    print(f"Loading {C.DATASET} ({C.DATASET_CONFIG}, {C.SPLIT})...")
    ds = load_dataset(C.DATASET, C.DATASET_CONFIG, split=C.SPLIT)
    examples = ds.to_list()
    print(f"  {len(examples)} examples available")

    questions, corpus, stats = build(examples, C.N_QUESTIONS, C.SEED)

    C.DATA_DIR.mkdir(exist_ok=True)
    for path, obj in [(C.QUESTIONS_FILE, questions), (C.CORPUS_FILE, corpus),
                      (C.STATS_FILE, stats)]:
        path.write_text(json.dumps(obj, indent=1, ensure_ascii=False), encoding="utf-8")
        print(f"  saved {path.relative_to(C.ROOT)}")

    print("\nStats:")
    for k, v in stats.items():
        print(f"  {k}: {v}")

    q = questions[0]
    gold = [p for p in corpus if p["pid"] in q["gold_pids"]]
    print("\nExample:")
    print(f"  Q: {q['question']}")
    print(f"  A: {q['answer']}")
    for p in gold:
        print(f"  Gold passage: {p['text'][:150]}...")


if __name__ == "__main__":
    main()