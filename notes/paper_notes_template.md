# Paper notes

## Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks (Lewis et al., 2020)
Link: https://arxiv.org/abs/2005.11401  (NeurIPS 2020, Facebook AI / UCL / NYU)

- Problem: Large language models store knowledge in their weights. They can't
  easily update it, can't show where an answer came from, and sometimes make facts up
  ("hallucinate"). Knowledge-intensive tasks (open-domain QA, fact checking) suffer most.

- Method: Combine two kinds of memory:
  - Parametric: a pretrained seq2seq generator (BART-large).
  - Non-parametric: a dense vector index of Wikipedia (~21M passages of 100 words),
    searched with a DPR retriever (BERT-based bi-encoder) and FAISS.
  The retriever fetches the top-k passages; the generator conditions on the question
  plus those passages to produce the answer. Trained end-to-end with no labels for
  which passage is "correct"; the query encoder is fine-tuned, while the document
  encoder and index stay frozen (re-encoding 21M passages during training is too costly).
  Two variants:
  - RAG-Sequence: uses the same retrieved passage for the whole answer.
  - RAG-Token: can draw on a different passage for each generated token.

- Data: Open-domain QA (Natural Questions, TriviaQA, WebQuestions, CuratedTrec),
  abstractive QA (MS MARCO), Jeopardy question generation, fact verification (FEVER).

- Key results:
  - State of the art on all four open-domain QA benchmarks at the time, beating much
    larger "closed-book" models that rely only on their weights.
  - In human evaluation, RAG's generations were judged more factual and more specific
    than BART without retrieval.
  - Index hot-swapping: replacing the 2018 Wikipedia index with a 2016 one changed the
    model's answers about world leaders accordingly, so knowledge can be updated
    without retraining.
  - Close to state of the art on FEVER without any supervision on retrieval.

- Limitation: Answer quality depends on retrieval quality, but the paper doesn't
  directly measure whether answers stay faithful to the retrieved passages. When the
  retriever returns irrelevant passages, the generator can still answer from its own
  weights, which may be wrong. The setup relies on Wikipedia and a fixed document index.

- Useful for my project: This is the baseline architecture (retriever + generator).
  My project tests the gap left by this paper: how hallucination changes when
  retrieval is weak, and how to reduce it.


  ## Lost in the Middle: How Language Models Use Long Contexts (Liu et al., 2023)
**Link:** https://arxiv.org/abs/2307.03172  (Stanford / UC Berkeley; published in TACL 2024)

- **Problem:** Models now accept very long inputs, but do they actually use
  information placed anywhere in that input equally well?

- **Method:** Controlled experiments that move the relevant information to different
  positions in the input:
  - Multi-document QA: the model gets 10, 20 or 30 retrieved passages, only one of
    which contains the answer; the position of that passage is varied.
  - Key-value retrieval: a synthetic task where the model must find one value in a
    long list of key-value pairs.
  Tested open and closed models, including ones built for extended context.

- **Data:** NaturalQuestions-Open (multi-document QA), synthetic JSON key-value pairs.

- **Key results:**
  - U-shaped curve: accuracy is highest when the relevant passage is at the start or
    end of the input, and drops clearly when it sits in the middle.
  - Models marketed for longer contexts were not better at using their context.
  - Adding more retrieved passages helps only up to a point; beyond that, accuracy
    barely improves even though the right passage is more likely to be included.

- **Limitation:** Tests mostly single-answer QA; doesn't propose a full fix.
  Models change quickly, so the exact numbers may not hold for newer models.

- **Useful for my project:**
  - Order of retrieved chunks matters, so I must keep chunk order fixed across runs
    (or test it) to avoid hidden bias.
  - Supports using a reranker that puts the most relevant chunk first.
  - Justifies keeping top-k small (5) rather than dumping 20+ chunks into the prompt.

## Reading list
- [X] Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- [X] Liu et al. 2023 — Lost in the Middle
- [ ] Es et al. 2023 — RAGAS
- [ ] Asai et al. 2023 — Self-RAG
- [ ] Yan et al. 2024 — Corrective Retrieval Augmented Generation
- [ ] Niu et al. 2024 — RAGTruth
