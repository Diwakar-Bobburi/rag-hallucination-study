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


  ## RAGAS: Automated Evaluation of Retrieval Augmented Generation (Es et al., 2023)
**Link:** https://arxiv.org/abs/2309.15217  (Cardiff University / Exploding Gradients; EACL 2024 demo track)

- **Problem:** Evaluating RAG systems usually needs human-written reference answers,
  which are slow and expensive to create. We need automatic, reference-free metrics
  that check both the retrieval and the generation steps.

- **Method:** Three metrics, each computed by prompting an LLM:
  - Faithfulness: does the answer stick to the retrieved context?
    1. LLM splits the answer into short, standalone statements.
    2. LLM checks each statement: can it be inferred from the context? (yes/no)
    3. Score = supported statements / total statements.
  - Answer relevance: does the answer address the question?
    LLM generates possible questions from the answer; score = average embedding
    similarity between those and the original question.
  - Context relevance: is the retrieved context focused?
    LLM extracts only the sentences needed to answer; score = needed / total sentences.

- **Data:** WikiEval, a small dataset the authors built from 50 Wikipedia pages
  (created after the LLM's training cutoff), with human judgements of answer and
  context quality.

- **Key results:** RAGAS scores agreed with human judgements better than simple
  baselines (e.g. asking an LLM for a direct 1–10 score). Agreement was strongest for
  faithfulness and weakest for context relevance.

- **Limitation:** The judge is itself an LLM, so it can make mistakes or be biased.
  WikiEval is small. Results depend on the judge model and the exact prompts used.

- **Useful for my project:**
  - My faithfulness metric = this 3-step method, run with my own judge model and
    prompts, which I will document fully in the paper.
  - Must validate the judge against my manual labels on ~100 answers, since the
    paper itself shows LLM judges are imperfect.
  - "Hallucination rate" = share of answers with at least one unsupported statement.



## Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection (Asai et al., 2023)
**Link:** https://arxiv.org/abs/2310.11511  (University of Washington / AI2 / IBM; ICLR 2024)

- **Problem:** Standard RAG always retrieves a fixed number of passages, even when
  retrieval isn't needed or the passages are irrelevant, and never checks whether its
  answer is actually supported by them. This can lower quality and cause hallucination.

- **Method:** Train the model to "reflect" on its own process using special
  reflection tokens it generates alongside the text:
  - Retrieve: should I retrieve for this part? (yes / no / continue)
  - IsRel: is this retrieved passage relevant?
  - IsSup: is my output fully / partially / not supported by the passage?
  - IsUse: how useful is my output overall? (1–5)
  Training: a critic model (trained on GPT-4 labels) inserts these tokens into training
  data; then the generator (Llama 2, 7B and 13B) is fine-tuned on that data.
  Inference: the model retrieves only when needed, generates candidate segments for
  several passages, and picks the best one using its own reflection scores.

- **Data:** Short-form QA (PopQA, TriviaQA), fact checking (PubHealth), reasoning
  (ARC-Challenge), long-form generation with citations (biographies, ASQA).

- **Key results:** Outperformed ChatGPT and retrieval-augmented Llama 2-chat on
  most tasks. Clear gains in factuality and in citation accuracy for long-form
  outputs. Reflection weights let users trade off fluency against strict support.

- **Limitation:** Requires fine-tuning a model, so it can't be applied to API-only
  models. Relies on GPT-4-generated labels for training. Outputs can still contain
  unsupported claims.

- **Useful for my project:**
  - Strong evidence that "decide when to trust retrieval" and "check support" reduce
    hallucination.
  - IsSup is essentially the faithfulness check I am measuring.
  - My Step 7 fix is a training-free, prompt-only version of this idea (abstain when
    retrieval confidence is low + force citations). This is a clear positioning
    statement for my paper: cheaper, works with any API model.


  ## Corrective Retrieval Augmented Generation (Yan et al., 2024)
**Link:** https://arxiv.org/abs/2401.15884

- **Problem:** RAG trusts whatever the retriever returns. When retrieval is wrong or
  irrelevant, the generator builds its answer on bad evidence and hallucinates.
  Nothing in standard RAG checks retrieval quality before generating.

- **Method:** Add a lightweight "retrieval evaluator" (a small fine-tuned T5 model)
  that scores how relevant the retrieved documents are to the question, then acts
  on one of three confidence levels:
  - Correct: keep the documents, but refine them: split into small strips and drop
    irrelevant strips ("decompose-then-recompose").
  - Incorrect: throw the documents away and use web search instead.
  - Ambiguous: combine refined documents and web search results.
  Plug-and-play: works on top of standard RAG and Self-RAG without retraining them.

- **Data:** PopQA (short-form QA), Biography (long-form generation), PubHealth
  (fact checking), ARC-Challenge (reasoning).

- **Key results:** Improved both standard RAG and Self-RAG across all four tasks,
  showing that checking retrieval quality before generation makes RAG more robust,
  especially when retrieval is poor.

- **Limitation:** The evaluator must be fine-tuned. The web search fallback adds
  latency and cost, and web results can themselves be unreliable. Extra components
  make the pipeline more complex.

- **Useful for my project:**
  - Direct support for my core idea: retrieval quality should be checked before
    trusting the answer.
  - My fix is a simpler, training-free variant: use the cross-encoder reranker's
    score as the retrieval confidence signal (no fine-tuned evaluator), and abstain
    ("I don't know") instead of falling back to web search.
  - Clear comparison point for the paper: CRAG corrects retrieval; I measure how
    much simply refusing on low confidence reduces hallucination, and at what cost
    to accuracy.

## RAGTruth: A Hallucination Corpus for Developing Trustworthy Retrieval-Augmented Language Models (Niu et al., 2024)
**Link:** https://arxiv.org/abs/2401.00396  (ACL 2024)

- **Problem:** RAG still hallucinates, but there was no large, human-labelled dataset
  of hallucinations in real RAG outputs to measure them or train detectors on.

- **Method:** Built a corpus of ~18,000 responses generated by six LLMs (GPT-3.5,
  GPT-4, Llama 2 7B/13B/70B, Mistral 7B) for RAG-style tasks. Human annotators marked
  hallucinations at the word/span level and assigned a type:
  - Evident conflict: clearly contradicts the context.
  - Subtle conflict: changes the meaning slightly (e.g. a wrong detail).
  - Evident baseless info: adds facts not found in the context.
  - Subtle baseless info: adds unsupported inferences or opinions.
  Then tested hallucination detection methods on it.

- **Data:** Three tasks: question answering (MS MARCO), data-to-text writing
  (business data from Yelp), and news summarisation.

- **Key results:**
  - Every model hallucinated; rates differed a lot between models (GPT-4 lowest)
    and between tasks (data-to-text highest).
  - Prompting a strong LLM to detect hallucinations worked only moderately.
  - A smaller model (Llama 2 13B) fine-tuned on RAGTruth detected hallucinations
    competitively with prompt-based GPT-4.

- **Limitation:** English only, three task types, and the models tested are now
  older. Human labelling is expensive, so the corpus can't easily be extended.

- **Useful for my project:**
  - Use their four hallucination types as the categories in my manual labelling and
    error analysis (Step 8), so my results are comparable to published work.
  - Their finding that LLM-based detection is imperfect supports validating my judge
    against manual labels.
  - Optional: test my faithfulness judge on a RAGTruth QA subset to report its
    accuracy against human labels.


    
## Reading list
- [X] Lewis et al. 2020 — Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
- [X] Liu et al. 2023 — Lost in the Middle
- [X] Es et al. 2023 — RAGAS
- [X] Asai et al. 2023 — Self-RAG
- [X] Niu et al. 2024 — RAGTruth
