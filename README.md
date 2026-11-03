# Retrieval Quality and Hallucination in RAG with Small Open LLMs

**Author:** Diwakar Bobburi

## Research questions
1. How do retrieval choices (BM25, dense, hybrid, reranking) affect answer faithfulness in RAG?
2. Can abstention or forced citation reduce hallucination without large accuracy loss?

## Setup
```bash
git clone <your-repo-url>
cd rag-hallucination-study
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env    # then paste your Groq key into .env
python check_setup.py
```

## Structure
| Folder | Contents |
|---|---|
| `data/` | Questions, gold answers, corpus, FAISS index |
| `retrieval/` | BM25, dense, hybrid, reranker |
| `generation/` | LLM calls via Groq |
| `eval/` | EM/F1, RAGAS, manual labels |
| `results/` | Run outputs and tables |
| `notebooks/` | Exploration and plots |
| `notes/` | Paper notes |

All settings live in `config.py`.

## Status
- [x] Step 1: Setup
- [X] Step 2: Literature review
- [ ] Step 3: Data
- [ ] Step 4: Retrievers
- [ ] Step 5: Generation
- [ ] Step 6: Evaluation
- [ ] Step 7: Mitigation
- [ ] Step 8: Analysis
- [ ] Step 9: Paper
- [ ] Step 10: Release
