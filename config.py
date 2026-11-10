"""Central settings for every experiment. Change values here, not inside scripts."""
from pathlib import Path

ROOT = Path(__file__).parent
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"

# Data
DATASET = "hotpotqa/hotpot_qa"
DATASET_CONFIG = "distractor"
SPLIT = "validation"
N_QUESTIONS = 500
# Paragraphs from this many EXTRA questions are added to the corpus as noise only
# (never asked). Makes retrieval harder and more realistic than 10 passages/question.
N_NOISE_QUESTIONS = 1500
SEED = 42

QUESTIONS_FILE = DATA_DIR / "questions.json"
CORPUS_FILE = DATA_DIR / "corpus.json"
STATS_FILE = DATA_DIR / "stats.json"

# Retrieval
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
TOP_K = 5             # chunks given to the LLM
RERANK_CANDIDATES = 20
RRF_K = 60            # Reciprocal Rank Fusion constant

# Generation (Groq model IDs; check console.groq.com/docs/models if one is retired)
# Picked from the models your key can access (check_setup.py prints the list).
# qwen is a "preview" model on Groq and may be retired; swap it if it disappears.
LLM_MODELS = [
    "openai/gpt-oss-20b",
    "qwen/qwen3.8-27b",
]
TEMPERATURE = 0.0
# Both models "think" before answering; the budget must cover thinking + answer.
MAX_TOKENS = 2048
# Extra per-model settings to keep thinking short. If Groq rejects one, the script
# drops it automatically and prints a warning.
MODEL_PARAMS = {
    "openai/gpt-oss-20b": {"reasoning_effort": "low"},
    "qwen/qwen3.8-27b": {"reasoning_effort": "none"},
    "openai/gpt-oss-120b": {"reasoning_effort": "low"},
}

# Evaluation (Step 6)
# Main judge: Gemma 4 31B (Google), a different model family from both generators.
# Second judge (agreement check only): openai/gpt-oss-120b.
JUDGE_MODEL = "gemma-4-31b-it"