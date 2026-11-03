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
SEED = 42

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
MAX_TOKENS = 512   # both models reason before answering, so leave room