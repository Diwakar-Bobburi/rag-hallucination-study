"""The four retrievers compared in this study.

Every retriever has the same interface:
    retrieve(query: str, k: int) -> list of (pid, score), best first
"""
import re

import numpy as np

import config as C

_TOKEN = re.compile(r"\w+")


def tokenize(text):
    return _TOKEN.findall(text.lower())


class BM25Retriever:
    """Keyword search: scores passages by overlapping (weighted) words."""
    name = "bm25"

    def __init__(self, corpus):
        from rank_bm25 import BM25Okapi
        self.pids = [p["pid"] for p in corpus]
        self.bm25 = BM25Okapi([tokenize(p["text"]) for p in corpus])

    def retrieve(self, query, k):
        scores = self.bm25.get_scores(tokenize(query))
        top = np.argsort(-scores)[:k]
        return [(self.pids[i], float(scores[i])) for i in top]


class DenseRetriever:
    """Semantic search: embeds query and passages, returns nearest by cosine similarity."""
    name = "dense"
    # BGE models work best when short queries get this prefix (passages get none).
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, corpus, model_name=C.EMBED_MODEL, cache_file=None):
        import faiss
        from sentence_transformers import SentenceTransformer
        self.pids = [p["pid"] for p in corpus]
        self.model = SentenceTransformer(model_name)
        cache_file = cache_file or (C.DATA_DIR / "corpus_embeddings.npy")

        if cache_file.exists():
            emb = np.load(cache_file)
            if emb.shape[0] != len(corpus):
                raise ValueError(f"{cache_file.name} is out of date. Delete it and rerun.")
            print(f"  loaded cached embeddings from {cache_file.name}")
        else:
            print(f"  embedding {len(corpus)} passages (one-off, cached afterwards)...")
            emb = self.model.encode([p["text"] for p in corpus], batch_size=64,
                                    normalize_embeddings=True, show_progress_bar=True)
            emb = np.asarray(emb, dtype="float32")
            np.save(cache_file, emb)

        self.index = faiss.IndexFlatIP(emb.shape[1])  # inner product = cosine (normalised)
        self.index.add(emb)

    def retrieve(self, query, k):
        q = self.model.encode([self.QUERY_PREFIX + query], normalize_embeddings=True)
        scores, idx = self.index.search(np.asarray(q, dtype="float32"), k)
        return [(self.pids[i], float(s)) for i, s in zip(idx[0], scores[0])]


class HybridRetriever:
    """Combines BM25 and dense rankings with Reciprocal Rank Fusion (RRF).
    A passage's score = sum over both rankings of 1 / (RRF_K + rank)."""
    name = "hybrid"

    def __init__(self, bm25, dense, pool=100, rrf_k=C.RRF_K):
        self.bm25, self.dense, self.pool, self.rrf_k = bm25, dense, pool, rrf_k

    def retrieve(self, query, k):
        fused = {}
        for ranking in (self.bm25.retrieve(query, self.pool),
                        self.dense.retrieve(query, self.pool)):
            for rank, (pid, _) in enumerate(ranking, start=1):
                fused[pid] = fused.get(pid, 0.0) + 1.0 / (self.rrf_k + rank)
        return sorted(fused.items(), key=lambda x: -x[1])[:k]


class RerankRetriever:
    """Takes a base retriever's top candidates and re-scores each (query, passage)
    pair with a cross-encoder, which reads both together and is more accurate.
    Its scores are also the confidence signal used for abstention in Step 7."""
    name = "hybrid_rerank"

    def __init__(self, base, corpus, model_name=C.RERANK_MODEL,
                 n_candidates=C.RERANK_CANDIDATES):
        from sentence_transformers import CrossEncoder
        self.base, self.n_candidates = base, n_candidates
        self.text = {p["pid"]: p["text"] for p in corpus}
        self.model = CrossEncoder(model_name)

    def retrieve(self, query, k):
        cands = [pid for pid, _ in self.base.retrieve(query, self.n_candidates)]
        scores = self.model.predict([(query, self.text[pid]) for pid in cands])
        ranked = sorted(zip(cands, map(float, scores)), key=lambda x: -x[1])
        return ranked[:k]