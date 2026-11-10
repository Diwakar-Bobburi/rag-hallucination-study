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
