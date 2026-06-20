"""
Hybrid Retriever  (BM25 sparse  +  dense embeddings).

Combines lexical BM25 with semantic similarity, exactly like Module 10 Task 3
and the MiniHackathon scaffold:

    score_hybrid = alpha * dense_score + (1 - alpha) * sparse_score

Dense vectors are read straight from the persistent Chroma index (built once at
ingest time) — the corpus is never re-embedded here, only the query is. Both
scores are min-max normalised to [0, 1] before fusion. Returns the top-k
documents with their fused score + metadata, used as cited evidence ([src-#]).
"""

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.knowledge_base import get_collection
from src.utils import get_embedder


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.max() - x.min() < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


class HybridRetriever:
    """Loads the prebuilt Chroma index once (docs + embeddings), then answers queries."""

    def __init__(self, alpha: float = config.HYBRID_ALPHA):
        self.alpha = alpha
        data = get_collection().get(include=["embeddings", "documents", "metadatas"])
        self.ids = data["ids"]
        self.texts = data["documents"]
        self.meta = data["metadatas"]
        self.doc_emb = np.asarray(data["embeddings"], dtype=float)  # already normalised at index time

        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])
        self.embedder = get_embedder()

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[dict]:
        sparse = np.array(self.bm25.get_scores(query.lower().split()))
        q_emb = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        dense = self.doc_emb @ q_emb  # cosine (vectors are normalised)

        fused = self.alpha * _minmax(dense) + (1 - self.alpha) * _minmax(sparse)
        top_idx = np.argsort(fused)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            m = self.meta[i]
            results.append({
                "rank": rank,
                "id": self.ids[i],
                "title": m.get("title", ""),
                "text": self.texts[i],
                "source": m.get("source", ""),
                "source_type": m.get("source_type", ""),
                "url": m.get("url", ""),
                "date": m.get("date", ""),
                "score": round(float(fused[i]), 4),
                "dense": round(float(dense[i]), 4),
                "sparse": round(float(sparse[i]), 4),
            })
        return results


if __name__ == "__main__":
    r = HybridRetriever()
    for d in r.retrieve(config.ENGINE_QUERIES["opportunities"]):
        print(f"[{d['rank']}] {d['score']:.3f}  {d['title'][:80]}  ({d['source']})")
