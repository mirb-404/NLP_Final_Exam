"""
Hybrid Retriever  (BM25 sparse  +  dense embeddings).

Combines lexical BM25 with semantic similarity, exactly like Module 10 Task 3
and the MiniHackathon scaffold:

    score_hybrid = alpha * dense_score + (1 - alpha) * sparse_score

- Sparse: rank_bm25.BM25Okapi over whitespace-tokenised documents.
- Dense : SentenceTransformer cosine similarity (normalised embeddings).

Both scores are min-max normalised to [0, 1] before fusion so they are
comparable. Returns the top-k documents with their fused score + metadata,
which the agents use as cited evidence ([src-#]).
"""

import numpy as np
from rank_bm25 import BM25Okapi

from src import config
from src.preprocess import load_corpus
from src.utils import get_embedder


def _minmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.max() - x.min() < 1e-9:
        return np.zeros_like(x)
    return (x - x.min()) / (x.max() - x.min())


class HybridRetriever:
    """Builds the BM25 + dense indexes once, then answers queries."""

    def __init__(self, alpha: float = config.HYBRID_ALPHA):
        self.alpha = alpha
        self.df = load_corpus()
        self.texts = (self.df["title"] + ". " + self.df["text"]).tolist()

        # Sparse index
        self.bm25 = BM25Okapi([t.lower().split() for t in self.texts])

        # Dense index (precompute corpus embeddings once)
        self.embedder = get_embedder()
        self.doc_emb = self.embedder.encode(
            self.texts, convert_to_numpy=True, normalize_embeddings=True
        )

    def retrieve(self, query: str, k: int = config.TOP_K) -> list[dict]:
        # sparse scores
        sparse = np.array(self.bm25.get_scores(query.lower().split()))

        # dense scores (cosine == dot product of normalised vectors)
        q_emb = self.embedder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True
        )[0]
        dense = self.doc_emb @ q_emb

        # fuse
        fused = self.alpha * _minmax(dense) + (1 - self.alpha) * _minmax(sparse)
        top_idx = np.argsort(fused)[::-1][:k]

        results = []
        for rank, i in enumerate(top_idx, start=1):
            row = self.df.iloc[i]
            results.append(
                {
                    "rank": rank,
                    "id": row["id"],
                    "title": row["title"],
                    "text": row["text"],
                    "source": row["source"],
                    "source_type": row["source_type"],
                    "url": row["url"],
                    "date": row["date"],
                    "score": round(float(fused[i]), 4),
                    "dense": round(float(dense[i]), 4),
                    "sparse": round(float(sparse[i]), 4),
                }
            )
        return results


if __name__ == "__main__":
    r = HybridRetriever()
    for d in r.retrieve(config.ENGINE_QUERIES["opportunities"]):
        print(f"[{d['rank']}] {d['score']:.3f}  {d['title'][:80]}  ({d['source']})")
