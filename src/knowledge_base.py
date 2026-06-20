"""
Task 2 — Knowledge Repository.

Stores and indexes the clean corpus in a persistent ChromaDB collection
(Module 10 pattern). We compute embeddings with the shared SentenceTransformer
model and hand them to Chroma explicitly, so the embedding step is transparent.

Public API:
    build_index(df)          -> (re)build the Chroma collection from the corpus
    get_collection()         -> the persistent Chroma collection
    count()                  -> number of indexed documents
"""

import chromadb

from src import config
from src.utils import get_embedder


def _client():
    # Persistent on-disk store so the index survives between runs.
    return chromadb.PersistentClient(path=str(config.CHROMA_DIR))


def get_collection():
    return _client().get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )


def build_index(df) -> None:
    """Embed every document and (re)load it into the Chroma collection."""
    client = _client()
    # Start fresh each build so re-collection does not duplicate rows.
    try:
        client.delete_collection(config.COLLECTION_NAME)
    except Exception:
        pass
    collection = client.get_or_create_collection(
        name=config.COLLECTION_NAME, metadata={"hnsw:space": "cosine"}
    )

    embedder = get_embedder()
    texts = (df["title"] + ". " + df["text"]).tolist()
    embeddings = embedder.encode(
        texts, convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=True
    ).tolist()

    collection.add(
        ids=df["id"].astype(str).tolist(),
        documents=texts,
        embeddings=embeddings,
        metadatas=[
            {
                "title": r.title,
                "source": r.source,
                "source_type": r.source_type,
                "url": r.url,
                "date": str(r.date),
            }
            for r in df.itertuples()
        ],
    )
    print(f"[knowledge_base] indexed {collection.count()} documents in Chroma")


def count() -> int:
    return get_collection().count()


if __name__ == "__main__":
    from src.preprocess import load_corpus

    build_index(load_corpus())
