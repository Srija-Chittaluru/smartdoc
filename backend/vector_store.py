"""Step 2 of the pipeline: embed the chunks and store them in ChromaDB.

Why a vector database instead of keyword search?
    Keyword search matches letters. If the handbook says "annual leave" and the
    employee asks about "vacation days", keyword search finds nothing. An
    embedding turns text into a list of numbers that represents its *meaning*,
    so "vacation days" lands close to "annual leave" in that number space and
    we find the right paragraph.

Why ChromaDB and not a Python list of vectors?
    Chroma writes to disk (./chroma_db), so ingesting once survives restarts,
    and it does the nearest-neighbour search with an index rather than
    comparing the query against every chunk one at a time.
"""

import shutil
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from backend.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL, MAX_DISTANCE, TOP_K

# Loading the embedding model takes a few seconds, so we do it once and reuse it.
_collection = None


def get_collection():
    """Open (or create) the on-disk Chroma collection."""
    global _collection
    if _collection is not None:
        return _collection

    # PersistentClient = saved to disk. The alternative, chromadb.Client(),
    # would be in-memory only and lost on every restart.
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )

    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        # Cosine similarity compares direction, not length, which is the right
        # measure for sentence embeddings. Chroma defaults to squared L2.
        metadata={"hnsw:space": "cosine"},
    )
    return _collection


def rebuild(chunks: List[Dict], batch_size: int = 100) -> int:
    """Wipe the collection and store the given chunks. Returns how many were stored.

    We rebuild from scratch rather than appending so that re-running ingest
    after editing a PDF does not leave stale duplicates behind.
    """
    global _collection

    # Delete the whole index directory rather than calling delete_collection().
    # Chroma leaves its on-disk index segments behind when a collection is
    # dropped, so repeated re-ingests would slowly fill the folder with orphans.
    # This is also simpler: there is no chance of stale chunks surviving.
    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    _collection = None

    collection = get_collection()

    # Chroma embeds in batches; small batches keep memory flat and let us
    # print progress on a big document set.
    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[f"chunk-{start + i}" for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[
                {"source": c["source"], "page": c["page"], "section": c["section"]}
                for c in batch
            ],
        )
        print(f"  -  embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    return collection.count()


def search(question: str, top_k: int = TOP_K) -> List[Dict]:
    """Find the chunks most similar in meaning to the question.

    Chunks further away than MAX_DISTANCE are dropped. Chroma always returns
    its top_k nearest neighbours even when nothing is relevant, so without this
    filter an off-topic question would still arrive at the LLM with four
    confident-looking but unrelated paragraphs attached.
    """
    collection = get_collection()
    if collection.count() == 0:
        return []

    response = collection.query(
        query_texts=[question],
        n_results=min(top_k, collection.count()),
    )

    results = []
    for text, metadata, distance in zip(
        response["documents"][0],
        response["metadatas"][0],
        response["distances"][0],
    ):
        if distance > MAX_DISTANCE:
            continue
        results.append(
            {
                "text": text,
                "source": metadata["source"],
                "page": metadata["page"],
                "section": metadata.get("section", ""),
                # Turn distance into a 0-1 "how relevant is this" score that is
                # easier to read in the UI than a raw cosine distance.
                "score": round(max(0.0, 1 - distance / 2), 3),
            }
        )
    return results


def stats() -> Dict:
    """Summary of what is currently indexed, for the UI sidebar."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return {"chunks": 0, "documents": []}

    stored = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in stored["metadatas"]})
    return {"chunks": total, "documents": sources}
