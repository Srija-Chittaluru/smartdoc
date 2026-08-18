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

import math
import shutil
from typing import Dict, List

import chromadb
from chromadb.utils import embedding_functions

from backend import document_scope, lexical
from backend.config import (
    CANDIDATE_K,
    CHROMA_DIR,
    COLLECTION_NAME,
    EMBEDDING_MODEL,
    LEXICAL_K,
    MAX_DISTANCE,
    RELATIVE_MARGIN,
    RRF_K,
    STRONG_MATCH,
    TOP_K,
    W_LEXICAL,
    W_SEMANTIC,
)
_collection = None
_embedder = None

def get_collection():
    """Open (or create) the on-disk Chroma collection."""
    global _collection, _embedder
    if _collection is not None:
        return _collection

    client = chromadb.PersistentClient(path=str(CHROMA_DIR))

    embedder = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )
    _embedder = embedder
    _collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        embedding_function=embedder,
        metadata={"hnsw:space": "cosine"},
    )
    return _collection

def chunk_metadata(chunk: Dict, **extra) -> Dict:
    """Flatten one chunk into the metadata Chroma stores beside it.

    A table row's cells are stored as their own fields, prefixed so a column
    called "Source" or "Page" cannot overwrite the citation fields. Chroma only
    accepts flat scalars here, which is why the cells are flattened rather than
    nested.

    Shared with library.index_document so a PDF added through the UI carries the
    same fields as one added by ingest.py.
    """
    meta = {
        "source": chunk["source"],
        "page": chunk["page"],
        "section": chunk["section"],
    }
    for key, value in chunk.get("fields", {}).items():
        meta[f"col_{key}"] = value
    meta.update(extra)
    return meta

def invalidate() -> None:
    """Forget what was derived from the collection, after it has been changed.

    The keyword index and the corpus snapshot are both built from Chroma's
    contents. Anything that adds or removes chunks has to call this, or a search
    would score the question against documents that are no longer there.
    """
    global _corpus_cache
    _corpus_cache = None
    lexical.reset()
    document_scope.reset()

def rebuild(chunks: List[Dict], batch_size: int = 100) -> int:
    """Wipe the collection and store the given chunks. Returns how many were stored.

    We rebuild from scratch rather than appending so that re-running ingest
    after editing a PDF does not leave stale duplicates behind.
    """
    global _collection, _corpus_cache

    if CHROMA_DIR.exists():
        shutil.rmtree(CHROMA_DIR)
    _collection = None
    invalidate()

    collection = get_collection()

    for start in range(0, len(chunks), batch_size):
        batch = chunks[start : start + batch_size]
        collection.add(
            ids=[f"chunk-{start + i}" for i in range(len(batch))],
            documents=[c["text"] for c in batch],
            metadatas=[chunk_metadata(c) for c in batch],
        )
        print(f"  -  embedded {min(start + batch_size, len(chunks))}/{len(chunks)}")

    return collection.count()

_corpus_cache = None

def _corpus():
    """Every stored chunk as (ids, documents, metadatas), in one fixed order.

    The count check is a backstop: callers are expected to call invalidate()
    after changing the collection, and this catches the case where one forgets.
    """
    global _corpus_cache
    if _corpus_cache is None or len(_corpus_cache[1]) != get_collection().count():
        stored = get_collection().get(include=["documents", "metadatas"])
        _corpus_cache = (stored["ids"], stored["documents"], stored["metadatas"])
        lexical.reset()
    return _corpus_cache

def _cosine_distances(question: str, ids: List[str]) -> List[float]:
    """The true cosine distance from the question to each of these chunks.

    A keyword-only hit never came back from the vector query, so it carries no
    distance - and the meter beside a citation in the UI is a similarity meter.
    Rather than invent a number, or show a BM25 score where a similarity
    belongs, the real distance is computed from the embeddings Chroma already
    stored. Only the handful of keyword hits are fetched.

    Chroma's "cosine" space is 1 - cosine similarity, and this matches it.
    """
    if not ids:
        return []

    get_collection()  # populates _embedder
    query = list(_embedder([question])[0])
    query_norm = math.sqrt(sum(v * v for v in query)) or 1.0

    stored = get_collection().get(ids=ids, include=["embeddings"])
    vectors = dict(zip(stored["ids"], stored["embeddings"]))

    distances = []
    for chunk_id in ids:
        vector = vectors[chunk_id]
        dot = sum(a * b for a, b in zip(query, vector))
        norm = math.sqrt(sum(v * v for v in vector)) or 1.0
        distances.append(1.0 - dot / (query_norm * norm))
    return distances

def lexical_candidates(
    question: str, limit: int = LEXICAL_K, source: str = None
) -> List[Dict]:
    """Chunks the keyword leg admits, best BM25 first.

    Empty unless the question names something specific enough to identify a
    record - see lexical.LexicalIndex.value_terms. That restriction is what
    keeps this leg from weakening the out-of-scope guard: an ordinary question
    offers it no anchor, so the leg contributes nothing and the distance filters
    decide alone, exactly as before.
    """
    ids, documents, metadatas = document_scope.restrict(_corpus(), source)
    if not documents:
        return []

    positions = document_scope.keyword_index(source, documents).candidates(question, limit)
    if not positions:
        return []

    chosen = [ids[position] for position in positions]
    distances = _cosine_distances(question, chosen)

    return [
        {
            "text": documents[position],
            "source": metadatas[position]["source"],
            "page": metadatas[position]["page"],
            "section": metadatas[position].get("section", ""),
            "score": round(max(0.0, 1 - distances[order] / 2), 3),
            "match": "keyword",
        }
        for order, position in enumerate(positions)
    ]


def _fuse(semantic: List[Dict], keyword: List[Dict], top_k: int) -> List[Dict]:
    """Order the admitted chunks by Reciprocal Rank Fusion.

    Ordering only. Admission was already decided by each leg on its own scale,
    which is what keeps MAX_DISTANCE meaningful - an RRF score is a number
    around 0.016 with no relation to cosine distance, and gating on it would
    throw away the separation those thresholds were measured to capture.

    A chunk both legs found collects a term from each and rises above one that
    only appeared in either. Its "match" becomes "both", so the UI can say why.
    """
    fused: Dict[str, Dict] = {}

    for weight, results in ((W_SEMANTIC, semantic), (W_LEXICAL, keyword)):
        for rank, chunk in enumerate(results, start=1):
            entry = fused.get(chunk["text"])
            if entry is None:
                entry = dict(chunk)
                entry["_rrf"] = 0.0
                fused[chunk["text"]] = entry
            elif entry["match"] != chunk["match"]:
                entry["match"] = "both"
            entry["_rrf"] += weight / (RRF_K + rank)
    ranked = sorted(fused.values(), key=lambda c: (-c["_rrf"], -c["score"]))[:top_k]
    for chunk in ranked:
        chunk.pop("_rrf", None)
        chunk.pop("distance", None)
    return ranked

def semantic_candidates(question: str, total: int, source: str = None) -> List[Dict]:
    """The meaning-based leg, gated exactly as it always was.

    Two stages, because one threshold cannot answer both questions that matter.

    Stage 1 - absolute. Chroma always returns its nearest neighbours even when
    nothing is relevant, so an off-topic question would otherwise arrive at the
    LLM with four confident-looking but unrelated paragraphs attached. Anything
    further away than MAX_DISTANCE is not about the subject at all, and goes.

    Stage 2 - relative. Surviving stage 1 only means a chunk is on-topic; it
    says nothing about whether it is *as good as* what else we found. A chunk
    at 0.73 and a chunk at 0.28 both pass, and both used to be handed to the
    model as equally-weighted numbered sources. Here we keep only the chunks
    within RELATIVE_MARGIN of the best one, so how many come back depends on
    how many are genuinely good rather than being fixed at four.
    """
    response = get_collection().query(
        query_texts=[question],
        n_results=min(CANDIDATE_K, total),
        # One stored field decides the whole scope. `source` is the filename
        # every chunk was saved with, so restricting to a document needs no
        # second collection - Chroma never looks at the other documents' chunks.
        where={"source": source} if source else None,
    )
    candidates = [
        (distance, text, metadata)
        for text, metadata, distance in zip(
            response["documents"][0],
            response["metadatas"][0],
            response["distances"][0],
        )
        if distance <= MAX_DISTANCE
    ]
    if not candidates:
        return []
    best = candidates[0][0]
    kept = [c for c in candidates if c[0] <= best + RELATIVE_MARGIN]

    return [
        {
            "text": text,
            "source": metadata["source"],
            "page": metadata["page"],
            "section": metadata.get("section", ""),
            "score": round(max(0.0, 1 - distance / 2), 3),
            "match": "semantic",
            "distance": distance,
        }
        for distance, text, metadata in kept
    ]

def search(
    question: str,
    top_k: int = TOP_K,
    lexical_question: str = None,
    source: str = None,
) -> List[Dict]:
    """Find the chunks that answer the question, by meaning and by keyword.

    Two retrievers, because they fail in opposite directions. The embedding
    finds "annual leave" for "vacation days" and cannot separate "04-02-2026"
    from "22-08-2026"; BM25 does the reverse.

    Each leg admits chunks on its own scale - MAX_DISTANCE and RELATIVE_MARGIN
    for meaning, a rare value term for keywords. Nothing is admitted on a fused
    score: an RRF number sits around 0.016 and has no relation to cosine
    distance, so gating on it would discard the separation those thresholds were
    measured to capture. Fusion decides order only.

    `lexical_question` is the text to match literally, when it differs from the
    text to embed. A follow-up is rewritten with the previous question attached
    so it can be embedded at all, but feeding that to BM25 would let the earlier
    question's keywords outvote the current one. See rag.retrieve.

    `source` restricts the search to one document, and defaults to none, which
    is the original library-wide behaviour. Both legs are restricted, and so is
    the identifier guard below, so a question the chosen document cannot answer
    comes back empty rather than being answered out of a different document.
    """
    total = get_collection().count()
    if total == 0:
        return []

    asked = lexical_question or question
    semantic = semantic_candidates(question, total, source)
    keyword = lexical_candidates(asked, source=source)
    scoped = document_scope.restrict(_corpus(), source)
    index = document_scope.keyword_index(source, scoped[1])
    if (
        semantic
        and lexical.mentions_value(asked)
        and semantic[0]["distance"] > STRONG_MATCH
    ):
        semantic = []
    if index.absent_identifiers(asked):
        semantic = []
    return _fuse(semantic, keyword, top_k)

def stats() -> Dict:
    """Summary of what is currently indexed, for the UI sidebar."""
    collection = get_collection()
    total = collection.count()
    if total == 0:
        return {"chunks": 0, "documents": []}

    stored = collection.get(include=["metadatas"])
    sources = sorted({m["source"] for m in stored["metadatas"]})
    return {"chunks": total, "documents": sources}
