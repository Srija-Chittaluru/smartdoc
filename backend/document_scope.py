"""Restricting retrieval to one chosen document.

The Ask page has a document selector. "All Documents" is the original
behaviour and takes exactly the path it always did - this module is not
involved at all. Choosing one PDF narrows the search to that PDF.

Both retrievers have to be narrowed, not one. Narrowing only the vector leg
would leave BM25 free to return another document's chunks, and those would be
cited under a question the reader had scoped to a single file.

There is no second index and no second collection. Every chunk was already
stored with a `source` field (see vector_store.chunk_metadata), so the vector
leg hands that field to Chroma as a `where` filter and the keyword leg scores
only the rows carrying it. Everything after retrieval - fusion, the relevance
gates, the citations - is the shared code, untouched.
"""

from typing import Dict, List, Optional, Sequence, Tuple

from backend import lexical

Corpus = Tuple[List[str], List[str], List[Dict]]

# One BM25 index per document, keyed by document name.
#
# A document cannot borrow the library's index, and not only for speed. BM25
# scores a term by how rare it is and how long the average chunk runs, and both
# of those are properties of the collection being searched. Measured over the
# whole library, a term that appears in every chunk of the chosen PDF still
# looks rare, and would pinpoint nothing.
_indexes: Dict[str, lexical.LexicalIndex] = {}
_sizes: Dict[str, int] = {}


def reset() -> None:
    """Forget every per-document index.

    Called from vector_store.invalidate, alongside the reset of the library-wide
    index, so a document that has just been re-indexed or deleted cannot still
    be scored against the chunks it used to have.
    """
    _indexes.clear()
    _sizes.clear()


def restrict(corpus: Corpus, source: Optional[str]) -> Corpus:
    """The corpus rows belonging to one document, in the same shape as the whole.

    `corpus` is (ids, documents, metadatas) as vector_store keeps it. All three
    lists are filtered together, so a position still means the same chunk in
    each of them and the callers' existing indexing carries over unchanged.

    No source means no restriction: the whole corpus comes back as it is.
    """
    if not source:
        return corpus

    ids, documents, metadatas = corpus
    kept = [
        position
        for position, meta in enumerate(metadatas)
        if meta.get("source") == source
    ]
    return (
        [ids[position] for position in kept],
        [documents[position] for position in kept],
        [metadatas[position] for position in kept],
    )


def keyword_index(source: Optional[str], documents: Sequence[str]) -> lexical.LexicalIndex:
    """The BM25 index to score against: the whole library, or one document.

    Rebuilt when the document's chunk count changes, which is the same cheap
    staleness check lexical.get_index uses for the library-wide index.
    """
    if not source:
        return lexical.get_index(documents)

    if _indexes.get(source) is None or _sizes.get(source) != len(documents):
        _indexes[source] = lexical.LexicalIndex(documents)
        _sizes[source] = len(documents)
    return _indexes[source]
