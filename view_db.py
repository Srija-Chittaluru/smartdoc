"""Look inside the ChromaDB index from the command line.

    .venv/bin/python view_db.py                 summary + every chunk, previewed
    .venv/bin/python view_db.py --full          show each chunk's whole text
    .venv/bin/python view_db.py --vectors       show the embedding numbers too
    .venv/bin/python view_db.py --find leave    only chunks containing "leave"

Read-only: it never writes to the database.
"""

import argparse
from collections import Counter

from backend.config import CHROMA_DIR, COLLECTION_NAME, EMBEDDING_MODEL
from backend.vector_store import get_collection


def main():
    parser = argparse.ArgumentParser(description="Inspect the SmartDoc vector index.")
    parser.add_argument("--full", action="store_true", help="print each chunk in full")
    parser.add_argument("--vectors", action="store_true", help="print embedding numbers")
    parser.add_argument("--find", metavar="TEXT", help="only chunks containing TEXT")
    args = parser.parse_args()

    collection = get_collection()
    total = collection.count()

    print(f"\nDatabase   {CHROMA_DIR}")
    print(f"Collection {COLLECTION_NAME}")
    print(f"Model      {EMBEDDING_MODEL}")
    print(f"Chunks     {total}")

    if total == 0:
        print("\nNothing indexed yet. Run:  .venv/bin/python ingest.py\n")
        return

    wanted = ["documents", "metadatas"] + (["embeddings"] if args.vectors else [])
    stored = collection.get(include=wanted)

    # Chunks per document, so you can see how the index is distributed.
    counts = Counter(meta["source"] for meta in stored["metadatas"])
    print("\nChunks per document")
    for name, count in counts.most_common():
        print(f"  {count:>3}  {name}")

    print(f"\n{'-' * 78}")

    shown = 0
    for index, chunk_id in enumerate(stored["ids"]):
        text = stored["documents"][index]
        meta = stored["metadatas"][index]

        if args.find and args.find.lower() not in text.lower():
            continue
        shown += 1

        print(f"\n[{chunk_id}]  {meta['source']}  ·  page {meta['page']}")
        if meta.get("section"):
            print(f"  section: {meta['section']}")
        if meta.get("indexed_at"):
            print(f"  indexed: {meta['indexed_at']}")

        body = text if args.full else text[:220].replace("\n", " ") + (
            "…" if len(text) > 220 else ""
        )
        print(f"  {body}")

        if args.vectors:
            vector = [round(float(value), 4) for value in stored["embeddings"][index]]
            print(f"  vector ({len(vector)} numbers): {vector[:8]} …")

    if args.find:
        print(f"\n{shown} of {total} chunks contain {args.find!r}")
    print()


if __name__ == "__main__":
    main()
