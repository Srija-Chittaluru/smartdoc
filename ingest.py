"""Build the search index from the PDFs in documents/.

Run this once after adding or changing PDFs:

    python ingest.py
"""

from backend.chunker import chunk_all_pdfs
from backend.config import CHUNK_OVERLAP, CHUNK_SIZE, DOCUMENTS_DIR, EMBEDDING_MODEL
from backend.vector_store import rebuild


def main():
    print(f"\nReading PDFs from {DOCUMENTS_DIR}")
    print(f"Chunk size {CHUNK_SIZE} chars, overlap {CHUNK_OVERLAP} chars\n")

    try:
        chunks = chunk_all_pdfs(DOCUMENTS_DIR)
    except FileNotFoundError as error:
        print(f"{error}\nAdd some PDFs to documents/ and run this again.")
        return

    if not chunks:
        print("No text could be extracted from any PDF. Nothing to index.")
        return

    print(f"\nEmbedding {len(chunks)} chunks with {EMBEDDING_MODEL}")
    print("(first run downloads the model, about 90 MB)\n")
    stored = rebuild(chunks)

    print(f"\nDone. {stored} chunks stored in ChromaDB.")
    print("Start the app with:  ./run.sh\n")


if __name__ == "__main__":
    main()
