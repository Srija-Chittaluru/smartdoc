"""FastAPI backend. Exposes the RAG pipeline over HTTP.

Keeping the pipeline behind an API means the Streamlit UI is only a client -
a mobile app, a Slack bot or a curl script could use the exact same endpoints.

Run with:  uvicorn backend.main:app --reload
"""

import json
from typing import Optional

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from backend import library, rag, vector_store
from backend.config import (
    CHAT_MODEL,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    MAX_DISTANCE,
    RARE_DF_RATIO,
    RELATIVE_MARGIN,
    STRONG_MATCH,
    TOP_K,
)

app = FastAPI(
    title="SmartDoc API",
    description="Ask questions in plain English, get cited answers from your PDFs.",
    version="1.0.0",
)


class Turn(BaseModel):
    question: str
    answer: str


class AskRequest(BaseModel):
    question: str = Field(..., description="A plain-English question.")
    # Optional and empty by default, so an existing client that sends only a
    # question keeps working exactly as before.
    history: list[Turn] = Field(
        default_factory=list,
        description="Earlier turns, so a follow-up question can be understood.",
    )
    # The filename of one indexed document, to answer from that document alone.
    # None - the default - searches the whole library, exactly as before.
    source: Optional[str] = Field(
        default=None,
        description="Restrict the answer to this document. None searches all of them.",
    )


class Citation(BaseModel):
    source: str
    page: int
    section: str
    score: float
    text: str
    # Which retriever found this chunk: "semantic" by meaning, "keyword" by a
    # literal value in the question, "both" when the two agreed. `score` is a
    # cosine similarity in every case - never a BM25 or fusion score. Defaulted
    # so an older cached reply without the field still validates.
    match: str = "semantic"


class AskResponse(BaseModel):
    answer: str
    citations: list[Citation] = []
    status: str
    # Added, not required: an older client that ignores this field is unaffected,
    # but without declaring it here the response model would strip it out.
    retrieval_seconds: Optional[float] = None


@app.get("/health")
def health():
    """Is the API up, and does it have anything indexed?

    Answers even when the vector store is unreadable - the interface needs to
    tell the difference between "the service is down" and "the index is broken".
    """
    try:
        indexed = vector_store.stats()
    except Exception:
        return {
            "status": "index_unavailable",
            "chunks_indexed": 0,
            "documents": [],
            "error": "The document index could not be opened. It may need rebuilding.",
        }

    return {
        "status": "ok",
        "chunks_indexed": indexed["chunks"],
        "documents": indexed["documents"],
    }


@app.get("/config")
def show_config():
    """The settings behind the answers. Handy for the demo and for debugging."""
    return {
        "chunk_size": CHUNK_SIZE,
        "chunk_overlap": CHUNK_OVERLAP,
        "top_k": TOP_K,
        "embedding_model": EMBEDDING_MODEL,
        "chat_model": CHAT_MODEL,
        "vector_db": "ChromaDB (persisted to ./chroma_db)",
        # Retrieval is hybrid: cosine distance and BM25, each gated on its own
        # scale. Worth exposing because these are the numbers that decide when
        # the answer is "I don't know", and they are corpus-dependent.
        "retrieval": "hybrid (semantic + BM25, fused by RRF)",
        "max_distance": MAX_DISTANCE,
        "relative_margin": RELATIVE_MARGIN,
        "strong_match": STRONG_MATCH,
        "rare_df_ratio": RARE_DF_RATIO,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    """Answer a question from the indexed documents.

    Always returns HTTP 200 with a `status` field rather than raising, so the
    UI can show a helpful message instead of a stack trace.
    """
    return rag.answer_question(request.question, _history_of(request), request.source)


def _history_of(request: AskRequest) -> list:
    """The earlier turns as plain dicts, which is what rag expects."""
    return [turn.model_dump() for turn in request.history]


@app.post("/ask/stream")
def ask_stream(request: AskRequest):
    """The same answer, streamed as the model writes it.

    Sent as server-sent events: one `data: {...}` line per event. /ask is left
    exactly as it was, so nothing that already works has to change.
    """

    def events():
        for event in rag.answer_stream(
            request.question, _history_of(request), request.source
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(
        events(),
        media_type="text/event-stream",
        # Stops a proxy from buffering the stream into one lump.
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# --- Document library ---------------------------------------------------------
# Like /ask, these return HTTP 200 with an `ok` flag rather than raising, so the
# UI can report what happened without handling status codes.


def _done(**fields):
    return {"ok": True, **fields}


def _failed(message: str):
    return {"ok": False, "error": message}


@app.get("/documents")
def documents():
    """Every document with its pages, chunks, index date and status."""
    try:
        return {"documents": library.list_documents(), "totals": library.totals()}
    except Exception:
        # A missing or corrupt chroma_db must not turn the page into a stack
        # trace - the upload panel above the table still needs to work.
        return {
            "documents": [],
            "totals": {"documents": 0, "pages": 0, "chunks": 0},
            "error": "The document index could not be opened. It may need rebuilding.",
        }


@app.get("/documents/{name}")
def document(name: str):
    """Detail for one document, for the metadata view."""
    try:
        return _done(document=library.document_metadata(name))
    except (ValueError, FileNotFoundError) as error:
        return _failed(str(error))


@app.post("/documents/upload")
async def upload(files: list[UploadFile] = File(...)):
    """Save and index one or more PDFs.

    Reports per file rather than failing the whole batch, so one bad PDF in a
    drag-and-drop of ten does not lose the other nine.
    """
    results = []

    for upload_file in files:
        name = upload_file.filename or "unnamed.pdf"
        try:
            data = await upload_file.read()

            # Checked before anything is written, so a bad file never lands on
            # disk. Covers unsupported, empty, corrupted, encrypted, 0-page.
            problem = library.check_pdf(data, name)
            if problem:
                results.append(
                    {
                        "name": name,
                        "status": problem,
                        "detail": library.UPLOAD_PROBLEMS[problem],
                    }
                )
                continue

            duplicate = library.find_duplicate(data, name)
            if duplicate:
                results.append({"name": name, "status": "duplicate", "detail": duplicate})
                continue

            library.save_upload(data, name)
            chunks = library.index_document(name)

            # A readable PDF whose pages are all images: the file is fine, but
            # there is no text to answer from. Kept rather than deleted so it
            # can be re-indexed after being run through OCR.
            if chunks == 0:
                results.append(
                    {
                        "name": name,
                        "status": "scanned",
                        "detail": library.UPLOAD_PROBLEMS["scanned"],
                    }
                )
            else:
                results.append({"name": name, "status": "indexed", "chunks": chunks})

        except Exception:
            results.append(
                {
                    "name": name,
                    "status": "error",
                    "detail": "This file could not be indexed.",
                }
            )

    return _done(results=results)


@app.post("/documents/{name}/reindex")
def reindex(name: str):
    """Re-chunk and re-embed one document, replacing its existing chunks."""
    try:
        return _done(chunks=library.index_document(name))
    except ValueError as error:
        return _failed(str(error))
    except FileNotFoundError:
        return _failed(f"{name} is no longer on disk, so it cannot be re-indexed.")
    except Exception as error:
        return _failed(f"Could not re-index {name}: {error}")


@app.delete("/documents/{name}")
def delete_document(name: str):
    """Remove one document from the index and from disk."""
    try:
        library.delete_document(name)
        return _done()
    except ValueError as error:
        return _failed(str(error))
    except Exception as error:
        return _failed(f"Could not delete {name}: {error}")


@app.delete("/documents")
def clear_library():
    """Empty the whole library. The UI confirms before calling this."""
    try:
        library.clear_library()
        return _done()
    except Exception as error:
        return _failed(f"Could not clear the library: {error}")
