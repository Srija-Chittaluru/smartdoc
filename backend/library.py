"""Managing the document library: add, re-index, delete, and per-document stats.

This module only *calls* the existing chunker and vector store. How chunking,
embedding and retrieval work is unchanged - see chunker.py and vector_store.py.

Chunk IDs written here are namespaced by filename ("handbook.pdf::0"), so one
document can be replaced without touching any other.
"""

import hashlib
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List, Optional

from pypdf import PdfReader

from backend.chunker import chunk_pdf
from backend.config import DOCUMENTS_DIR
from backend.vector_store import chunk_metadata, get_collection, invalidate

# Every way an upload can fail, and what to tell the person who uploaded it.
# Kept in one place so the API and the interface cannot describe the same
# problem in two different ways.
UPLOAD_PROBLEMS = {
    "unsupported": "Only PDF files are supported.",
    "empty_file": "The file is empty — there is nothing to index.",
    "corrupted": "This file could not be read as a PDF. It may be corrupted.",
    "empty_pdf": "This PDF has no pages.",
    "encrypted": "This PDF is password-protected, so its text cannot be read.",
    "scanned": (
        "No text could be extracted. This looks like a scan or a photo — "
        "SmartDoc needs a PDF with a real text layer."
    ),
}


def check_pdf(data: bytes, filename: str) -> Optional[str]:
    """Return a problem code from UPLOAD_PROBLEMS, or None if the file is usable.

    Run before the file is written anywhere, so a corrupt or empty upload never
    lands in documents/ and never has to be cleaned up afterwards.
    """
    if not filename.lower().endswith(".pdf"):
        return "unsupported"

    if not data:
        return "empty_file"

    # A file can be named .pdf and not be one. Every real PDF starts with the
    # %PDF marker, allowing for a little leading junk.
    if b"%PDF" not in data[:1024]:
        return "unsupported"

    try:
        reader = PdfReader(BytesIO(data))
        page_total = len(reader.pages)
    except Exception:
        return "corrupted"

    if reader.is_encrypted:
        return "encrypted"
    if page_total == 0:
        return "empty_pdf"
    return None


def safe_name(name: str) -> str:
    """Check that a document name really is a plain PDF filename.

    Names arrive from the browser, so this must never be able to point outside
    documents/. Anything containing a path is rejected rather than repaired,
    because a silently "cleaned" path is how traversal bugs survive.
    """
    if name != Path(name).name or not name.lower().endswith(".pdf"):
        raise ValueError(f"Not a valid document name: {name}")
    return name


# Page counts, keyed by file path. Opening a PDF to count its pages is the
# slowest thing list_documents() does, and it does it for every file on every
# call - so the result is kept until the file itself changes.
_page_cache: Dict[str, tuple] = {}


def page_count(path: Path) -> int:
    """Total pages in a PDF, read from the file rather than from the index.

    The index only knows about pages that held extractable text, so a scanned
    page would be missing from that count.

    Cached against the file's modification time and size, so an edited or
    replaced file is always re-read and a stale count cannot survive.
    """
    try:
        stat = path.stat()
    except OSError:
        return 0

    fingerprint = (stat.st_mtime, stat.st_size)
    cached = _page_cache.get(str(path))
    if cached and cached[0] == fingerprint:
        return cached[1]

    try:
        pages = len(PdfReader(str(path)).pages)
    except Exception:
        pages = 0

    _page_cache[str(path)] = (fingerprint, pages)
    return pages


def _digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def find_duplicate(data: bytes, filename: str) -> Optional[str]:
    """Say why an upload is a duplicate, or return None if it is new.

    Checked two ways, because the same document often arrives under a new name:
    by filename, then by content.
    """
    if (DOCUMENTS_DIR / filename).exists():
        return "already in the library"

    incoming = _digest(data)
    for existing in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        if _digest(existing.read_bytes()) == incoming:
            return f"identical to {existing.name}"
    return None


def save_upload(data: bytes, filename: str) -> Path:
    """Write an uploaded PDF into documents/ and return its path."""
    path = DOCUMENTS_DIR / safe_name(filename)
    DOCUMENTS_DIR.mkdir(exist_ok=True)
    path.write_bytes(data)
    return path


def remove_from_index(name: str) -> None:
    """Delete every chunk belonging to one document."""
    get_collection().delete(where={"source": safe_name(name)})
    # The keyword index was built from the chunks that have just gone.
    invalidate()


def index_document(name: str) -> int:
    """Chunk one PDF and store it, replacing whatever was indexed before.

    Removing first is what makes this safe to call twice - a re-index cannot
    leave stale chunks from the previous version behind.
    """
    name = safe_name(name)
    remove_from_index(name)

    chunks = chunk_pdf(DOCUMENTS_DIR / name)
    if not chunks:
        return 0

    indexed_at = datetime.now().strftime("%Y-%m-%d %H:%M")
    get_collection().add(
        ids=[f"{name}::{i}" for i in range(len(chunks))],
        documents=[c["text"] for c in chunks],
        # Shared with ingest so a PDF added here carries the same table-cell
        # fields as one added by ingest.py.
        metadatas=[chunk_metadata(c, indexed_at=indexed_at) for c in chunks],
    )
    invalidate()
    return len(chunks)


def delete_document(name: str) -> None:
    """Remove a document from the index and from disk."""
    name = safe_name(name)
    remove_from_index(name)
    path = DOCUMENTS_DIR / name
    if path.exists():
        path.unlink()


def clear_library() -> None:
    """Empty the index and delete every PDF."""
    for path in DOCUMENTS_DIR.glob("*.pdf"):
        path.unlink()

    # Chroma refuses delete(where={}), so the IDs have to be listed explicitly.
    collection = get_collection()
    ids = collection.get()["ids"]
    if ids:
        collection.delete(ids=ids)

    invalidate()
    _page_cache.clear()


def _describe_status(indexed: bool, on_disk: bool) -> str:
    if indexed and on_disk:
        return "Indexed"
    if on_disk:
        return "Not indexed"
    return "File missing"


def _why_not_indexed(path: Path) -> str:
    """Say what is stopping this PDF from being indexed.

    Only called for the handful of rows that are not indexed, so it can afford
    to open the file. The same problems are described in UPLOAD_PROBLEMS, and the
    wording is reused rather than rewritten.
    """
    try:
        reader = PdfReader(str(path))
        if reader.is_encrypted:
            return UPLOAD_PROBLEMS["encrypted"]
        if len(reader.pages) == 0:
            return UPLOAD_PROBLEMS["empty_pdf"]
        # A few pages are enough to tell a text PDF from a scan, and keeps this
        # quick on long documents.
        if not any((page.extract_text() or "").strip() for page in reader.pages[:3]):
            return UPLOAD_PROBLEMS["scanned"]
    except Exception:
        return UPLOAD_PROBLEMS["corrupted"]

    return (
        "The file is readable, it just has no chunks in the index yet — "
        "it was added to the documents folder rather than uploaded here. "
        "Use the recycle button to index it."
    )


def _status_reason(status: str, path: Optional[Path]) -> str:
    """The hover explanation behind a status tag."""
    if status == "Indexed":
        return "Chunked and stored in the index, so questions can be answered from it."
    if status == "File missing":
        return "Its chunks are still in the index, but the PDF is no longer on disk."
    return _why_not_indexed(path)


def list_documents() -> List[Dict]:
    """One row per document, for the library table.

    Covers three cases, because they need different actions: indexed and
    present, present but never indexed, and indexed but the file has since gone.
    """
    stored = get_collection().get(include=["metadatas"])

    from_index: Dict[str, Dict] = {}
    for meta in stored["metadatas"]:
        row = from_index.setdefault(
            meta["source"], {"chunks": 0, "pages": 0, "indexed_at": ""}
        )
        row["chunks"] += 1
        row["pages"] = max(row["pages"], meta.get("page", 0))
        # Chunks written before indexed_at existed have no stamp; keep the
        # newest one found rather than an empty string.
        row["indexed_at"] = max(row["indexed_at"], meta.get("indexed_at") or "")

    on_disk = {path.name: path for path in DOCUMENTS_DIR.glob("*.pdf")}

    rows = []
    for name in sorted(set(from_index) | set(on_disk), key=str.lower):
        indexed = from_index.get(name)
        path = on_disk.get(name)
        status = _describe_status(bool(indexed), bool(path))
        rows.append(
            {
                "name": name,
                "pages": page_count(path) if path else (indexed or {}).get("pages", 0),
                "chunks": indexed["chunks"] if indexed else 0,
                "indexed_at": (indexed or {}).get("indexed_at") or "—",
                "status": status,
                "reason": _status_reason(status, path),
                "size_kb": round(path.stat().st_size / 1024) if path else 0,
            }
        )
    return rows


def totals() -> Dict:
    """Library-wide counts for the summary tiles."""
    rows = list_documents()
    return {
        "documents": len(rows),
        "pages": sum(row["pages"] for row in rows),
        "chunks": sum(row["chunks"] for row in rows),
    }


def document_metadata(name: str) -> Dict:
    """Everything known about one document, for the metadata dialog."""
    name = safe_name(name)
    row = next((r for r in list_documents() if r["name"] == name), None)
    if row is None:
        raise FileNotFoundError(f"{name} is not in the library")

    stored = get_collection().get(where={"source": name}, include=["metadatas"])
    sections = sorted({m.get("section", "") for m in stored["metadatas"] if m.get("section")})

    return {
        **row,
        "indexed_pages": len({m["page"] for m in stored["metadatas"]}),
        "sections": sections[:25],
    }
