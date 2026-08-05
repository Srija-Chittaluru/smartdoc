"""The only place the interface talks to the backend.

Every function returns a `(data, error)` pair and never raises, so a page can
show a helpful message instead of a stack trace. Error text here is read by
users, so it says what happened - not which host failed.
"""

import json

import requests

from backend.config import API_URL

TIMEOUT = 60  # seconds
INDEXING_TIMEOUT = 300  # embedding a large PDF takes a while


def _request(method: str, path: str, payload=None, files=None, timeout=TIMEOUT):
    """Send one request. Returns (data, error) - exactly one of them is None."""
    try:
        url = f"{API_URL}{path}"
        if method == "post":
            response = requests.post(url, json=payload, files=files, timeout=timeout)
        elif method == "delete":
            response = requests.delete(url, timeout=timeout)
        else:
            response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.json(), None

    except requests.exceptions.ConnectionError:
        return None, "The document service isn't responding. Please try again in a moment."
    except requests.exceptions.Timeout:
        return None, "That request took too long. Please try again."
    except Exception:
        return None, "Something went wrong. Please try again."


# --- Asking questions --------------------------------------------------------


def get_health():
    """Which documents are indexed, and how many chunks."""
    return _request("get", "/health")


def get_config():
    """The active chunk size, models and vector-DB settings."""
    return _request("get", "/config")


def ask_question(question: str, history=None):
    """Ask the pipeline. Data holds `answer`, `citations` and `status`.

    `history` carries earlier turns so a follow-up can be understood.
    """
    return _request(
        "post", "/ask", payload={"question": question, "history": history or []}
    )


def ask_question_streamed(question: str, history=None):
    """Ask the pipeline and yield events as the answer is written.

    Yields the backend's own events - `start`, `token`, `final` - plus an
    `error` event of its own if the connection fails, so the caller only ever
    has to read events and never has to catch anything.
    """
    try:
        response = requests.post(
            f"{API_URL}/ask/stream",
            json={"question": question, "history": history or []},
            stream=True,
            timeout=TIMEOUT,
        )
        response.raise_for_status()

        for line in response.iter_lines(decode_unicode=True):
            # Server-sent events separate messages with blank lines; only the
            # "data:" lines carry anything.
            if line and line.startswith("data: "):
                yield json.loads(line[len("data: ") :])

    except requests.exceptions.ConnectionError:
        yield _stream_error("The document service isn't responding. Please try again in a moment.")
    except requests.exceptions.Timeout:
        yield _stream_error("That question took too long to answer. Please try again.")
    except Exception:
        yield _stream_error("Something went wrong while answering. Please try again.")


def _stream_error(message: str) -> dict:
    return {"type": "error", "message": message}


# --- Managing the library ----------------------------------------------------


def get_documents():
    """Table rows plus library-wide totals."""
    return _request("get", "/documents")


def get_document(name: str):
    """Detail for one document."""
    return _request("get", f"/documents/{name}")


def upload_document(filename: str, data: bytes):
    """Send one PDF to be saved and indexed.

    Sent one file per request rather than as a batch, so the page can report
    real progress through a multi-file upload instead of guessing.
    """
    files = [("files", (filename, data, "application/pdf"))]
    return _request("post", "/documents/upload", files=files, timeout=INDEXING_TIMEOUT)


def reindex_document(name: str):
    """Re-chunk and re-embed one document."""
    return _request("post", f"/documents/{name}/reindex", timeout=INDEXING_TIMEOUT)


def delete_document(name: str):
    """Remove one document from the index and from disk."""
    return _request("delete", f"/documents/{name}")


def clear_library():
    """Remove every document. Confirm before calling this."""
    return _request("delete", "/documents", timeout=INDEXING_TIMEOUT)


# --- Analytics ---------------------------------------------------------------


def get_analytics():
    """Dashboard figures: query stats, totals and index composition."""
    return _request("get", "/analytics")
