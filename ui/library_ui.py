"""The pieces that make up the Document Library page.

Kept out of views/library.py so that the page file reads as a short list of
what is on the page, and each piece here can be understood on its own.
"""

import streamlit as st

from ui import api

# Width of each table column, in the order they appear.
COLUMNS = [3.4, 0.8, 0.9, 1.7, 1.3, 0.5, 0.5, 0.5]

STATUS_CLASS = {
    "Indexed": "sd-tag--ok",
    "Not indexed": "sd-tag--warn",
    "File missing": "sd-tag--bad",
}


# --- Upload ------------------------------------------------------------------


def upload_panel():
    """Drag-and-drop upload for one or more PDFs, with real progress.

    Files are sent one at a time so the progress bar reflects actual work; a
    single batch request could only ever show nought then done.
    """
    st.markdown('<div class="sd-eyebrow">Add documents</div>', unsafe_allow_html=True)

    # The key changes after each upload, which is how the file list is cleared.
    round_number = st.session_state.get("upload_round", 0)
    files = st.file_uploader(
        "Drop PDFs here, or browse",
        type="pdf",
        accept_multiple_files=True,
        key=f"uploader-{round_number}",
    )

    if not files:
        return

    if st.button(f"Upload and index {len(files)} file(s)", type="primary"):
        _upload_all(files)


def _upload_all(files):
    """Upload each file in turn, then reload the page with the results."""
    progress = st.progress(0.0, text="Starting…")
    results = []

    for number, uploaded in enumerate(files, start=1):
        progress.progress(
            (number - 1) / len(files),
            text=f"Indexing {uploaded.name}  ({number} of {len(files)})",
        )
        data, error = api.upload_document(uploaded.name, uploaded.getvalue())

        if error:
            results.append({"name": uploaded.name, "status": "error", "detail": error})
        else:
            results.extend(data["results"])

    progress.progress(1.0, text="Done")

    # Remembered across the rerun so the table below refreshes and the results
    # are still shown.
    st.session_state.upload_results = results
    st.session_state.upload_round = st.session_state.get("upload_round", 0) + 1
    st.rerun()


# How each upload outcome is shown. The wording itself comes from the backend
# (library.UPLOAD_PROBLEMS), so a problem is described the same way everywhere;
# this only decides whether it reads as a success, a skip or a failure.
REJECTED_AS_WARNING = {"duplicate", "scanned", "encrypted", "empty_pdf"}
REJECTED_AS_ERROR = {"unsupported", "empty_file", "corrupted", "error"}


def show_upload_results():
    """Report what happened to each file in the last upload, then forget it."""
    results = st.session_state.pop("upload_results", None)
    if not results:
        return

    for result in results:
        name = result["name"]
        status = result["status"]
        detail = result.get("detail", "This file could not be indexed.")

        if status == "indexed":
            st.success(f"**{name}** — indexed into {result['chunks']} chunks.")
        elif status == "duplicate":
            st.warning(f"**{name}** — skipped, {detail}.")
        elif status in REJECTED_AS_WARNING:
            st.warning(f"**{name}** — {detail}")
        elif status in REJECTED_AS_ERROR:
            st.error(f"**{name}** — {detail}")
        else:
            # An outcome added to the backend but not yet described here still
            # reports something useful rather than nothing.
            st.warning(f"**{name}** — {detail}")


# --- Totals and toolbar ------------------------------------------------------


def totals_row(totals: dict):
    """Three tiles: how many documents, pages and chunks the library holds."""
    st.markdown(
        '<div class="sd-stats sd-stats--wide">'
        f'<div class="sd-stat"><b>{totals["documents"]}</b><span>Documents</span></div>'
        f'<div class="sd-stat"><b>{totals["pages"]}</b><span>Pages</span></div>'
        f'<div class="sd-stat"><b>{totals["chunks"]}</b><span>Chunks</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def toolbar():
    """Search box and sort order. Returns (query, newest_first)."""
    search, sort, danger = st.columns([3, 1.6, 1.6], vertical_alignment="bottom")

    query = search.text_input("Search documents", placeholder="Filter by file name…")
    order = sort.selectbox("Sort", ["A → Z", "Z → A"], label_visibility="visible")

    danger.markdown('<div class="sd-spacer"></div>', unsafe_allow_html=True)
    if danger.button("Clear entire library", width="stretch"):
        confirm_clear_library()

    return query, order == "Z → A"


def apply_search_and_sort(rows: list, query: str, descending: bool) -> list:
    """Filter by name, then order alphabetically."""
    if query:
        rows = [row for row in rows if query.lower() in row["name"].lower()]
    return sorted(rows, key=lambda row: row["name"].lower(), reverse=descending)


# --- The table ---------------------------------------------------------------


def table_header():
    header = ["Document", "Pages", "Chunks", "Indexed", "Status", "", "", ""]
    for column, label in zip(st.columns(COLUMNS), header):
        column.markdown(
            f'<div class="sd-th">{label}</div>', unsafe_allow_html=True
        )


def table_row(row: dict):
    """One document: its numbers, its status, and its three actions."""
    name, pages, chunks, indexed, status, view, redo, remove = st.columns(
        COLUMNS, vertical_alignment="center"
    )

    name.markdown(f'<div class="sd-td sd-td--name">{row["name"]}</div>', unsafe_allow_html=True)
    pages.markdown(f'<div class="sd-td sd-num">{row["pages"]}</div>', unsafe_allow_html=True)
    chunks.markdown(f'<div class="sd-td sd-num">{row["chunks"]}</div>', unsafe_allow_html=True)
    indexed.markdown(f'<div class="sd-td sd-muted">{row["indexed_at"]}</div>', unsafe_allow_html=True)
    status.markdown(
        f'<div class="sd-td"><span class="sd-tag {STATUS_CLASS[row["status"]]}">'
        f'{row["status"]}</span></div>',
        unsafe_allow_html=True,
    )

    if view.button("🔍", key=f"view-{row['name']}", help="View metadata"):
        show_metadata(row["name"])

    if redo.button("♻️", key=f"redo-{row['name']}", help="Re-index this document"):
        _reindex(row["name"])

    if remove.button("🗑️", key=f"del-{row['name']}", help="Delete this document"):
        confirm_delete(row["name"])


def _reindex(name: str):
    with st.spinner(f"Re-indexing {name}…"):
        data, error = api.reindex_document(name)

    if error or not data["ok"]:
        st.error(error or data["error"])
        return

    st.session_state.upload_results = [
        {"name": name, "status": "indexed", "chunks": data["chunks"]}
    ]
    st.rerun()


# --- Dialogs -----------------------------------------------------------------


@st.dialog("Document details")
def show_metadata(name: str):
    """Everything known about one document."""
    data, error = api.get_document(name)

    if error or not data["ok"]:
        st.error(error or data["error"])
        return

    document = data["document"]
    rows = [
        ("File name", document["name"]),
        ("Status", document["status"]),
        ("Pages in file", document["pages"]),
        ("Pages with text", document["indexed_pages"]),
        ("Chunks indexed", document["chunks"]),
        ("Indexed at", document["indexed_at"]),
        ("File size", f"{document['size_kb']} KB"),
    ]
    st.markdown(
        "".join(
            f'<div class="sd-kv"><span>{key}</span><b>{value}</b></div>'
            for key, value in rows
        ),
        unsafe_allow_html=True,
    )

    if document["sections"]:
        st.markdown('<div class="sd-eyebrow">Sections found</div>', unsafe_allow_html=True)
        st.markdown("\n".join(f"- {section}" for section in document["sections"]))
    else:
        st.caption("No section headings were detected in this document.")


@st.dialog("Delete this document?")
def confirm_delete(name: str):
    """Deleting removes the file as well as its chunks, so it is confirmed.

    Not asked for in the spec, but this cannot be undone from the interface.
    """
    st.write(f"**{name}** will be removed from the index and deleted from disk.")
    st.caption("This cannot be undone.")

    cancel, delete = st.columns(2)
    if cancel.button("Cancel", width="stretch"):
        st.rerun()
    if delete.button("Delete", type="primary", width="stretch"):
        data, error = api.delete_document(name)
        if error or not data["ok"]:
            st.error(error or data["error"])
        else:
            st.rerun()


@st.dialog("Clear the entire library?")
def confirm_clear_library():
    """Wipes every document. Guarded by typing the word, not just a click."""
    data, _ = api.get_documents()
    count = data["totals"]["documents"] if data else 0

    st.write(f"All **{count} documents** will be deleted from the index and from disk.")
    st.caption("This cannot be undone.")

    typed = st.text_input("Type DELETE to confirm")

    cancel, wipe = st.columns(2)
    if cancel.button("Cancel", width="stretch"):
        st.rerun()
    if wipe.button("Clear library", type="primary", width="stretch", disabled=typed != "DELETE"):
        with st.spinner("Clearing…"):
            result, error = api.clear_library()
        if error or not result["ok"]:
            st.error(error or result["error"])
        else:
            st.rerun()
