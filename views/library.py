"""Page 2 - Document Library. All document management lives here.

The page is a short list of what appears on it; each piece is built in
ui/library_ui.py.
"""

import streamlit as st

from ui import api, components, library_ui

components.page_header(
    "📚",
    "Document Library",
    "Upload, inspect and re-index the documents SmartDoc answers from.",
)

library_ui.show_upload_results()
library_ui.upload_panel()

st.divider()

documents, error = api.get_documents()

if error:
    st.error(error)
    st.stop()

# The index can be unreadable while the service itself is fine. Uploading still
# works in that state, so the panel above stays usable and only the table stops.
if documents.get("error"):
    st.warning(documents["error"])

library_ui.totals_row(documents["totals"])

query, newest_first = library_ui.toolbar()
rows = library_ui.apply_search_and_sort(documents["documents"], query, newest_first)

if not documents["documents"]:
    st.info("The library is empty. Upload a PDF above to get started.")
elif not rows:
    st.info(f"No documents match “{query}”.")
else:
    library_ui.table_header()
    for row in rows:
        library_ui.table_row(row)
