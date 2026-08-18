"""SmartDoc - entry point.

This file does three things and nothing else: set the page up, load the theme,
and declare the navigation. Each page lives in views/, and the code they share
lives in ui/, so no single file has to be read to understand another.

The interface never imports the RAG pipeline. It talks to the FastAPI backend
over HTTP, which keeps the two independent.

Run with:  streamlit run app.py
"""

import streamlit as st

from ui.theme import apply_theme

st.set_page_config(
    page_title="SmartDoc — Ask your documents",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

apply_theme()

# Ask Documents is the landing page: default=True makes it the one that opens.
pages = [
    st.Page("views/ask.py", title="Ask Documents", icon="💬", default=True),
    st.Page("views/library.py", title="Document Library", icon="📚"),
    st.Page("views/about.py", title="About", icon="ℹ️"),
]

navigation = st.navigation(pages)

# Drawn after the nav links, so it sits below them in the sidebar. Imported
# here rather than at the top because it calls the backend, and that should
# happen once the page is already configured.
from ui.components import sidebar_status  # noqa: E402

sidebar_status()

st.sidebar.markdown(
    '<p class="sd-footnote">Answers are drawn only from the indexed documents, '
    "and each one carries a page-level citation. If the documents don't cover a "
    "question, SmartDoc says so rather than guessing.</p>",
    unsafe_allow_html=True,
)

navigation.run()
