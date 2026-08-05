"""Loads the stylesheet.

Called once from app.py. Because the entry script reruns before every page,
one call there styles every page.
"""

from pathlib import Path

import streamlit as st

STYLESHEET = Path(__file__).resolve().parent.parent / "static" / "styles.css"


def apply_theme():
    """Inject static/styles.css into the page.

    Read on every rerun rather than cached, so editing the CSS only needs a
    browser refresh.
    """
    st.markdown(f"<style>{STYLESHEET.read_text()}</style>", unsafe_allow_html=True)
