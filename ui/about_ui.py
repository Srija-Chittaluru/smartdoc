"""The three sections of the About page.

Content lives in the lists below, so the wording can be changed without
touching any markup.
"""

import streamlit as st

INTRO = (
    "SmartDoc is an AI assistant for your company's documents. Finding one "
    "policy detail normally means opening several PDFs and scrolling through "
    "pages that all look alike. Instead, upload your documents once and ask "
    "questions in plain English, the way you would ask a colleague. Every "
    "answer is drawn only from your own documents, and shows the document, "
    "page and section it came from — so you can verify it rather than take it "
    "on trust."
)

CAPABILITIES = [
    ("📄", "Upload PDF Documents", "Drag your PDFs in and they are ready to search in seconds."),
    ("💬", "Ask Questions in Natural Language", "Type a question the way you would say it out loud."),
    ("🔍", "AI-Powered Semantic Search", "Finds answers by meaning, so “vacation days” matches “annual leave”."),
    ("📑", "View Source Citations", "Every answer names the document, page and section behind it."),
    ("📚", "Manage Document Library", "Review, re-index or remove documents whenever you need to."),
]

BENEFITS = [
    ("⚡", "Save Time", "Quickly find information without manually reading large PDFs."),
    ("✅", "Reliable Answers", "Every response is backed by document citations."),
    ("🚀", "Improve Productivity", "Employees can access information instantly using AI."),
]


def _title(text: str):
    st.markdown(f'<h2 class="sd-section-title">{text}</h2>', unsafe_allow_html=True)


def _cards(items: list) -> str:
    """Build one grid of cards. Each card is an icon, a name and one sentence."""
    return "".join(
        '<div class="sd-card">'
        f'<div class="sd-card-icon">{icon}</div>'
        f"<h4>{name}</h4><p>{sentence}</p>"
        "</div>"
        for icon, name, sentence in items
    )


def intro():
    _title("About SmartDoc")
    st.markdown(f'<p class="sd-lead">{INTRO}</p>', unsafe_allow_html=True)


def capabilities():
    _title("What You Can Do")
    st.markdown(f'<div class="sd-cards">{_cards(CAPABILITIES)}</div>', unsafe_allow_html=True)


def benefits():
    _title("Why SmartDoc?")
    st.markdown(f'<div class="sd-cards">{_cards(BENEFITS)}</div>', unsafe_allow_html=True)
