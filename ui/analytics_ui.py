"""The tiles, charts and table that make up the Analytics page.

Charts are plain st.bar_chart / st.line_chart over a small DataFrame. Nothing
here computes statistics - the backend does that in metrics.py, so the numbers
on this page and the numbers in the API are always the same.
"""

import pandas as pd
import streamlit as st

ACCENT = "#2563eb"


def _tile(value, label: str, small: bool = False) -> str:
    """One statistic. `small` is for text values, which need room to breathe."""
    extra = " sd-stat--text" if small else ""
    return (
        f'<div class="sd-stat{extra}"><b>{value}</b><span>{label}</span></div>'
    )


def kpi_tiles(summary: dict, totals: dict):
    """The six headline numbers, in two rows of three."""
    st.markdown(
        '<div class="sd-stats sd-stats--wide">'
        + _tile(totals["documents"], "Documents indexed")
        + _tile(totals["chunks"], "Chunks")
        + _tile(summary["questions_asked"], "Questions asked")
        + "</div>"
        '<div class="sd-stats sd-stats--wide">'
        + _tile(f"{summary['avg_seconds']:.2f}s", "Avg response time")
        + _tile(f"{summary['avg_similarity']:.2f}", "Avg similarity score")
        + _tile(summary["most_queried_document"], "Most queried document", small=True)
        + "</div>",
        unsafe_allow_html=True,
    )


def _section(title: str):
    st.markdown(f'<div class="sd-eyebrow">{title}</div>', unsafe_allow_html=True)


def chunks_per_document(data: list):
    """How the index is distributed across documents."""
    _section("Chunks per document")
    if not data:
        st.caption("Nothing is indexed yet.")
        return

    frame = pd.DataFrame(data)
    st.bar_chart(
        frame,
        x="document",
        y="chunks",
        x_label="Chunks",
        y_label="",
        color=ACCENT,
        horizontal=True,   # file names are long; horizontal keeps them readable
        height=max(160, 42 * len(frame)),
    )


def questions_over_time(data: list):
    """How many questions were asked each day."""
    _section("Questions over time")
    if not data:
        st.caption("No questions have been asked yet.")
        return

    frame = pd.DataFrame(data)

    # A line needs two points to be a line. With a single day of history a bar
    # is the honest way to show it.
    if len(frame) < 2:
        st.bar_chart(frame, x="date", y="questions", y_label="Questions", color=ACCENT)
    else:
        st.line_chart(frame, x="date", y="questions", y_label="Questions", color=ACCENT)


def similarity_distribution(data: list):
    """Where retrieval confidence actually lands."""
    _section("Similarity score distribution")

    frame = pd.DataFrame(data)
    if frame["queries"].sum() == 0:
        st.caption("No answered questions yet.")
        return

    st.bar_chart(
        frame, x="range", y="queries", x_label="Top similarity score",
        y_label="Questions", color=ACCENT,
    )


def document_usage(data: list):
    """Which documents actually answer questions."""
    _section("Document usage")
    if not data:
        st.caption("No documents have been cited yet.")
        return

    frame = pd.DataFrame(data)
    st.bar_chart(
        frame,
        x="document",
        y="citations",
        x_label="Times cited",
        y_label="",
        color=ACCENT,
        horizontal=True,
        height=max(160, 42 * len(frame)),
    )


def recent_queries(entries: list):
    """The last few questions, with how they went."""
    _section("Recent queries")
    if not entries:
        st.caption("No questions have been asked yet.")
        return

    frame = pd.DataFrame(
        [
            {
                "When": entry["at"].replace("T", "  "),
                "Question": entry["question"],
                "Answered": "Yes" if entry["status"] == "ok" else "No",
                "Top score": entry.get("top_score"),
                "Seconds": entry.get("seconds"),
            }
            for entry in entries
        ]
    )
    st.dataframe(frame, hide_index=True, width="stretch")
