"""Small render helpers shared by the pages.

Each function draws one piece of the interface and returns nothing. Keeping
them here means a page file reads as a list of what is on the page.
"""

import html

import streamlit as st
import streamlit.components.v1 as components

from ui import api


def page_header(icon: str, title: str, subtitle: str):
    """The band at the top of every page: mark, title, one line of context."""
    st.markdown(
        f'<div class="sd-hero"><div class="sd-mark">{icon}</div>'
        f"<div><h1>{title}</h1><p>{subtitle}</p></div></div>",
        unsafe_allow_html=True,
    )


def operator_note(detail: str, command: str):
    """Put the technical cause and its fix one click away.

    Whoever runs the app needs the command; whoever reads the app does not.
    Written as inline markdown rather than st.code, which keeps the copy
    button off it.
    """
    with st.expander("Technical details"):
        st.markdown(f"{detail}\n\nResolved by running `{command}` in the project folder.")


def sidebar_status():
    """A compact index-health panel, shown under the navigation on every page."""
    health, error = api.get_health()

    if error:
        st.sidebar.markdown(
            '<div class="sd-status"><span class="sd-dot sd-dot--down"></span>'
            "Service unavailable</div>",
            unsafe_allow_html=True,
        )
        with st.sidebar:
            if st.button("Try again", width="stretch"):
                st.rerun()
            operator_note(error, "./run.sh")
        return

    if health["chunks_indexed"] == 0:
        st.sidebar.markdown(
            '<div class="sd-status"><span class="sd-dot sd-dot--warn"></span>'
            "No documents yet</div>",
            unsafe_allow_html=True,
        )
        with st.sidebar:
            operator_note("The search index is empty.", "python ingest.py")
        return

    st.sidebar.markdown(
        '<div class="sd-status"><span class="sd-dot sd-dot--live"></span>'
        "Index ready</div>"
        '<div class="sd-stats">'
        f'<div class="sd-stat"><b>{len(health["documents"])}</b><span>Documents</span></div>'
        f'<div class="sd-stat"><b>{health["chunks_indexed"]}</b><span>Chunks</span></div>'
        "</div>",
        unsafe_allow_html=True,
    )


def confidence_label(score: float) -> str:
    """Turn a 0-1 similarity score into words, so it needs no explaining."""
    if score >= 0.80:
        return "High confidence"
    if score >= 0.72:
        return "Moderate confidence"
    return "Low confidence"


def confidence_badge(citations: list):
    """A single badge for the answer as a whole, from its best-matching source.

    The per-source meters inside each citation say how good *that* excerpt was;
    this says how well supported the answer is overall, without having to open
    anything.
    """
    if not citations:
        return

    best = max(float(c["score"]) for c in citations)
    label = confidence_label(best)
    tone = "ok" if best >= 0.80 else "warn" if best >= 0.72 else "bad"

    st.markdown(
        f'<span class="sd-badge sd-badge--{tone}">'
        f'<span class="sd-badge-dot"></span>{label} · {best:.2f}</span>',
        unsafe_allow_html=True,
    )


def render_citations(citations: list):
    """Show the sources behind an answer.

    Each one is expandable and carries the document name, the page number, the
    section, a similarity meter and the excerpt the answer came from.
    """
    if not citations:
        return

    plural = "source" if len(citations) == 1 else "sources"
    st.markdown(
        f'<div class="sd-sources-label">{len(citations)} {plural}</div>',
        unsafe_allow_html=True,
    )

    for number, citation in enumerate(citations, start=1):
        label = f"[{number}]  {citation['source']}  ·  page {citation['page']}"
        if citation["section"]:
            label += f"  ·  {citation['section']}"

        with st.expander(label):
            score = float(citation["score"])
            st.markdown(
                '<div class="sd-cite-meter">'
                f'<span class="sd-cite-strength">{confidence_label(score)}</span>'
                '<span class="sd-cite-track">'
                f'<span class="sd-cite-fill" style="width:{score * 100:.0f}%"></span>'
                "</span>"
                f'<span class="sd-cite-strength">{score:.2f}</span>'
                "</div>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f'<p class="sd-quote">{citation["text"]}</p>', unsafe_allow_html=True
            )


def answer_footer(seconds: float, citations: list, retrieval_seconds=None):
    """One muted line under an answer: how long it took, and how well it matched.

    Retrieval and total are shown separately because they answer different
    questions - retrieval is the search, the rest is the model writing.
    """
    parts = [f"Answered in {seconds:.1f}s"]
    if retrieval_seconds is not None:
        parts.append(f"retrieval {retrieval_seconds * 1000:.0f}ms")
    if citations:
        best = max(float(c["score"]) for c in citations)
        parts.append(f"top score {best:.2f}")
        parts.append(f"{len(citations)} sources")
    st.markdown(
        f'<div class="sd-answer-note">{"  ·  ".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def copy_answer(text: str):
    """A copy icon at the end of the answer. One click copies it.

    Only the browser can reach the clipboard, so this is a tiny HTML component
    rather than a Streamlit widget. The answer is carried in a hidden textarea
    instead of being pasted into the script, which means quotes and newlines in
    the answer cannot break the JavaScript.
    """
    components.html(
        COPY_WIDGET.replace("__TEXT__", html.escape(text)),
        height=34,
    )


# Two ways to copy, because the modern one is not always permitted inside an
# embedded frame: try the async Clipboard API, and fall back to execCommand,
# which is old but still works everywhere. The icon confirms with a tick.
COPY_WIDGET = """
<textarea id="source" readonly>__TEXT__</textarea>
<button id="copy" title="Copy answer" aria-label="Copy answer">&#128203;</button>

<style>
  body { margin: 0; background: transparent; }
  #source { position: fixed; top: 0; left: 0; height: 1px; width: 1px;
            opacity: 0; border: none; resize: none; }
  #copy {
    font-size: 15px;
    line-height: 1;
    padding: 4px 6px;
    border: none;
    border-radius: 6px;
    background: transparent;
    opacity: .5;
    cursor: pointer;
    transition: opacity .15s ease, background .15s ease;
  }
  #copy:hover { opacity: 1; background: rgba(15, 23, 42, .06); }
</style>

<script>
  const source = document.getElementById("source");
  const button = document.getElementById("copy");

  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(source.value);
    } catch (error) {
      source.select();
      document.execCommand("copy");
    }
    button.innerHTML = "&#10003;";
    button.title = "Copied";
    setTimeout(() => {
      button.innerHTML = "&#128203;";
      button.title = "Copy answer";
    }, 1400);
  });
</script>
"""
