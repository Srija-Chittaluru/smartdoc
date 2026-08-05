"""The diagrams and panels for the About page.

Everything here is drawn with plain HTML and the project's own CSS classes, so
the diagrams match the rest of the interface and need no drawing library.

Numbers are read from the live /config response rather than written into the
text, so this page cannot drift out of date when a setting changes.
"""

import streamlit as st

# Each stage of the pipeline: title, what happens, and what does it.
# `{...}` placeholders are filled from the running configuration.
PIPELINE = [
    ("PDF", "The text layer is read page by page, so every chunk keeps an exact page number.", "pypdf"),
    ("Chunking", "Split into {chunk_size}-character pieces with {chunk_overlap} characters of overlap, breaking at section and sentence boundaries.", "LangChain"),
    ("Embeddings", "Each chunk becomes 384 numbers representing its meaning. Runs locally — no API key, no cost.", "{embedding_model}"),
    ("Vector Database", "Chunks and their metadata are stored on disk and indexed for nearest-neighbour search by cosine distance.", "ChromaDB"),
    ("Retriever", "The question is embedded the same way; the closest {top_k} chunks are returned, and anything too distant is dropped.", "ChromaDB"),
    ("LLM", "The surviving chunks are handed over as numbered sources, with a prompt that forbids outside knowledge.", "{chat_model}"),
    ("Answer with Citations", "Two to four sentences with [1] markers, each resolving to a document, page and section.", "SmartDoc"),
]

TECH_STACK = [
    ("FastAPI", "API layer", "Serves the pipeline over HTTP, so the interface is only a client — a Slack bot or a script could use the same endpoints."),
    ("LangChain", "Text splitting", "RecursiveCharacterTextSplitter breaks pages at the most natural boundary available rather than at a fixed offset."),
    ("Sentence Transformers", "Embeddings", "Runs all-MiniLM-L6-v2 on the machine itself. Re-indexing costs nothing and works offline."),
    ("ChromaDB", "Vector store", "Persists to disk, searches with an index rather than one comparison at a time, and keeps metadata beside each vector — which is what makes citations possible."),
    ("Streamlit", "Interface", "The four pages you are looking at. It talks to the API over HTTP and holds no AI logic."),
]

ARCHITECTURE = [
    ("Interface", "Streamlit · port 8501", ["Ask Documents", "Document Library", "Analytics", "About"]),
    ("API", "FastAPI · port 8000", ["/ask", "/documents", "/analytics", "/health"]),
    ("Pipeline", "Python modules", ["chunker.py", "vector_store.py", "rag.py", "library.py"]),
    ("Storage", "Local and remote", ["documents/ (PDFs)", "chroma_db/ (vectors)", "OpenAI (generation)"]),
]


def section(title: str):
    st.markdown(f'<div class="sd-eyebrow">{title}</div>', unsafe_allow_html=True)


def overview():
    """What the project is and what problem it solves."""
    st.markdown(
        '<div class="sd-welcome">'
        "<h3>What SmartDoc is</h3>"
        "<p>Employees lose time hunting for answers buried in company PDFs. "
        "SmartDoc lets them ask in plain English and get a short answer "
        "<b>with a citation showing the exact document, page and section it came "
        "from</b> — so the answer can be verified rather than trusted.</p>"
        "<p style='margin-top:.7rem'>It searches by <b>meaning, not keywords</b>: "
        "the handbook says “annual leave” while people ask about “vacation days”, "
        "and those match because their embeddings sit close together. When the "
        "documents do not cover a question, SmartDoc says so instead of "
        "inventing an answer.</p>"
        "</div>",
        unsafe_allow_html=True,
    )


def architecture_diagram():
    """Four layers, top to bottom, showing what talks to what."""
    section("Architecture")

    blocks = []
    for index, (name, detail, items) in enumerate(ARCHITECTURE):
        chips = "".join(f'<span class="sd-chip">{item}</span>' for item in items)
        blocks.append(
            '<div class="sd-layer">'
            f'<div class="sd-layer-head"><b>{name}</b><span>{detail}</span></div>'
            f'<div class="sd-chips">{chips}</div>'
            "</div>"
        )
        if index < len(ARCHITECTURE) - 1:
            blocks.append('<div class="sd-link"><span>HTTP</span></div>' if index == 0
                          else '<div class="sd-link"></div>')

    st.markdown(f'<div class="sd-arch">{"".join(blocks)}</div>', unsafe_allow_html=True)
    st.caption(
        "Only the API layer touches the pipeline. The interface never imports it, "
        "which is why either side can change without the other."
    )


def pipeline_flow(settings: dict):
    """The seven stages, PDF through to a cited answer."""
    section("The RAG pipeline")

    stages = []
    for number, (title, description, tech) in enumerate(PIPELINE, start=1):
        stages.append(
            '<div class="sd-step">'
            f'<div class="sd-step-num">{number}</div>'
            f'<div class="sd-step-body"><h4>{title}</h4>'
            f"<p>{description.format(**settings)}</p></div>"
            f'<div class="sd-step-tech">{tech.format(**settings)}</div>'
            "</div>"
        )
        if number < len(PIPELINE):
            stages.append('<div class="sd-step-arrow">↓</div>')

    st.markdown(f'<div class="sd-flow">{"".join(stages)}</div>', unsafe_allow_html=True)


def tech_stack():
    """What each dependency is there for."""
    section("Tech stack")

    cards = "".join(
        '<div class="sd-tech">'
        f'<div class="sd-tech-head"><b>{name}</b><span>{role}</span></div>'
        f"<p>{why}</p>"
        "</div>"
        for name, role, why in TECH_STACK
    )
    st.markdown(f'<div class="sd-tech-grid">{cards}</div>', unsafe_allow_html=True)
