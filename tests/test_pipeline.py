"""Tests for the parts of the pipeline that do not need an API key.

Run with:  .venv/bin/python -m pytest -v

These cover chunking, retrieval and the input guards. Answer generation is not
tested here because it would cost money and depend on the network; the guards
that run *before* the API call are tested instead.
"""

import pytest

from backend import rag, vector_store
from backend.chunker import chunk_pdf, clean_text, heading_text, looks_like_heading
from backend.config import CHUNK_SIZE, DOCUMENTS_DIR, MAX_QUESTION_LENGTH


# --- Heading detection -------------------------------------------------------
# Pages arrive as Markdown, so a heading is either "## Title" or a fully bold
# line. Both forms appear in the sample documents.

@pytest.mark.parametrize(
    "line",
    ["## 2. Annual Leave", "### 3.1 Scope", "# IT SECURITY POLICY", "**Working from Another Country**"],
)
def test_recognises_headings(line):
    assert looks_like_heading(line)


@pytest.mark.parametrize(
    "line",
    [
        "30 minutes of inactivity.",                      # sentence fragment
        "Employees receive 12 days of paid sick leave,",   # mid-sentence
        "Passwords must be at least 14 characters long and include upper case letters",
        # A bold lead-in is not a heading: the sentence carries on after it.
        "**Standard Equipment:** Available to all team members.",
    ],
)
def test_rejects_non_headings(line):
    assert not looks_like_heading(line)


def test_heading_markers_are_stripped_for_the_citation():
    assert heading_text("### Global Accessibility Definition") == "Global Accessibility Definition"
    assert heading_text("**2. Annual Leave**") == "2. Annual Leave"


# --- Cleaning ----------------------------------------------------------------

def test_joins_wrapped_lines_into_one_paragraph():
    raw = "Standard working hours are 9:30 AM\nto 6:30 PM, Monday to Friday."
    assert clean_text(raw) == "Standard working hours are 9:30 AM to 6:30 PM, Monday to Friday."


def test_repairs_words_hyphenated_across_a_line_break():
    """The converter marks a soft hyphen as struck through, so it arrives as ~~-~~."""
    assert "compensation" in clean_text("Annual compen ~~-~~\n\nsation is reviewed.")


def test_keeps_struck_through_text_but_drops_the_markers():
    assert clean_text("Contact ~~People Ops~~ HelpLab.") == "Contact People Ops HelpLab."


def test_keeps_heading_attached_to_its_section():
    cleaned = clean_text("**2. Annual Leave**\n\nEmployees get 21 days\n\nper calendar year.")
    assert cleaned == "**2. Annual Leave**\nEmployees get 21 days per calendar year."


def test_two_blank_lines_start_a_new_paragraph():
    """One blank line is a wrapped sentence; two is a real paragraph break."""
    cleaned = clean_text("Working hours are\n\n9:30 to 6:30.\n\n\nLeave accrues monthly.")
    assert cleaned == "Working hours are 9:30 to 6:30.\n\nLeave accrues monthly."


# --- Chunking ----------------------------------------------------------------

def test_chunks_carry_citation_metadata():
    chunks = chunk_pdf(DOCUMENTS_DIR / "employee_handbook.pdf")
    assert chunks, "expected at least one chunk"

    for chunk in chunks:
        assert chunk["source"] == "employee_handbook.pdf"
        assert chunk["page"] >= 1
        assert chunk["text"].strip()


def test_chunks_respect_the_configured_size():
    chunks = chunk_pdf(DOCUMENTS_DIR / "employee_handbook.pdf")
    # The splitter may overshoot slightly when a single word cannot be broken.
    assert all(len(c["text"]) <= CHUNK_SIZE * 1.1 for c in chunks)


# --- Retrieval ---------------------------------------------------------------
# These need the index built first: python ingest.py

def test_finds_the_right_document_for_a_relevant_question():
    hits = vector_store.search("How many days of annual leave do I get?")
    assert hits, "expected a match - did you run `python ingest.py`?"
    assert hits[0]["source"] == "employee_handbook.pdf"
    assert "leave" in hits[0]["text"].lower()


def test_matches_meaning_not_just_keywords():
    """"vacation" never appears in the handbook; "annual leave" does."""
    hits = vector_store.search("vacation days")
    assert hits
    assert hits[0]["source"] == "employee_handbook.pdf"


@pytest.mark.parametrize(
    "question",
    [
        "How do I bake sourdough bread?",
        "What is the capital of France?",
        # This one shares the word "training" with the onboarding guide, so it
        # slipped through an earlier, looser distance threshold.
        "How do I train a puppy?",
        "What is the weather tomorrow?",
    ],
)
def test_returns_nothing_for_an_out_of_scope_question(question):
    assert vector_store.search(question) == []


def test_keeps_finding_vaguely_phrased_in_scope_questions():
    """The threshold must not be so strict that short real queries get refused."""
    for question in ["vacation days", "home office allowance", "notice period"]:
        assert vector_store.search(question), f"wrongly refused: {question}"


# --- Input guards (these run before any API call) ----------------------------

@pytest.mark.parametrize("question", ["", "   ", None])
def test_empty_question_is_rejected_politely(question):
    result = rag.answer_question(question)
    assert result["status"] == "empty_question"
    assert result["citations"] == []


def test_very_long_question_is_rejected():
    result = rag.answer_question("leave policy " * 500)
    assert result["status"] == "question_too_long"
    assert str(MAX_QUESTION_LENGTH) in result["answer"]


def test_out_of_scope_question_refuses_without_calling_the_model():
    result = rag.answer_question("Who won the 1998 football World Cup?")
    assert result["status"] == "no_match"
    assert result["answer"] == rag.NO_ANSWER
    assert result["citations"] == []
