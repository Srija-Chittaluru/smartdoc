"""Tests for the parts of the pipeline that do not need an API key.

Run with:  .venv/bin/python -m pytest -v

These cover chunking, retrieval and the input guards. Answer generation is not
tested here because it would cost money and depend on the network; the guards
that run *before* the API call are tested instead.
"""

import pytest

from backend import lexical, rag, vector_store
from backend.chunker import (
    chunk_pdf,
    clean_text,
    drop_cell_spill,
    find_section,
    heading_text,
    looks_like_heading,
    restore_table_header,
    row_fields,
)
from backend.config import CHUNK_SIZE, DOCUMENTS_DIR, MAX_QUESTION_LENGTH


# The library is not fixed - documents get swapped in and out - so tests that
# need a real PDF ask for one rather than naming it. Naming a file made four
# tests fail the moment the sample documents were replaced, which says nothing
# about the pipeline.
def any_pdf():
    for path in sorted(DOCUMENTS_DIR.glob("*.pdf")):
        if chunk_pdf(path):
            return path
    pytest.skip("no readable PDF in the documents folder")


@pytest.fixture(scope="module")
def sample_chunks():
    return chunk_pdf(any_pdf())


# --- Heading detection -------------------------------------------------------
# Pages arrive as Markdown, so a heading is either "## Title" or a fully bold
# line. Both forms appear in the sample documents.

@pytest.mark.parametrize(
    "line",
    [
        "## 2. Annual Leave",
        "### 3.1 Scope",
        "# IT SECURITY POLICY",
        "**Working from Another Country**",
        # A numbered heading arrives as two bold runs, not one.
        "**2** **Travel expenses**",
    ],
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

def test_chunks_carry_citation_metadata(sample_chunks):
    assert sample_chunks, "expected at least one chunk"

    source = sample_chunks[0]["source"]
    for chunk in sample_chunks:
        assert chunk["source"] == source
        assert chunk["page"] >= 1
        assert chunk["text"].strip()


def test_a_section_carries_onto_the_pages_it_continues_on():
    """A table or section running past a page break leaves the later pages with
    no heading of their own. Those chunks must still name the section."""
    continuation = "Mileage can be claimed at 45p per mile."

    assert find_section(continuation, continuation) == ""
    assert find_section(continuation, continuation, "2 Travel expenses") == "2 Travel expenses"


def test_a_continuing_table_gets_its_real_header_back():
    """Page two of a table has no header, so the converter promotes its first
    data row into one. That row must go back to being data."""
    page_two = "|Legal|Supplier performance review|02-06-2026|\n|---|---|---|\n|Marketing|Event logistics|05-09-2026|"

    restored = restore_table_header(page_two, ["Office Function", "Activity", "Target Date"])

    assert restored.startswith("|Office Function|Activity|Target Date|")
    # The promoted row is still present, below the separator, as data.
    assert "|Legal|Supplier performance review|02-06-2026|" in restored
    assert restored.count("Supplier performance review") == 1


def test_table_rows_are_not_joined_into_one_another():
    """Rows are joined the way wrapped prose is, they stop being rows."""
    rows = "|Legal|Low|Completed|\n|Marketing|Medium|In Progress|"
    assert clean_text(rows).count("\n") >= 1


def test_chunks_respect_the_configured_size(sample_chunks):
    # The splitter may overshoot slightly when a single word cannot be broken.
    assert all(len(c["text"]) <= CHUNK_SIZE * 1.1 for c in sample_chunks)


# --- Table pages -------------------------------------------------------------
# A column too wide for the page is emitted as loose fragments above the table
# rather than inside it. They repeat on every page and answer nothing.

def test_cell_spill_is_dropped():
    page = (
        "# **Operations Register**\n\ntarget date.\n\n"
        "Vendor or internal team coordination required.\n\n"
        "|Facilities|Quarterly inspection|01-01-2026|"
    )
    cleaned = drop_cell_spill(page)

    assert "target date." not in cleaned
    assert "Vendor or internal team" not in cleaned
    # The rows and the document's own heading are not fragments.
    assert "|Facilities|Quarterly inspection|01-01-2026|" in cleaned
    assert "# **Operations Register**" in cleaned


def test_a_real_description_survives_the_spill_filter():
    """Long enough to be prose, so it is content rather than a stray cell."""
    page = (
        "Internal office-management register covering facilities, IT, finance, "
        "procurement, HR operations, security and administration.\n\n"
        "|Legal|Contract review|22-08-2026|"
    )
    assert "Internal office-management register" in drop_cell_spill(page)


def test_a_row_is_split_into_its_cells():
    columns = ["Office Function", "Request / Activity", "Target Date"]
    piece = "|IT Support|Laptop replacement and<br>configuration|04-02-2026|"

    assert row_fields(piece, columns) == {
        "office_function": "IT Support",
        "request_activity": "Laptop replacement and configuration",
        "target_date": "04-02-2026",
    }


def test_a_chunk_keeping_its_header_row_is_still_one_record():
    """The header is a row by shape but holds column names, not a record."""
    columns = ["Office Function", "Target Date"]
    piece = "|Office Function|Target Date|\n|Legal|22-08-2026|"

    assert row_fields(piece, columns) == {
        "office_function": "Legal",
        "target_date": "22-08-2026",
    }


@pytest.mark.parametrize(
    "piece",
    [
        # Two records: no single row to name.
        "|Legal|22-08-2026|\n|Marketing|05-09-2026|",
        # Wrong shape for these columns - a second table on the same page.
        "|Legal|Contract review|22-08-2026|",
        "Ordinary prose, no row at all.",
    ],
)
def test_no_fields_when_the_chunk_is_not_one_record(piece):
    assert row_fields(piece, ["Office Function", "Target Date"]) == {}


def test_register_rows_each_carry_their_cells():
    chunks = chunk_pdf(DOCUMENTS_DIR / "corporate_office_operations_register.pdf")
    rows = [c for c in chunks if c["fields"]]

    assert rows, "expected the register's rows to be parsed into fields"
    # Every target date is distinct: the rows are separate records that an
    # embedding cannot tell apart, which is why the cells are stored.
    dates = [c["fields"]["target_date"] for c in rows]
    assert len(set(dates)) == len(dates)
    assert all(c["fields"]["owner"] for c in rows)


# --- Retrieval ---------------------------------------------------------------
# These need the index built first: python ingest.py

def test_finds_the_chunk_a_question_was_taken_from():
    """Self-calibrating: the question is built from a chunk that is indexed, so
    this tests retrieval rather than which documents happen to be in the folder."""
    documents = vector_store._corpus()[1]
    longest = max(documents, key=len)
    # A distinctive run of words from the middle, away from any column prefix.
    words = " ".join(longest.split()).split(" ")
    probe = " ".join(words[len(words) // 3 :][:12])

    hits = vector_store.search(probe)
    assert hits, "expected a match - did you run `python ingest.py`?"
    assert any(probe.lower() in " ".join(h["text"].split()).lower() for h in hits)


def test_matches_meaning_not_just_keywords():
    """A word the corpus never uses must still reach the right subject.

    This is what the embedding is for, and what the keyword leg must not break:
    if the lexical gate ever admitted or rejected on ordinary words, a question
    phrased entirely in synonyms would come back empty.
    """
    index = lexical.get_index(vector_store._corpus()[1])
    probe = next(
        (w for w in ["grievance", "vacation", "sabbatical", "misconduct"]
         if w not in index.document_frequency),
        None,
    )
    if probe is None:
        pytest.skip("no out-of-vocabulary probe word available for this corpus")

    assert vector_store.search(f"What is the {probe} process?"), (
        f"{probe!r} is in none of the documents, so only the embedding can "
        f"answer it - the keyword leg must not have blocked it"
    )


# --- Hybrid retrieval --------------------------------------------------------
# The keyword leg exists because an embedding cannot represent a symbol:
# "04-02-2026" and "22-08-2026" differ in no way it captures, so the row holding
# the date asked about ranked 79th of 232. BM25 ranked it first.

def a_stored_date():
    """A target date that really is in the index, so the test cannot go stale."""
    for metadata in vector_store._corpus()[2]:
        if metadata.get("col_target_date"):
            return metadata["col_target_date"]
    pytest.skip("no table rows with a target date are indexed")


def test_an_exact_date_finds_the_row_that_holds_it():
    date = a_stored_date()
    hits = vector_store.search(f"What is due on {date}?")

    assert hits, "expected a match - did you run `python ingest.py`?"
    assert date in hits[0]["text"]
    assert hits[0]["match"] in ("keyword", "both")


def test_the_semantic_leg_alone_would_have_missed_it():
    """The reason the keyword leg is here, asserted rather than assumed."""
    date = a_stored_date()
    question = f"What is due on {date}?"
    total = vector_store.get_collection().count()

    alone = vector_store.semantic_candidates(question, total)
    assert not any(date in chunk["text"] for chunk in alone[:4])
    assert any(date in chunk["text"] for chunk in vector_store.search(question))


def test_a_keyword_hit_is_scored_as_a_similarity_not_a_bm25_score():
    """The meter in the UI is a similarity meter, so the number behind it has to
    be one - a BM25 score or an RRF score there would be meaningless."""
    date = a_stored_date()
    hits = vector_store.search(f"What is due on {date}?")
    keyword = [h for h in hits if h["match"] in ("keyword", "both")]

    assert keyword
    for hit in keyword:
        assert 0.0 <= hit["score"] <= 1.0
        # Cross-check against the distance Chroma itself reports for that chunk.
        ids = [
            chunk_id
            for chunk_id, text in zip(*vector_store._corpus()[:2])
            if text == hit["text"]
        ]
        distance = vector_store._cosine_distances(f"What is due on {date}?", ids)[0]
        assert abs(hit["score"] - round(max(0.0, 1 - distance / 2), 3)) < 0.01
        # An exact keyword match is not a perfect similarity, and must not claim
        # to be. The old code hard-coded 1.0 here.
        assert hit["score"] < 1.0


def test_a_date_we_do_not_hold_answers_nothing():
    """Better than the four chunks that happen to sit nearest."""
    assert vector_store.search("What is due on 31-12-2099?") == []


def test_an_identifier_we_do_not_hold_answers_nothing():
    """The case no distance threshold can catch: a near-miss code embeds as
    closely as the real one - 0.266 against 0.267 - so only the absence of the
    literal string distinguishes them."""
    assert vector_store.search("What does policy Calfus-ISMS-PL-99 say?") == []


def test_a_bare_number_is_not_treated_as_an_identifier():
    """"45" is a quantity, not the name of a thing. A question mentioning one the
    corpus never states must still be answered from the policy."""
    assert not lexical.looks_like_identifier("45")
    assert not lexical.looks_like_identifier("2024")
    assert lexical.looks_like_identifier("04-02-2026")
    assert lexical.looks_like_identifier("calfus-isms-pl-15")


def test_an_ordinary_question_never_reaches_the_keyword_leg():
    """No value-shaped term means no anchor, so the distance filters decide
    alone and the semantic behaviour is exactly what it was."""
    question = "What are the travel entitlements?"
    assert vector_store.lexical_candidates(question) == []
    assert all(hit["match"] == "semantic" for hit in vector_store.search(question))


def test_a_common_capitalised_phrase_is_not_an_anchor():
    """"Internal Committee" is in 26 chunks - a subject, not an identifier. If it
    admitted chunks, it would drag 26 of them past the distance guard."""
    index = lexical.get_index(vector_store._corpus()[1])
    assert index.value_terms("Who is on the Internal Committee?") == []


def test_a_follow_up_does_not_let_the_previous_question_match_literally():
    """The rewrite exists so a referential follow-up can be embedded at all, but
    BM25 is literal: given both questions it would match the earlier one's
    keywords just as readily as the current one's."""
    date = a_stored_date()
    history = [{"question": f"What is due on {date}?"}]
    rewritten = rag.search_text_for("What about the owner?", history)

    # The rewrite carries the old date, so the keyword leg must not see it.
    assert date in rewritten
    assert vector_store.lexical_candidates("What about the owner?") == []
    hits = vector_store.search(rewritten, lexical_question="What about the owner?")
    assert all(hit["match"] == "semantic" for hit in hits)


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


# --- Summarising a whole document --------------------------------------------
# "Summarise this document" names no subject, so there is nothing to retrieve
# against; it reads the document instead of searching it.

@pytest.mark.parametrize(
    "question",
    ["summarise this document", "Summarize this document", "give me a summary",
     "what is this document about?", "key points please"],
)
def test_recognises_a_summary_request(question):
    assert rag.is_summary_request(question)


@pytest.mark.parametrize(
    "question",
    ["How many days of annual leave do I get?",
     "Summarise the notice period rules, the payroll cut-off dates and who signs them off"],
)
def test_a_question_about_a_topic_is_not_a_summary_request(question):
    assert not rag.is_summary_request(question)


def test_a_summary_needs_a_document_to_be_chosen():
    result = rag.answer_question("summarise this document")
    assert result["status"] == "pick_document"
    assert result["citations"] == []


def test_an_overview_spans_the_chosen_document():
    """The excerpts a summary is written from must cover the whole file, and
    nothing from any other file."""
    source = any_pdf().name
    chunks = vector_store.document_overview(source)
    if not chunks:
        pytest.skip(f"{source} is not indexed")

    assert {chunk["source"] for chunk in chunks} == {source}
    pages = [chunk["page"] for chunk in chunks]
    assert pages == sorted(pages)


def test_a_summary_request_retrieves_the_document_not_a_search():
    source = any_pdf().name
    if not vector_store.document_overview(source):
        pytest.skip(f"{source} is not indexed")

    found = rag.retrieve("summarise this document", source=source)
    assert found["chunks"], "a summary must not come back empty-handed"
    assert all(chunk["match"] == "overview" for chunk in found["chunks"])


# --- Names, however the question capitalises them ----------------------------
# A question typed into a chat box is usually all lowercase, and a person's name
# is the one thing the embedding cannot pin down.

def a_stored_name():
    """A name the corpus itself writes capitalised, and holds in few chunks."""
    index = lexical.get_index(vector_store._corpus()[1])
    for phrase in sorted(index.name_pairs):
        if 0 < index.phrase_count(phrase) <= index.rare_ceiling:
            return phrase
    pytest.skip("no rare proper name is indexed")


def test_a_name_anchors_the_keyword_leg_in_any_case():
    name = a_stored_name()
    index = lexical.get_index(vector_store._corpus()[1])

    assert index.value_terms(f"who is {name}") == [name]
    assert index.value_terms(f"who is {name.title()}?") == [name]


def test_a_lowercase_name_reaches_the_chunk_that_holds_it():
    name = a_stored_name()
    hits = vector_store.search(f"who is {name}")

    assert hits, "expected a match - did you run `python ingest.py`?"
    assert any(name in hit["text"].lower() for hit in hits)
    assert any(hit["match"] in ("keyword", "both") for hit in hits)


def test_an_ordinary_pair_of_words_is_not_a_name():
    """Only pairs the documents write as a proper name qualify, so an everyday
    phrase cannot admit chunks this way."""
    index = lexical.get_index(vector_store._corpus()[1])
    for question in ["how do I train a puppy?", "what are the travel entitlements?"]:
        assert index.named_phrases(question) == []


def test_a_name_split_across_two_lines_of_a_table_is_still_one_name():
    """PDF tables arrive with "<br>" where the cell wrapped, which used to cut a
    name in half and leave nothing for the question to match."""
    index = lexical.LexicalIndex(["|Prepared by|Arshi<br>Dutta/Anisha Singh|"])
    assert index.phrase_count("arshi dutta") == 1
    assert index.value_terms("who is arshi dutta") == ["arshi dutta"]
    assert index.candidates("who is arshi dutta", 4) == [0]
