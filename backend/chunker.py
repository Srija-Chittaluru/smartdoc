"""Step 1 of the pipeline: turn PDFs into clean, citable text chunks.

Each chunk remembers which file and which page it came from, plus the section
heading it sits under. That metadata is what lets every answer show a real
citation instead of just "somewhere in the docs".

Pages are read as Markdown rather than plain text. A PDF carries no structure,
so recognising a heading used to mean guessing from how a line read - short,
Title Case, no full stop. Markdown states it outright, either as "## Heading" or
as a line that is entirely bold, so the guessing is gone and a section can be
split at the boundary its author intended.
"""

import re
from pathlib import Path
from typing import Dict, List

import pymupdf
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE

# Chunks shorter than this are usually page numbers or stray headers.
MIN_CHUNK_LENGTH = 50

# A line that is nothing but bold text. PDFs whose sections are bold rather than
# larger arrive this way instead of as "##".
#
# The repeat matters: a numbered heading is emitted as two separate bold runs,
# "**2** **Travel expenses**", because the number and the title are different
# spans in the PDF. Matching only a single run silently missed every numbered
# section in such a document and left its chunks with no section at all.
BOLD_LINE = re.compile(r"(?:\*\*[^*]+\*\*\s*)+")


def looks_like_heading(line: str) -> bool:
    """Is this line a section title? Markdown says so directly."""
    line = line.strip()
    return line.startswith("#") or bool(BOLD_LINE.fullmatch(line))


def heading_text(line: str) -> str:
    """A heading without its Markdown markers, for a readable citation."""
    return re.sub(r"[*#]", "", line).strip()


# Finding a table means analysing every vector path on the page, which costs
# about two seconds on a page printed from a browser - and finds nothing, since
# those pages draw their shading with thousands of tiny paths rather than ruled
# cell borders. A real ruled table is drawn with tens of lines: the sample
# documents here use 69 and 95 per page, against 2652 for the browser print.
# Counting the paths first is cheap; the detection it skips is not.
MAX_TABLE_PATHS = 500

# How many lines at each end of a page can be a running header or footer.
EDGE_LINES = 3

# Table pages are split far more finely than prose. A chunk holding four rows
# describes four unrelated records, and its embedding is the average of all
# four - close to none of them. Asking about one row then loses to any document
# that discusses the subject in prose. One record per chunk keeps the embedding
# about one thing. Overlap is left off: a row does not continue into the next.
TABLE_CHUNK_SIZE = 300

# A contents line - "5 Authorisation of Expense Claims ......... 6". It is
# navigation, not an answer, and it competes with the section it points at.
CONTENTS_LINE = re.compile(r"\.{6,}\s*\d+")


def table_columns(page) -> List[str]:
    """The column names of the widest table on a page, or [] if it has none.

    Widest, not first: a page often opens with a bordered text box - a title
    panel, a running header - which is detected as a one-column table. Taking
    the first would carry that box's text forward as if it were the column
    names of the real table further down the page. Anything narrower than two
    columns is not a table at all.
    """
    try:
        if len(page.get_drawings()) > MAX_TABLE_PATHS:
            return []
        tables = page.find_tables().tables
    except Exception:
        return []

    widest: List[str] = []
    for table in tables:
        names = [" ".join(n.split()) for n in table.header.names if n and n.strip()]
        if len(names) > max(len(widest), 1):
            widest = names
    return widest


# The "|---|---|" line that separates a Markdown table's header from its body.
TABLE_SEPARATOR = re.compile(r"\|(?:\s*-+\s*\|)+")


def restore_table_header(text: str, columns: List[str]) -> str:
    """Give a continuing table back the header it left on an earlier page.

    A table carrying on past a page break has no header row of its own, so the
    converter promotes that page's first *data* row into one. Two things go
    wrong: the columns end up named after one arbitrary row, and that row stops
    reading as data. Here the promoted row is put back where it belongs and the
    real header - captured on the page the table started on - goes above it.
    """
    if not columns:
        return text

    header = "|" + "|".join(columns) + "|"
    restored: List[str] = []

    for line in text.split("\n"):
        # Only step in when the promoted row is the right shape for this header.
        # A page can hold more than one table, and pushing a seven-column header
        # onto a three-column one would describe it wrongly.
        promoted = restored[-1].strip() if restored else ""
        if TABLE_SEPARATOR.fullmatch(line.strip()) and promoted.count("|") - 1 == len(columns):
            restored.pop()
            restored += [header, line, promoted]
        else:
            restored.append(line)

    return "\n".join(restored)


def _shape(line: str) -> str:
    """A line with its digits blanked, so "page 3/14" and "page 8/14" match."""
    return re.sub(r"\d+", "#", line.strip())


def find_page_furniture(pages: List[str], min_share: float = 0.5) -> set:
    """Find the running headers and footers to strip before chunking.

    A PDF printed from a web page repeats a title line and a URL on every page.
    Those lines are short and full of the document's own keywords, so they embed
    as a near-perfect match for any question about the document - and being
    their own chunks, they crowd the real content out of the top results.

    A line is furniture if it appears on at least half the pages. Digits are
    ignored when comparing, because the page number is the one part that
    changes. The three-page floor means a short document, where a heading could
    legitimately appear on every page, is left alone.
    """
    seen_on: Dict[str, int] = {}
    for text in pages:
        lines = [line for line in text.split("\n") if line.strip()]
        # Only the top and bottom of a page can hold a running header or footer.
        # Restricting the search there is what protects repetitive *content*: a
        # register listing "Facilities Team High Open" on every page repeats
        # exactly like a footer does, but it sits in the middle of the page, and
        # deleting it would take most of the document with it.
        for shape in {_shape(line) for line in lines[:EDGE_LINES] + lines[-EDGE_LINES:]}:
            seen_on[shape] = seen_on.get(shape, 0) + 1

    threshold = max(3, int(len(pages) * min_share))
    return {shape for shape, count in seen_on.items() if count >= threshold}


def clean_text(raw: str) -> str:
    """Rebuild readable paragraphs from one page of Markdown.

    The converter emits one line per *visual* line, separated by a blank line,
    and separates real paragraphs by two. So three-or-more newlines is where a
    paragraph actually starts; a single blank line is just a wrapped sentence,
    and we join it back up.

    Headings stay on their own line so each one remains attached to the section
    it introduces.
    """
    # A word hyphenated across a line break arrives as "anti ~~-~~\n\ndiscrimination",
    # because the converter marks the soft hyphen as struck through. Closing the
    # break rebuilds the word.
    text = re.sub(r"\s*~~-~~\s*\n\s*\n\s*", "", raw)
    # Any other struck-through run is real text; keep it, drop the markers.
    text = text.replace("~~", "")

    # A block is one section: an optional heading plus the lines under it.
    blocks: List[Dict] = []

    for paragraph in re.split(r"\n{3,}", text):
        blocks.append({"heading": "", "lines": []})

        for line in paragraph.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()

            if not line:
                continue
            if looks_like_heading(line) or line.startswith("|"):
                # A heading, or a table row. Both go on a line of their own -
                # rows especially, because joining them the way wrapped prose is
                # joined would run every row of a table into one another and
                # undo the reason for reading the table properly at all.
                blocks.append({"heading": line, "lines": []})
            else:
                # A wrapped line: join it onto the paragraph being built.
                blocks[-1]["lines"].append(line)

    # Inside a block: heading on line 1, its paragraph joined onto line 2, so
    # the two stay together in the same chunk.
    # Between blocks: a blank line, which is the splitter's preferred break.
    rendered = []
    for block in blocks:
        parts = [block["heading"]] if block["heading"] else []
        if block["lines"]:
            parts.append(" ".join(block["lines"]))
        if parts:
            rendered.append("\n".join(parts))

    return "\n\n".join(rendered).strip()


def find_section(page_text: str, chunk: str, carried: str = "") -> str:
    """Describe which section(s) of the page this chunk covers.

    A chunk can span more than one short section, so we list every heading
    inside it rather than only the first. Citing "4. Air Travel > 5.
    Accommodation" is honest; citing only "4. Air Travel" for an answer about
    hotels would send the reader to the wrong paragraph.

    `carried` is the last heading seen on an earlier page. A section - or a
    table - that runs over a page boundary leaves the pages after the first with
    no heading of their own, and those chunks used to be stored with no section
    at all. Falling back to the heading the section actually started under is
    what keeps them attached to it.
    """
    headings = [
        heading_text(line) for line in chunk.split("\n") if looks_like_heading(line)
    ]
    if headings:
        # A chunk covering many short sections would produce an unreadable
        # chain, so show the range rather than every heading in it.
        if len(headings) > 3:
            headings = [headings[0], "...", headings[-1]]
        return " > ".join(headings)

    # No heading inside the chunk: it is the tail of a long section that began
    # earlier on the page, so walk back and take the last heading above it.
    position = page_text.find(chunk[:80])
    if position == -1:
        return carried

    heading = carried
    for line in page_text[:position].split("\n"):
        if looks_like_heading(line):
            heading = heading_text(line)
    return heading


def chunk_pdf(pdf_path: Path) -> List[Dict]:
    """Read one PDF and return its chunks, each with citation metadata."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        is_separator_regex=True,
        # Break at the most natural boundary available, in this order: at a
        # heading, then between paragraphs, then between lines, then sentence,
        # then word, then raw character.
        #
        # The heading rule is what Markdown buys us. Without it the splitter
        # packs to 800 characters and a chunk can open with the tail of the
        # previous section - which is how the accessibility definition ended up
        # behind 200 characters about reporting family relationships, and stopped
        # being retrievable at all.
        separators=[r"\n(?=#|\*\*)", r"\n\n", r"\n", r"\. ", " ", ""],
    )
    # The same splitter, sized for one table record rather than a paragraph.
    row_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=0,
        is_separator_regex=True,
        separators=[r"\n(?=\|)", r"\n\n", r"\n", r"\. ", " ", ""],
    )

    # page_chunks keeps one entry per page. The plain string form would be
    # easier, but it flattens the document and the page number - which every
    # citation depends on - would be gone.
    #
    # ignore_graphics is not an optimisation, it is a correctness fix. A table
    # drawn with background shading rather than ruled lines is not recognised as
    # a table, and the converter then drops the text sitting on top of that
    # shading - silently. It cost us the whole Roles/Responsibility table. With
    # graphics ignored, such a table arrives as plain text rather than as a
    # Markdown table: less structured, but present, which is the trade we want.
    # It is also what makes ingest fast, because a page printed from a browser
    # can carry tens of thousands of vector paths that no longer need analysing.
    raw_pages = [
        page["text"]
        for page in pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True, ignore_graphics=True
        )
    ]

    # Pages holding a ruled table are read a second time with graphics enabled.
    # That is the only way to get real Markdown rows, and rows are what keep a
    # value attached to the record it describes - without them "High" is just a
    # word floating near several different rows, and the answer to "what is the
    # priority of X" becomes whichever priority happens to sit nearest.
    # Only pages that actually have a table pay for the second read.
    document = pymupdf.open(str(pdf_path))
    page_columns = [table_columns(page) for page in document]
    for number, names in enumerate(page_columns):
        if names:
            raw_pages[number] = pymupdf4llm.to_markdown(
                str(pdf_path), pages=[number], page_chunks=True
            )[0]["text"]
    # The repeated header and footer can only be recognised by comparing pages
    # against each other, so this needs all of them in hand.
    furniture = find_page_furniture(raw_pages)

    chunks = []
    # The heading a section started under, carried forward so the pages it
    # continues onto are not left unlabelled. See find_section.
    carried = ""
    # Likewise the column names of a table, which appear only on the page the
    # table began on.
    columns: List[str] = []

    # We split page by page rather than treating the PDF as one long string.
    # It costs us overlap across a page boundary, but it buys an exact page
    # number for every chunk - which is what makes a citation verifiable.
    for page_number, raw in enumerate(raw_pages, start=1):
        # A table running past a page break has no header row on the pages it
        # continues onto, so the detector reads that page's first *data* row as
        # the header. The same number of columns means it is still the same
        # table, so the header captured where the table started is the real one.
        found = page_columns[page_number - 1]
        continuing = bool(found) and found != columns and len(found) == len(columns)
        if found and not continuing:
            columns = found

        body = "\n".join(
            line for line in raw.split("\n") if _shape(line) not in furniture
        )
        if continuing:
            body = restore_table_header(body, columns)

        page_text = clean_text(body)
        if not page_text:
            continue  # Scanned image page with no text layer - nothing to index.

        chosen = row_splitter if found else splitter

        for piece in chosen.split_text(page_text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LENGTH:
                continue
            # A contents page lists where things are, never what they say.
            if len(CONTENTS_LINE.findall(piece)) > 1:
                continue

            section = find_section(page_text, piece, carried)

            # A chunk holding no heading of its own - the middle of a long
            # section, or a table carrying on past a page break - says nothing
            # about its own subject once it is embedded alone. "IT Support,
            # Medium, Pending Approval, 24-08-2026" cannot match a question
            # about priority or target dates, because neither word is in it.
            # Naming the section, and the columns the values sit under, is what
            # makes such a chunk findable and readable.
            # A chunk of table rows gets the column names and nothing else. It
            # is short, so anything added to it takes up a large share of what
            # is embedded - and the same section name repeated across a hundred
            # row chunks makes them all look alike, which buries the very row
            # being asked for. Measured on one register row: the row alone sits
            # at 0.437 from its question, with the column names 0.543, and with
            # the section name as well 0.687 - thirteenth place instead of first.
            if piece.lstrip().startswith("|") or "\n|" in piece:
                prefix = [" | ".join(columns)] if columns else []
            elif section and not any(map(looks_like_heading, piece.split("\n"))):
                prefix = [section]
            else:
                prefix = []

            if prefix:
                piece = "\n".join(prefix) + "\n\n" + piece

            chunks.append(
                {
                    "text": piece,
                    "source": pdf_path.name,
                    "page": page_number,
                    "section": section,
                }
            )

        # Whatever this page ended under is what the next page continues under.
        for line in page_text.split("\n"):
            if looks_like_heading(line):
                carried = heading_text(line)

    return chunks


def chunk_all_pdfs(documents_dir: Path) -> List[Dict]:
    """Chunk every PDF in a folder. Skips files that fail to parse."""
    pdf_paths = sorted(documents_dir.glob("*.pdf"))
    if not pdf_paths:
        raise FileNotFoundError(f"No PDFs found in {documents_dir}")

    all_chunks = []
    for pdf_path in pdf_paths:
        try:
            chunks = chunk_pdf(pdf_path)
        except Exception as error:
            # One corrupt or password-protected PDF should not stop the ingest.
            print(f"  !  Skipped {pdf_path.name}: {error}")
            continue

        if not chunks:
            print(f"  !  {pdf_path.name} produced no text (is it a scan?)")
            continue

        print(f"  -  {pdf_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)

    return all_chunks
