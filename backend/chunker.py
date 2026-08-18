"""Step 1 of the pipeline: turn PDFs into clean, citable text chunks.

Each chunk carries its file, page and section heading, so every answer can show
a real citation. Pages are read as Markdown because it states headings outright
("## Heading", or a fully bold line) instead of leaving them to be guessed at.
"""

import re
from pathlib import Path
from typing import Dict, List

import pymupdf
import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE
MIN_CHUNK_LENGTH = 50
BOLD_LINE = re.compile(r"(?:\*\*[^*]+\*\*\s*)+")

def looks_like_heading(line: str) -> bool:
    """Is this line a section title? Markdown says so directly."""
    line = line.strip()
    return line.startswith("#") or bool(BOLD_LINE.fullmatch(line))

def heading_text(line: str) -> str:
    """A heading without its Markdown markers, for a readable citation."""
    return re.sub(r"[*#]", "", line).strip()

MAX_TABLE_PATHS = 500
EDGE_LINES = 3
TABLE_CHUNK_SIZE = 300
CONTENTS_LINE = re.compile(r"\.{6,}\s*\d+")

def table_columns(page) -> List[str]:
    """The column names of the widest table on a page, or [] if it has none.

    Widest, not first: a bordered title panel at the top of a page reads as a
    one-column table, and taking it would name the real table after that box.
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

TABLE_SEPARATOR = re.compile(r"\|(?:\s*-+\s*\|)+")

def restore_table_header(text: str, columns: List[str]) -> str:
    """Give a continuing table back the header it left on an earlier page.

    A table crossing a page break has no header there, so the converter promotes
    its first data row. That row is put back and the real header goes above it.
    """
    if not columns:
        return text

    header = "|" + "|".join(columns) + "|"
    restored: List[str] = []

    for line in text.split("\n"):
        promoted = restored[-1].strip() if restored else ""
        if TABLE_SEPARATOR.fullmatch(line.strip()) and promoted.count("|") - 1 == len(columns):
            restored.pop()
            restored += [header, line, promoted]
        else:
            restored.append(line)

    return "\n".join(restored)


def is_table_row(line: str) -> bool:
    """A Markdown table row, but not the |---|---| rule under its header."""
    line = line.strip()
    return line.startswith("|") and not TABLE_SEPARATOR.fullmatch(line)

def drop_cell_spill(page_text: str) -> str:
    """Remove the stray cell text a table page leaves outside its own rows.

    A column too wide for the page - here a Notes column reading "Documentation
    should updated after completio" - is emitted as loose fragments above the
    table instead of inside it. They are cut off mid-word, repeat on every page,
    and answer nothing, but they are long enough to survive MIN_CHUNK_LENGTH
    once several are joined into one chunk.

    Only short, heading-less, row-less blocks go. The register's own title and
    its one-line description are longer than a fragment and stay.
    """
    kept = []
    for block in page_text.split("\n\n"):
        lines = [line for line in block.split("\n") if line.strip()]
        if not lines:
            continue
        if any(is_table_row(line) or looks_like_heading(line) for line in lines):
            kept.append(block)
        elif len(block.strip()) >= MIN_CHUNK_LENGTH:
            kept.append(block)
    return "\n\n".join(kept)

def row_fields(piece: str, columns: List[str]) -> Dict[str, str]:
    """Pull a single table row apart into {column name: cell value}.

    An embedding averages a row into 384 numbers about its *meaning*, and
    "04-02-2026" against "14-10-2026" is not a difference in meaning - the five
    laptop-replacement rows sit within 0.013 of each other. Keeping the cells as
    their own fields is what lets an exact value be matched exactly.
    """
    if not columns:
        return {}

    cells_of = lambda line: [c.strip() for c in line.strip().strip("|").split("|")]
    rows = [
        line
        for line in piece.split("\n")
        if is_table_row(line) and cells_of(line) != columns
    ]
    if len(rows) != 1:
        return {}

    cells = cells_of(rows[0])
    if len(cells) != len(columns):
        return {}

    fields = {}
    for name, value in zip(columns, cells):
        key = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
        value = " ".join(value.replace("<br>", " ").split())
        if key and value:
            fields[key] = value
    return fields

def _shape(line: str) -> str:
    """A line with its digits blanked, so "page 3/14" and "page 8/14" match."""
    return re.sub(r"\d+", "#", line.strip())

def find_page_furniture(pages: List[str], min_share: float = 0.5) -> set:
    """Find the running headers and footers to strip before chunking.

    A repeated title line or URL is short and keyword-dense, so it embeds as a
    near-perfect match for any question and crowds out the real content. A line
    is furniture if it appears on at least half the pages, digits ignored.
    """
    seen_on: Dict[str, int] = {}
    for text in pages:
        lines = [line for line in text.split("\n") if line.strip()]
        for shape in {_shape(line) for line in lines[:EDGE_LINES] + lines[-EDGE_LINES:]}:
            seen_on[shape] = seen_on.get(shape, 0) + 1

    threshold = max(3, int(len(pages) * min_share))
    return {shape for shape, count in seen_on.items() if count >= threshold}

def clean_text(raw: str) -> str:
    """Rebuild readable paragraphs from one page of Markdown.

    The converter blank-line-separates every *visual* line and double-separates
    real paragraphs, so three-or-more newlines starts a paragraph and a single
    blank line is a wrapped sentence to join back up. Headings stay alone.
    """
    text = re.sub(r"\s*~~-~~\s*\n\s*\n\s*", "", raw)
    text = text.replace("~~", "")
    blocks: List[Dict] = []
    for paragraph in re.split(r"\n{3,}", text):
        blocks.append({"heading": "", "lines": []})
        for line in paragraph.split("\n"):
            line = re.sub(r"[ \t]+", " ", line).strip()

            if not line:
                continue
            if looks_like_heading(line) or line.startswith("|"):
                blocks.append({"heading": line, "lines": []})
            else:
                blocks[-1]["lines"].append(line)
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

    A chunk can span several short sections, so every heading inside it is
    listed rather than only the first. `carried` is the last heading from an
    earlier page, for sections and tables that run past a page boundary.
    """
    headings = [
        heading_text(line) for line in chunk.split("\n") if looks_like_heading(line)
    ]
    if headings:
        if len(headings) > 3:
            headings = [headings[0], "...", headings[-1]]
        return " > ".join(headings)
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
        separators=[r"\n(?=#|\*\*)", r"\n\n", r"\n", r"\. ", " ", ""],
    )
    row_splitter = RecursiveCharacterTextSplitter(
        chunk_size=TABLE_CHUNK_SIZE,
        chunk_overlap=0,
        is_separator_regex=True,
        separators=[r"\n(?=\|)", r"\n\n", r"\n", r"\. ", " ", ""],
    )
    raw_pages = [
        page["text"]
        for page in pymupdf4llm.to_markdown(
            str(pdf_path), page_chunks=True, ignore_graphics=True
        )
    ]
    document = pymupdf.open(str(pdf_path))
    page_columns = [table_columns(page) for page in document]
    for number, names in enumerate(page_columns):
        if names:
            raw_pages[number] = pymupdf4llm.to_markdown(
                str(pdf_path), pages=[number], page_chunks=True
            )[0]["text"]
    furniture = find_page_furniture(raw_pages)
    chunks = []
    carried = ""
    columns: List[str] = []
    for page_number, raw in enumerate(raw_pages, start=1):
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
        if found:
            page_text = drop_cell_spill(page_text)
        if not page_text:
            continue  # Scanned image page with no text layer - nothing to index.

        chosen = row_splitter if found else splitter

        for piece in chosen.split_text(page_text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LENGTH:
                continue
            if len(CONTENTS_LINE.findall(piece)) > 1:
                continue

            section = find_section(page_text, piece, carried)
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
                    "fields": row_fields(piece, columns),
                }
            )
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
            print(f"  !  Skipped {pdf_path.name}: {error}")
            continue
        if not chunks:
            print(f"  !  {pdf_path.name} produced no text (is it a scan?)")
            continue
        print(f"  -  {pdf_path.name}: {len(chunks)} chunks")
        all_chunks.extend(chunks)
    return all_chunks