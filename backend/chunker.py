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

import pymupdf4llm
from langchain_text_splitters import RecursiveCharacterTextSplitter

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE

# Chunks shorter than this are usually page numbers or stray headers.
MIN_CHUNK_LENGTH = 50

# A line that is nothing but bold text. PDFs whose sections are bold rather than
# larger arrive this way instead of as "##".
BOLD_LINE = re.compile(r"\*\*[^*]+\*\*")


def looks_like_heading(line: str) -> bool:
    """Is this line a section title? Markdown says so directly."""
    line = line.strip()
    return line.startswith("#") or bool(BOLD_LINE.fullmatch(line))


def heading_text(line: str) -> str:
    """A heading without its Markdown markers, for a readable citation."""
    return re.sub(r"[*#]", "", line).strip()


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
        for shape in {_shape(line) for line in text.split("\n") if line.strip()}:
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
            if looks_like_heading(line):
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


def find_section(page_text: str, chunk: str) -> str:
    """Describe which section(s) of the page this chunk covers.

    A chunk can span more than one short section, so we list every heading
    inside it rather than only the first. Citing "4. Air Travel > 5.
    Accommodation" is honest; citing only "4. Air Travel" for an answer about
    hotels would send the reader to the wrong paragraph.
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
        return ""

    heading = ""
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

    # page_chunks keeps one entry per page. The plain string form would be
    # easier, but it flattens the document and the page number - which every
    # citation depends on - would be gone.
    raw_pages = [
        page["text"] for page in pymupdf4llm.to_markdown(str(pdf_path), page_chunks=True)
    ]
    # The repeated header and footer can only be recognised by comparing pages
    # against each other, so this needs all of them in hand.
    furniture = find_page_furniture(raw_pages)

    chunks = []
    # We split page by page rather than treating the PDF as one long string.
    # It costs us overlap across a page boundary, but it buys an exact page
    # number for every chunk - which is what makes a citation verifiable.
    for page_number, raw in enumerate(raw_pages, start=1):
        body = "\n".join(
            line for line in raw.split("\n") if _shape(line) not in furniture
        )
        page_text = clean_text(body)
        if not page_text:
            continue  # Scanned image page with no text layer - nothing to index.

        for piece in splitter.split_text(page_text):
            piece = piece.strip()
            if len(piece) < MIN_CHUNK_LENGTH:
                continue
            chunks.append(
                {
                    "text": piece,
                    "source": pdf_path.name,
                    "page": page_number,
                    "section": find_section(page_text, piece),
                }
            )

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
