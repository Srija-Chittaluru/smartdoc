"""Step 1 of the pipeline: turn PDFs into clean, citable text chunks.

Each chunk remembers which file and which page it came from, plus the section
heading it sits under. That metadata is what lets every answer show a real
citation instead of just "somewhere in the docs".
"""

import re
from pathlib import Path
from typing import Dict, List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from backend.config import CHUNK_OVERLAP, CHUNK_SIZE

# Chunks shorter than this are usually page numbers or stray headers.
MIN_CHUNK_LENGTH = 50

# Words we do not expect to be capitalised in a Title Case heading.
SMALL_WORDS = {
    "a", "an", "and", "the", "of", "or", "to", "for", "in", "on", "with",
    "at", "by", "from", "per", "as", "is", "not", "vs",
}


def looks_like_heading(line: str) -> bool:
    """Decide whether a single line of extracted text is a section title.

    PDFs carry no structure, so a heading has to be recognised by how it reads.
    Headings in policy documents are short, unpunctuated at the end, and are
    either numbered ("2. Annual Leave"), ALL CAPS, or Title Case.

    The test is deliberately strict. A false positive would split a sentence out
    of its paragraph, which is worse than missing a heading - if we miss one,
    the citation still carries the correct document and page number.
    """
    line = line.strip()
    words = line.split()

    if not (3 < len(line) < 70) or not (1 <= len(words) <= 10):
        return False

    # Prose ends in punctuation; titles do not. This one test is what keeps
    # fragments like "30 minutes of inactivity." from being read as headings.
    if line.endswith((".", ",", ";", ":")):
        return False

    # "2. Annual Leave" or "3.1 Scope" - a number, then a capitalised word.
    numbered = bool(re.match(r"^\d+(\.\d+)*[\.\)]?\s+[A-Z]", line))

    if numbered or line.isupper():
        return True

    # Title Case: every significant word starts with a capital letter.
    alpha_words = [w for w in words if w[0].isalpha()]
    if len(alpha_words) < 2:
        return False
    return all(w[0].isupper() or w.lower() in SMALL_WORDS for w in alpha_words)


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
    """Rebuild readable paragraphs from raw PDF text.

    PDF extraction returns one line per *visual* line, so a single sentence
    arrives broken across three lines. We join those wrapped lines back into
    paragraphs, while keeping each heading on its own line so that the heading
    stays attached to the section it introduces.
    """
    # "compen-\nsation" -> "compensation" (words hyphenated across a line break)
    text = re.sub(r"-\n(\w)", r"\1", raw)

    # A block is one section: an optional heading plus the lines under it.
    blocks: List[Dict] = []

    for line in text.split("\n"):
        line = re.sub(r"[ \t]+", " ", line).strip()

        if not line:
            # A blank line ends the current paragraph.
            blocks.append({"heading": "", "lines": []})
        elif looks_like_heading(line):
            # Note: a title long enough to wrap onto two visual lines becomes
            # two headings here. We accept that rather than merging consecutive
            # headings, because merging also glues a document title onto the
            # first real section, which is worse. Every heading reported this
            # way is a literal line from the page.
            blocks.append({"heading": line, "lines": []})
        else:
            if not blocks:
                blocks.append({"heading": "", "lines": []})
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
        line.strip() for line in chunk.split("\n") if looks_like_heading(line.strip())
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
            heading = line.strip()
    return heading


def chunk_pdf(pdf_path: Path) -> List[Dict]:
    """Read one PDF and return its chunks, each with citation metadata."""
    reader = PdfReader(str(pdf_path))

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        # Break at the most natural boundary available, in this order:
        # between sections, then between heading and body, then sentence,
        # then word, then raw character.
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    # Read every page up front: the repeated header and footer can only be
    # recognised by comparing pages against each other.
    raw_pages = [page.extract_text() or "" for page in reader.pages]
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
