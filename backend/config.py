"""All tunable settings for SmartDoc live here, in one place.

Nothing secret is stored in this file. Secrets are read from the .env file,
which is git-ignored (see .gitignore).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# Load key=value pairs from .env into the environment.
load_dotenv()

# --- Folders -----------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Drop your PDFs in here, then run `python ingest.py`.
DOCUMENTS_DIR = PROJECT_ROOT / "documents"

# ChromaDB writes its files here, so the index survives a restart.
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "company_documents"


# --- Chunking ----------------------------------------------------------------
# Why 800 characters (~150-200 words)?
#   - Small enough that one chunk is about one idea, so the embedding is
#     specific rather than a blurry average of a whole page.
#   - Big enough to keep a full policy sentence and its condition together.
#     At 300 chars we kept splitting "employees get 18 days leave" away from
#     "...after completing probation", which produced confidently wrong answers.
#
# Why 150 characters of overlap (~19%)?
#   - A sentence that straddles a chunk boundary would otherwise be cut in half
#     and belong to neither chunk. The overlap means it appears whole in one of
#     them. 15-20% is the usual sweet spot: enough to protect boundaries,
#     not so much that the database fills up with duplicate text.

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150


# --- Retrieval ---------------------------------------------------------------

# How many chunks to send to the LLM as context.
# 4 keeps the prompt focused; more chunks start adding noise that the model
# then has to ignore.
TOP_K = 4

# Chroma returns a cosine distance: 0.0 = identical, 2.0 = opposite.
# Anything above this is treated as "not really about the question" and is
# dropped. This is what makes out-of-scope questions answer "I don't know"
# instead of retrieving the 4 least-irrelevant chunks and inventing something.
#
# 0.75 was picked by measuring, not guessing. Across 10 questions the documents
# do answer and 8 they do not, the distances separated cleanly:
#
#   in-scope:      0.276 - 0.615   (worst was the vague query "vacation days")
#   out-of-scope:  0.781 - 0.977   (closest was "How do I train a puppy?",
#                                   which brushes against "Mandatory Training")
#
# 0.75 sits in that gap. It leaves 0.135 of headroom for an in-scope question
# phrased worse than any we tested, while still rejecting all 8 off-topic ones.
# Re-measure this if you swap in a very different corpus - the gap moves.
MAX_DISTANCE = 0.75


# --- Models ------------------------------------------------------------------

# Embeddings run locally via sentence-transformers: no API key, no cost,
# works offline. 384 dimensions, fast on a laptop.
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")

# temperature=0 asks the model to be as deterministic as it can, so the same
# question gives an equivalent answer every time.
TEMPERATURE = 0

# Guard against someone pasting an entire novel into the question box.
MAX_QUESTION_LENGTH = 1000


# --- Backend address ---------------------------------------------------------

API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")
