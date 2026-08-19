"""All tunable settings for SmartDoc live here, in one place.

Nothing secret is stored in this file. Secrets are read from the .env file,
which is git-ignored (see .gitignore).
"""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DOCUMENTS_DIR = PROJECT_ROOT / "documents"

CHROMA_DIR = PROJECT_ROOT / "chroma_db"

COLLECTION_NAME = "company_documents"

CHUNK_SIZE = 800
CHUNK_OVERLAP = 150

TOP_K = 4
# How many excerpts a whole-document summary is written from.
SUMMARY_K = 14
CANDIDATE_K = 12
MAX_DISTANCE = 0.75
RELATIVE_MARGIN = 0.30
STRONG_MATCH = 0.60
BM25_K1 = 1.5
BM25_B = 0.75
LEXICAL_K = 8
RARE_DF_RATIO = 0.10
RRF_K = 60
W_SEMANTIC = 1.0
W_LEXICAL = 1.0

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
CHAT_MODEL = os.getenv("CHAT_MODEL", "gpt-4o-mini")
TEMPERATURE = 0
MAX_QUESTION_LENGTH = 1000
API_HOST = os.getenv("API_HOST", "127.0.0.1")
API_PORT = int(os.getenv("API_PORT", "8000"))
API_URL = os.getenv("API_URL", f"http://{API_HOST}:{API_PORT}")
