# 📄 SmartDoc — Ask your company documents anything

Employees waste time hunting for answers buried in PDFs. SmartDoc lets them ask
a question in plain English and get a short answer **with a citation showing
exactly which document, page and section it came from** — so the answer can be
verified, not just trusted.

If the answer isn't in the documents, SmartDoc says *"I don't know"* rather than
making something up.

---

## Quick start

```bash
# 1. Set up
python3 -m venv .venv
.venv/bin/pip install -r requirements.txt

# 2. Add your OpenAI key
cp .env.example .env
#    then edit .env and paste your key

# 3. Create seven sample PDFs (skip if you're using your own)
.venv/bin/python make_sample_pdfs.py

# 4. Build the search index
.venv/bin/python ingest.py

# 5. Run it
./run.sh
```

Then open **http://localhost:8501**.

To use your own documents: drop PDFs into `documents/` and re-run
`.venv/bin/python ingest.py`. Nothing else needs to change.

The seven sample PDFs are invented policies for a fictional company. Six use
numbered headings (`2. Annual Leave`); `vendor_procurement_sop.pdf` deliberately
uses unnumbered ALL-CAPS headings instead, so the corpus proves the chunker
isn't tuned to a single document format.

---

## How it works

```
PDFs ──▶ chunk ──▶ embed ──▶ ChromaDB          (once, via ingest.py)
                                  │
Question ──▶ embed ──▶ search ────┘──▶ top 4 chunks ──▶ GPT ──▶ answer + citations
```

**What happens when you press Enter**, step by step:

1. **Guard** — empty question, or one over 1,000 characters, is rejected before
   anything costs money.
2. **Embed the question** — the same local model that embedded the documents
   turns your question into 384 numbers representing its meaning.
3. **Search** — ChromaDB finds the 4 closest chunks by cosine distance.
4. **Filter** — chunks further than `MAX_DISTANCE` away are dropped. If nothing
   survives, we stop here and answer *"I don't know"* — the language model is
   never called, so it never gets the chance to improvise.
5. **Prompt** — the surviving chunks are pasted into the prompt as numbered
   sources, with a system prompt that forbids outside knowledge.
6. **Answer** — GPT writes 2–4 sentences with `[1]`, `[2]` markers, and the UI
   shows the matching document, page and section under each answer.

### Retrieval-Augmented Generation, in plain English

A language model only knows what it saw during training — it has never read your
handbook, and if you ask anyway it will invent a plausible answer. RAG fixes
this by **looking things up first**: we search your documents, hand the model the
relevant paragraphs, and ask it to answer using only those. It turns a
closed-book exam into an open-book one, which is why the answers can be cited.

---

## Design decisions

### Chunk size: 800 characters, 150 overlap

A chunk is the unit of retrieval, so its size decides what we can find.

- **Too small (~300 chars)** — meaning gets cut in half. While testing, *"employees
  are entitled to 21 days of leave"* landed in a different chunk from
  *"...may take accrued leave during their probation period"*, and answers came
  back confidently missing the condition.
- **Too large (~2,000 chars)** — one chunk covers five unrelated topics, so its
  embedding becomes a blurry average of all of them and matches nothing sharply.
- **800 characters (~150 words)** is roughly one policy section: specific enough
  to embed cleanly, complete enough to keep a rule and its exception together.

The **150-character overlap (~19%)** exists because a sentence sitting on a chunk
boundary would otherwise be split down the middle and belong to neither chunk
properly. The overlap guarantees it appears intact in at least one. 15–20% is the
usual sweet spot — more than that just fills the database with duplicated text.

Chunking is done **page by page**. This gives up overlap across a page boundary,
but it buys an exact page number for every single chunk — and a citation without
a page number isn't really verifiable.

### Why ChromaDB and not keyword search?

Keyword search matches **letters**. The handbook says *"annual leave"*; employees
ask about *"vacation days"*. Those share no keywords, so keyword search returns
nothing — while the embeddings put them right next to each other because they
mean the same thing. There's a test for exactly this
(`test_matches_meaning_not_just_keywords`).

ChromaDB specifically, over a plain Python list of vectors:

- It **persists to disk** (`./chroma_db`), so ingesting once survives a restart.
- It searches with an **HNSW index** instead of comparing the query against
  every chunk one at a time.
- It stores **metadata alongside each vector**, which is what makes citations
  possible at all.

### Why local embeddings but a hosted LLM?

Embeddings run on `all-MiniLM-L6-v2` via sentence-transformers: no API key, no
per-document cost, works offline, and re-indexing is free. Writing a fluent
answer is the harder job, so that goes to GPT.

### Cosine distance, not the default

Chroma defaults to squared L2 distance. We set `hnsw:space: cosine` because
cosine compares *direction* rather than magnitude, which is the right measure
for sentence embeddings.

---

## How it avoids hallucinating

Four independent layers, because a prompt alone is not enough:

1. **Distance threshold** — irrelevant chunks are dropped before the model sees
   them. `MAX_DISTANCE = 0.75` was chosen by measuring, not guessing. Across 10
   questions the documents answer and 8 they don't, cosine distance separated
   cleanly:

   | | range | closest case |
   |---|---|---|
   | In-scope | 0.276 – 0.615 | the vague query `"vacation days"` |
   | Out-of-scope | 0.781 – 0.977 | `"How do I train a puppy?"` — it brushes against *Mandatory Training* |

   0.75 sits in that gap, with 0.135 of headroom for an in-scope question
   phrased worse than any tested. An earlier setting of 0.85 let the puppy
   question through; a test now pins that case.
2. **Empty retrieval short-circuits** — if nothing survives the filter, the API
   is never called and the answer is a fixed *"I don't know"* string.
3. **A strict system prompt** — outside knowledge forbidden, one exact refusal
   sentence specified.
4. **A post-check** — if the model refuses anyway, we strip the citations, so an
   *"I don't know"* is never dressed up with sources.

## Where it is most likely to be wrong

Being honest about the limits:

- **Scanned PDFs.** `pypdf` reads a text layer. A photographed or scanned page
  has none, so it is silently skipped — ingest prints a warning. Fixing this
  needs OCR (e.g. Tesseract), which isn't wired up.
- **Answers that need two distant documents.** We retrieve 4 chunks by
  similarity. A question needing a rule from page 1 of one PDF *and* page 9 of
  another may only get one of them.
- **Tables and multi-column layouts.** Text extraction flattens a table into a
  run of words, so column relationships can be lost.
- **The section label is a heuristic.** `looks_like_heading()` recognises
  numbered, ALL CAPS and Title Case lines. An unusual heading style may be
  missed — the document and page number are still correct.
- **The threshold is tuned on these documents.** A very different corpus
  (legal contracts, dense technical specs) would want it re-measured.
- **Near-duplicate documents.** Two versions of the same policy will both be
  retrieved, and the model will cite both rather than knowing which is current.

---

## Testing

```bash
.venv/bin/python -m pytest tests/ -v
```

25 tests covering heading detection, text cleaning, chunk metadata and size,
semantic retrieval, out-of-scope refusal, and the input guards. They need no API
key — generation is deliberately not tested, since it would cost money and
depend on the network.

### Edge cases handled

| Input | Behaviour |
|---|---|
| Empty / whitespace question | "Please type a question first." No API call. |
| Question over 1,000 chars | Rejected with the actual length. No API call. |
| Question in another language | Answered in that language (rule 6 of the prompt) |
| Out-of-scope question | "I don't know based on the provided documents." |
| No documents indexed yet | Tells you to run `ingest.py` |
| Missing / invalid API key | Names the problem and points at `.env` — no crash |
| No internet | "Could not reach OpenAI. Check your internet connection." |
| Rate limited / out of credit | Explains it and suggests retrying |
| Backend not running | Sidebar shows the command to start it |
| Corrupt or password-locked PDF | Skipped with a warning; ingest continues |
| Scanned PDF with no text | Warned about by name, not silently ignored |

Answers use `temperature=0`, so the same question returns an equivalent answer
each time.

---

## Security

- **No secrets in code.** The key is read from `.env`, which is in `.gitignore`.
  `.env.example` is committed as a template and contains no real key.
- **Input is validated** — type-checked by Pydantic, then length-capped before
  reaching the API.
- **No SQL and no user-controlled file paths**, so there is no injection surface.
- **Errors never leak internals** to the UI; the API returns HTTP 200 with a
  `status` field instead of a stack trace.

---

## Project layout

```
smartdoc/
├── documents/                  ← put your PDFs here
├── chroma_db/                  ← generated vector index (git-ignored)
├── backend/
│   ├── config.py               all settings and tunables, one place
│   ├── chunker.py              PDF → clean, citable chunks
│   ├── vector_store.py         embeddings + ChromaDB
│   ├── rag.py                  query → retrieve → answer
│   └── main.py                 FastAPI endpoints
├── tests/test_pipeline.py      21 tests, no API key needed
├── app.py                      Streamlit chat UI
├── ingest.py                   build the index
├── make_sample_pdfs.py         generate the seven sample PDFs
├── run.sh                      start backend + UI together
├── requirements.txt
├── .env.example                template — safe to commit
└── .gitignore                  excludes .env, .venv, chroma_db
```

## API

| Endpoint | Purpose |
|---|---|
| `GET /health` | Is the API up, and how many chunks are indexed |
| `GET /config` | The active chunk size, model and vector DB settings |
| `POST /ask` | `{"question": "..."}` → answer + citations + status |

Interactive docs at **http://127.0.0.1:8000/docs** while the backend is running.

`status` is one of: `ok`, `empty_question`, `question_too_long`, `no_documents`,
`no_match`, `missing_api_key`, `api_error`.

## Stack

FastAPI · LangChain text splitters · ChromaDB · sentence-transformers · OpenAI · Streamlit
