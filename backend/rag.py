"""Step 3 of the pipeline: query -> retrieve -> answer.

RAG in one sentence: instead of hoping the language model memorised your
handbook, we look up the relevant paragraphs first and hand them to the model
as an open book, then ask it to answer using only what is on the page.

The pipeline is unchanged from the original single-function version; it is only
split into named steps so each one can be reused:

    validate_question  ->  retrieve  ->  build_prompt  ->  generate

answer_question() runs all four and returns a complete answer. answer_stream()
runs the same four but yields the answer as it arrives from the model. Both go
through the identical guards, so neither can produce an answer the other would
have refused.
"""

import time
from typing import Dict, Generator, List, Optional

from openai import OpenAI, APIError, APIConnectionError, AuthenticationError, RateLimitError

from backend import vector_store
from backend.config import CHAT_MODEL, MAX_QUESTION_LENGTH, OPENAI_API_KEY, TEMPERATURE

# The exact sentence the model must use when the documents do not answer the
# question. Having one fixed phrase makes it easy to detect and to test.
NO_ANSWER = "I don't know based on the provided documents."

SYSTEM_PROMPT = f"""You are a company document assistant. You answer questions using ONLY the numbered sources given to you.

Rules:
1. Use only facts stated in the sources. Never use outside knowledge, and never guess.
2. If the sources do not contain the answer, reply with exactly this sentence and nothing else: "{NO_ANSWER}"
3. Cite the source number in square brackets after each fact, like [1] or [2].
4. If the sources disagree, say so and cite both.
5. Be concise and plain-spoken. Two to four sentences is usually enough.
6. Answer in the language the question was asked in."""

# How many earlier turns to carry, so a follow-up can be understood without the
# prompt growing without limit.
HISTORY_TURNS = 3

# A short question leaning on one of these is asking about the previous answer
# rather than starting a new topic.
REFERENTIAL_WORDS = {
    "it", "its", "that", "this", "they", "them", "their", "those", "these",
    "he", "she", "his", "her", "there", "one", "ones", "same",
}
REFERENTIAL_OPENERS = (
    "what about", "how about", "and ", "also ", "what else", "any other",
    "why", "who else", "when", "can i", "does it", "do they", "is it",
)


def _client() -> OpenAI:
    return OpenAI(api_key=OPENAI_API_KEY)


def _reply(answer: str, citations: List[Dict], status: str, **extra) -> Dict:
    """The shape every caller gets back. `extra` adds timings without changing it."""
    return {"answer": answer, "citations": citations, "status": status, **extra}


# --- Step 1: guard the input -------------------------------------------------


def validate_question(question: str) -> Optional[Dict]:
    """Return a refusal if the question cannot be asked, else None.

    Run before anything costs money or touches the index.
    """
    if not question:
        return _reply("Please type a question first.", [], "empty_question")

    if len(question) > MAX_QUESTION_LENGTH:
        return _reply(
            f"That question is {len(question)} characters long. "
            f"Please shorten it to under {MAX_QUESTION_LENGTH}.",
            [],
            "question_too_long",
        )
    return None


# --- Step 2: retrieve --------------------------------------------------------


def is_follow_up(question: str) -> bool:
    """Decide whether a question only makes sense given the one before it.

    "What about part-timers?" cannot be embedded usefully on its own - there is
    no subject in it. Rewriting the search text for questions like this is what
    makes a follow-up findable.

    Deliberately narrow: a long or self-contained question is left alone,
    because widening the search text for a genuinely new topic would retrieve
    the *previous* topic's chunks instead.
    """
    words = question.lower().rstrip("?").split()
    if not words or len(words) > 8:
        return False

    if any(question.lower().startswith(opener) for opener in REFERENTIAL_OPENERS):
        return True
    return any(word in REFERENTIAL_WORDS for word in words)


def search_text_for(question: str, history: Optional[List[Dict]]) -> str:
    """What to actually embed: the question, or the question plus its context."""
    if not history or not is_follow_up(question):
        return question

    previous = history[-1].get("question", "")
    return f"{previous} {question}".strip() if previous else question


def retrieve(question: str, history: Optional[List[Dict]] = None) -> Dict:
    """Find the supporting chunks. Never raises.

    Returns {"chunks": [...], "seconds": float, "error": str|None}. The distance
    filter lives in vector_store.search and is not changed here.
    """
    started = time.perf_counter()
    try:
        chunks = vector_store.search(search_text_for(question, history))
        return {"chunks": chunks, "seconds": round(time.perf_counter() - started, 3), "error": None}
    except Exception:
        return {
            "chunks": [],
            "seconds": round(time.perf_counter() - started, 3),
            # Friendly on purpose: this reaches the screen.
            "error": "The document index could not be searched. It may need rebuilding.",
        }


def no_context_reply(seconds: float) -> Dict:
    """What to say when nothing was retrieved.

    Either nothing is indexed at all, or nothing was close enough. Both must
    refuse rather than let the model improvise.
    """
    try:
        empty_index = vector_store.stats()["chunks"] == 0
    except Exception:
        empty_index = False

    if empty_index:
        return _reply(
            "No documents are indexed yet. Add PDFs to documents/ and run "
            "`python ingest.py`.",
            [],
            "no_documents",
            retrieval_seconds=seconds,
        )
    return _reply(NO_ANSWER, [], "no_match", retrieval_seconds=seconds)


# --- Step 3: build the prompt ------------------------------------------------


def _format_sources(chunks: List[Dict]) -> str:
    """Format the retrieved chunks as a numbered list for the prompt."""
    blocks = []
    for number, chunk in enumerate(chunks, start=1):
        location = f"{chunk['source']}, page {chunk['page']}"
        if chunk["section"]:
            location += f", section \"{chunk['section']}\""
        blocks.append(f"[{number}] ({location})\n{chunk['text']}")
    return "\n\n".join(blocks)


def _format_history(history: Optional[List[Dict]]) -> str:
    """Earlier turns, clearly labelled as conversation - not as a source.

    Kept out of the numbered source list on purpose: rule 1 of the system prompt
    only permits facts from the sources, and a previous answer is not one.
    """
    if not history:
        return ""

    lines = []
    for turn in history[-HISTORY_TURNS:]:
        question = (turn.get("question") or "").strip()
        answer = (turn.get("answer") or "").strip()
        if question and answer:
            lines.append(f"Q: {question}\nA: {answer}")

    if not lines:
        return ""
    return (
        "Earlier in this conversation (for understanding what the question "
        "refers to - not a source of facts):\n\n" + "\n\n".join(lines) + "\n\n"
    )


def build_messages(question: str, chunks: List[Dict], history=None) -> List[Dict]:
    """The full prompt. SYSTEM_PROMPT is untouched, so the rules never move."""
    user_prompt = (
        f"{_format_history(history)}"
        f"Sources:\n\n{_format_sources(chunks)}\n\n"
        f"Question: {question}"
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


# --- Step 4: generate --------------------------------------------------------


def _llm_failure(error: Exception) -> Dict:
    """Turn an OpenAI exception into a message that says what to do about it."""
    if isinstance(error, AuthenticationError):
        return _reply("Your OPENAI_API_KEY was rejected. Check the key in .env.", [], "api_error")
    if isinstance(error, RateLimitError):
        return _reply(
            "The OpenAI account is rate-limited or out of credit. Try again shortly.",
            [],
            "api_error",
        )
    if isinstance(error, APIConnectionError):
        return _reply("Could not reach OpenAI. Check your internet connection.", [], "api_error")
    if isinstance(error, APIError):
        return _reply(f"OpenAI returned an error: {error}", [], "api_error")
    return _reply("The answer could not be generated. Please try again.", [], "api_error")


def _is_refusal(answer: str) -> bool:
    return NO_ANSWER.lower().rstrip(".") in answer.lower()


def _finish(answer: str, chunks: List[Dict], seconds: float) -> Dict:
    """Decide the final reply once the model has spoken.

    If the model refused anyway, the citations are stripped - an "I don't know"
    must never arrive dressed up with sources.
    """
    answer = answer.strip()
    if _is_refusal(answer):
        return _reply(NO_ANSWER, [], "no_match", retrieval_seconds=seconds)
    return _reply(answer, chunks, "ok", retrieval_seconds=seconds)


# --- The two public entry points ---------------------------------------------


def _prepare(question: str, history) -> Dict:
    """Run the guards and retrieval that both entry points share.

    Returns either {"reply": <finished reply>} to stop, or {"chunks", "seconds"}
    to carry on to the model.
    """
    question = (question or "").strip()

    refusal = validate_question(question)
    if refusal:
        return {"reply": refusal}

    found = retrieve(question, history)
    if found["error"]:
        return {"reply": _reply(found["error"], [], "api_error", retrieval_seconds=found["seconds"])}

    if not found["chunks"]:
        return {"reply": no_context_reply(found["seconds"])}

    # Retrieval happens before the key check on purpose: if nothing is relevant
    # we can refuse correctly with no key and no network at all.
    if not OPENAI_API_KEY:
        return {
            "reply": _reply(
                "Found relevant documents, but no OPENAI_API_KEY is set, so the answer "
                "cannot be written. Copy .env.example to .env and add your key.",
                found["chunks"],
                "missing_api_key",
                retrieval_seconds=found["seconds"],
            )
        }

    return {"question": question, "chunks": found["chunks"], "seconds": found["seconds"]}


def answer_question(question: str, history: Optional[List[Dict]] = None) -> Dict:
    """Run the full pipeline and return an answer plus its citations.

    `status` tells the UI what happened: ok, empty_question, question_too_long,
    no_documents, no_match, missing_api_key, or api_error.

    `history` is optional and defaults to none, so existing callers are
    unaffected.
    """
    prepared = _prepare(question, history)
    if "reply" in prepared:
        return prepared["reply"]

    try:
        response = _client().chat.completions.create(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            messages=build_messages(prepared["question"], prepared["chunks"], history),
        )
        answer = response.choices[0].message.content or ""
    except Exception as error:
        return _llm_failure(error)

    return _finish(answer, prepared["chunks"], prepared["seconds"])


def answer_stream(
    question: str, history: Optional[List[Dict]] = None
) -> Generator[Dict, None, None]:
    """The same pipeline, yielding the answer as the model writes it.

    Event types:
        start  - citations and retrieval time, before any text arrives
        token  - one piece of the answer
        final  - the complete reply, in the same shape answer_question returns

    A caller that only wants the outcome can ignore everything but `final`.
    """
    prepared = _prepare(question, history)
    if "reply" in prepared:
        yield {"type": "final", **prepared["reply"]}
        return

    chunks, seconds = prepared["chunks"], prepared["seconds"]
    yield {"type": "start", "citations": chunks, "retrieval_seconds": seconds}

    pieces = []
    try:
        stream = _client().chat.completions.create(
            model=CHAT_MODEL,
            temperature=TEMPERATURE,
            messages=build_messages(prepared["question"], chunks, history),
            stream=True,
        )
        for part in stream:
            text = part.choices[0].delta.content or ""
            if text:
                pieces.append(text)
                yield {"type": "token", "text": text}
    except Exception as error:
        yield {"type": "final", **_llm_failure(error)}
        return

    yield {"type": "final", **_finish("".join(pieces), chunks, seconds)}
