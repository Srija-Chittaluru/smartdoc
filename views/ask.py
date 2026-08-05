"""Page 1 - Ask Documents. The landing page.

The workflow is unchanged: type a question, the backend retrieves and answers,
the answer appears with its sources. This file decides what is on the page; the
HTTP calls live in ui/api.py and the shared pieces in ui/components.py.

Answers stream in as the model writes them. Follow-up questions work because the
last few turns are sent along, so "can I carry it over?" knows what "it" is.
"""

import time

import streamlit as st

from ui import api, components

# Shown whenever the documents cannot answer. One fixed sentence, so the app
# never dresses up a refusal as an answer.
NO_ANSWER_MESSAGE = (
    "The uploaded documents do not contain enough information to answer this question."
)

SUGGESTED_QUESTIONS = [
    "How many days of annual leave do I get?",
    "What is the policy on working remotely from abroad?",
    "How do I claim back a business expense?",
    "What are the password requirements?",
]

# How many earlier turns to send, and how many past questions to offer back.
HISTORY_TURNS = 3
RECENT_LIMIT = 6


# --- Conversation state ------------------------------------------------------


def conversation_history() -> list:
    """The last few completed question/answer pairs, for the backend.

    Only answered turns are included - sending a refusal as context would
    invite the model to treat "I don't know" as a fact about the documents.
    """
    turns = []
    for message in st.session_state.chat:
        if message["role"] == "assistant" and message.get("answered"):
            turns.append({"question": message["question"], "answer": message["content"]})
    return turns[-HISTORY_TURNS:]


def remember_question(question: str):
    """Keep a short list of past questions, most recent first, no repeats."""
    recent = st.session_state.setdefault("recent_questions", [])
    if question in recent:
        recent.remove(question)
    recent.insert(0, question)
    del recent[RECENT_LIMIT:]


def queue_question(question: str):
    """Hand a question to the next rerun.

    Suggestions, recent questions and the chat box all go through here, so there
    is only one path that asks anything.
    """
    st.session_state.pending_question = question
    st.rerun()


# --- Drawing an answer -------------------------------------------------------


def draw_answer(message: dict):
    """Render one assistant reply: badge, citations, timings, copy icon."""
    components.confidence_badge(message["citations"])
    components.render_citations(message["citations"])
    components.answer_footer(
        message["seconds"], message["citations"], message.get("retrieval_seconds")
    )
    components.copy_answer(message["content"])


def show_past_message(message: dict):
    """Redraw one message from the history."""
    avatar = "🧑" if message["role"] == "user" else "📄"
    with st.chat_message(message["role"], avatar=avatar):
        st.markdown(message["content"])
        if message["role"] == "assistant" and message.get("answered"):
            draw_answer(message)


def stream_new_answer(question: str) -> dict:
    """Ask the backend, streaming the answer in. Returns the message to keep.

    The generator handed to st.write_stream yields only text; everything else
    the backend sends is collected into `received` for use once it finishes.
    """
    received = {}

    def text_pieces():
        for event in api.ask_question_streamed(question, conversation_history()):
            kind = event.get("type")
            if kind == "token":
                yield event["text"]
            elif kind == "start":
                received["citations"] = event["citations"]
                received["retrieval_seconds"] = event["retrieval_seconds"]
            elif kind == "final":
                received["final"] = event
            elif kind == "error":
                received["error"] = event["message"]

    with st.chat_message("assistant", avatar="📄"):
        started = time.perf_counter()
        with st.spinner("Searching the documents…"):
            streamed = st.write_stream(text_pieces())
        seconds = time.perf_counter() - started

        if "error" in received:
            st.error(received["error"])
            return {"role": "assistant", "content": received["error"], "answered": False}

        final = received.get("final")
        if final is None:
            message = "The answer was cut short. Please try again."
            st.error(message)
            return {"role": "assistant", "content": message, "answered": False}

        status = final["status"]
        citations = final.get("citations", [])

        # Nothing relevant found, or nothing indexed at all. Both are "no
        # answer", so both get the same fixed sentence.
        if status in ("no_match", "no_documents"):
            st.warning(NO_ANSWER_MESSAGE)
            if status == "no_documents":
                st.caption("The document library is empty.")
            return {"role": "assistant", "content": NO_ANSWER_MESSAGE, "answered": False}

        # A setup problem - a missing key, say. Retrieval still worked, so show
        # what it found.
        if status != "ok":
            st.info(final["answer"])
            components.render_citations(citations)
            return {"role": "assistant", "content": final["answer"], "answered": False}

        message = {
            "role": "assistant",
            "question": question,
            # streamed is what the reader actually saw; final["answer"] is the
            # same text, and is the fallback if nothing streamed.
            "content": streamed or final["answer"],
            "citations": citations,
            "seconds": seconds,
            "retrieval_seconds": final.get("retrieval_seconds"),
            "answered": True,
        }
        draw_answer(message)
        return message


# --- The page ----------------------------------------------------------------

components.page_header(
    "📄",
    "SmartDoc",
    "Ask your company documents anything — every answer is cited to the page.",
)

if "chat" not in st.session_state:
    st.session_state.chat = []

# Resolved before anything is drawn: a question from a button has to replace the
# empty state on this run, not the next one. st.chat_input pins itself to the
# bottom of the viewport wherever it is called.
typed = st.chat_input("Ask about leave, expenses, security, onboarding…")
question = typed or st.session_state.pop("pending_question", None)

if not st.session_state.chat and not question:
    st.markdown(
        '<div class="sd-welcome"><h3>Ask in plain English</h3>'
        "<p>SmartDoc searches your documents by meaning rather than keywords, then "
        "answers using only what it found — with the document, page and section "
        "shown under every answer so you can check it yourself.</p>"
        "<p style='margin-top:.7rem'>Follow-up questions work too: ask "
        "<i>“can I carry it over?”</i> after a question about leave and it will "
        "know what you mean.</p></div>",
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sd-eyebrow">Suggested questions</div>', unsafe_allow_html=True)

    left, right = st.columns(2, gap="small")
    for index, suggestion in enumerate(SUGGESTED_QUESTIONS):
        column = left if index % 2 == 0 else right
        if column.button(suggestion, key=f"suggested-{index}"):
            queue_question(suggestion)

elif st.session_state.get("recent_questions"):
    with st.expander(f"Recent questions ({len(st.session_state.recent_questions)})"):
        for index, past in enumerate(st.session_state.recent_questions):
            if st.button(past, key=f"recent-{index}"):
                queue_question(past)

for message in st.session_state.chat:
    show_past_message(message)

if question:
    remember_question(question)
    st.session_state.chat.append({"role": "user", "content": question})
    with st.chat_message("user", avatar="🧑"):
        st.markdown(question)

    st.session_state.chat.append(stream_new_answer(question))
