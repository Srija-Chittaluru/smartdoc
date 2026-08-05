"""Remembers answers, so the same question is not paid for twice.

Answers are already deterministic - the model runs at temperature 0 - so asking
the same question again produces the same text. Serving the second one from
memory skips the OpenAI call entirely and returns in milliseconds instead of
seconds. In a document assistant this matters, because a handful of questions
("how many leave days do I get?") get asked over and over.

Only answers that succeeded are kept. Refusals cost nothing to reproduce, since
the distance filter stops them before the model is ever called.

Two things stop a stale answer being served:

  1. Any change to the library empties the cache (library.py calls clear()).
  2. The number of indexed chunks is part of the key, so an index rebuilt by
     another process - `python ingest.py` while the server is up - misses the
     old entries rather than matching them.

Memory only. Restarting the backend starts with an empty cache, which is the
safe direction to fail in.
"""

import hashlib
import json
from collections import OrderedDict
from typing import Dict, List, Optional

# Enough to cover the questions a team actually repeats, small enough that the
# cache cannot grow into a memory problem.
MAX_ENTRIES = 200

_entries: "OrderedDict[str, Dict]" = OrderedDict()
_hits = 0
_misses = 0


def _key(question: str, history: Optional[List[Dict]], chunk_count: int) -> str:
    """One string identifying this exact request.

    History is part of the key because the same words mean different things
    after a different previous turn: "can I carry it over?" depends entirely on
    what came before it.
    """
    previous = [
        (turn.get("question", ""), turn.get("answer", "")) for turn in (history or [])
    ]
    payload = json.dumps(
        {"q": question.strip().lower(), "h": previous, "n": chunk_count},
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get(question: str, history, chunk_count: int) -> Optional[Dict]:
    """Return a previously stored answer, or None."""
    global _hits, _misses

    found = _entries.get(_key(question, history, chunk_count))
    if found is None:
        _misses += 1
        return None

    _hits += 1
    # Copied so a caller adding fields to its reply cannot mutate what is stored.
    return dict(found)


def put(question: str, history, chunk_count: int, reply: Dict) -> None:
    """Store a successful answer. Anything else is ignored."""
    if reply.get("status") != "ok":
        return

    _entries[_key(question, history, chunk_count)] = dict(reply)

    # Oldest out first, so a long session cannot grow without limit.
    while len(_entries) > MAX_ENTRIES:
        _entries.popitem(last=False)


def clear() -> None:
    """Forget everything. Called whenever the library changes."""
    _entries.clear()


def stats() -> Dict:
    """How well the cache is doing, for the analytics endpoint."""
    asked = _hits + _misses
    return {
        "entries": len(_entries),
        "hits": _hits,
        "misses": _misses,
        "hit_rate": round(_hits / asked, 3) if asked else 0.0,
    }
