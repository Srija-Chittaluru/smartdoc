"""Records every question asked, so the Analytics page has something to show.

One JSON object per line in data/queries.jsonl. Append-only, which means a crash
mid-write can only ever damage the last line, and the file stays readable with
any tool.

Nothing here can affect an answer: record() swallows its own errors, because
losing a statistic is acceptable and failing a question is not.
"""

import json
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Dict, List

from backend.config import PROJECT_ROOT

LOG_PATH = PROJECT_ROOT / "data" / "queries.jsonl"

# Fixed buckets rather than computed ones, so the chart's x-axis does not move
# around as new queries come in.
SIMILARITY_BUCKETS = [
    ("< 0.65", 0.00, 0.65),
    ("0.65 – 0.70", 0.65, 0.70),
    ("0.70 – 0.75", 0.70, 0.75),
    ("0.75 – 0.80", 0.75, 0.80),
    ("0.80 – 0.85", 0.80, 0.85),
    ("0.85 +", 0.85, 1.01),
]

RECENT_LIMIT = 15


def record(question: str, result: Dict, seconds: float) -> None:
    """Append one query to the log. Never raises."""
    try:
        citations = result.get("citations") or []
        scores = [float(c["score"]) for c in citations]

        entry = {
            "at": datetime.now().isoformat(timespec="seconds"),
            "question": question[:300],
            "status": result.get("status", ""),
            "seconds": round(seconds, 3),
            "top_score": max(scores) if scores else None,
            "sources": sorted({c["source"] for c in citations}),
        }

        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with LOG_PATH.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry) + "\n")

    except Exception:
        # Deliberately silent. An answer must not fail because a log did.
        pass


def load() -> List[Dict]:
    """Read the log back. Skips any line that is not valid JSON."""
    if not LOG_PATH.exists():
        return []

    entries = []
    for line in LOG_PATH.read_text(encoding="utf-8").splitlines():
        try:
            entries.append(json.loads(line))
        except json.JSONDecodeError:
            continue  # a half-written final line from an interrupted write
    return entries


def clear() -> None:
    """Forget every recorded query."""
    if LOG_PATH.exists():
        LOG_PATH.unlink()


def summary() -> Dict:
    """Everything the Analytics page needs, in one pass over the log."""
    entries = load()

    # Only answered questions carry a similarity score, so averages and usage
    # counts are taken from those. Refusals still count as questions asked.
    answered = [e for e in entries if e.get("status") == "ok"]
    scores = [e["top_score"] for e in answered if e.get("top_score") is not None]

    usage = Counter()
    for entry in answered:
        for source in entry.get("sources", []):
            usage[source] += 1

    per_day = Counter(entry["at"][:10] for entry in entries if entry.get("at"))

    distribution = [
        {
            "range": label,
            "queries": sum(1 for score in scores if low <= score < high),
        }
        for label, low, high in SIMILARITY_BUCKETS
    ]

    return {
        "questions_asked": len(entries),
        "answered": len(answered),
        "unanswered": len(entries) - len(answered),
        "avg_seconds": round(sum(e["seconds"] for e in entries) / len(entries), 2)
        if entries
        else 0.0,
        "avg_similarity": round(sum(scores) / len(scores), 3) if scores else 0.0,
        "most_queried_document": usage.most_common(1)[0][0] if usage else "—",
        "document_usage": [
            {"document": name, "citations": count} for name, count in usage.most_common()
        ],
        "questions_per_day": [
            {"date": day, "questions": count} for day, count in sorted(per_day.items())
        ],
        "similarity_distribution": distribution,
        "recent": list(reversed(entries[-RECENT_LIMIT:])),
    }
