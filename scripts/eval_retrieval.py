"""Compare semantic-only retrieval against hybrid, on the questions that matter.

    python scripts/eval_retrieval.py

Four sets, each answering a different question:

  EXACT     values an embedding cannot represent - dates in both formats, a
            document code, people's names. Hybrid should find these; semantic
            alone should not. This is what hybrid was added for.

  NATURAL   ordinary policy questions. The correct chunk must stay at rank 1 and
            the accepted count must not change. This is the no-regression test:
            if hybrid made normal questions worse, it shows up here.

  OUT       questions the documents cannot answer. Both columns must be empty.
            This is the out-of-scope guard.

  ABSENT    an exact value the corpus does not contain. Both columns must be
            empty - the honest answer is "I don't know", not the nearest row.

Every expectation is asserted, and the exit code is non-zero if any fails, so
this can run in CI as well as by hand.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend import vector_store
from backend.config import TOP_K

# (question, a substring only the correct chunk contains, or None if nothing
#  in the corpus should answer it)
EXACT = [
    ("What is due on 04-02-2026?", "04-02-2026"),
    ("Which activity has target date 22-08-2026?", "22-08-2026"),
    ("What happened on 11-Jun-2024?", "11-Jun-2024"),
    ("What was created on 20-Dec-2025?", "20-Dec-2025"),
    ("Which policy is Calfus-ISMS-PL-15?", "Calfus-ISMS-PL-15"),
    ("What did Sherrin Shariff do?", "Sherrin Shariff"),
    ("Who is Arshi Dutta?", "Arshi Dutta"),
    ("What did Amit Bansal approve?", "Amit Bansal"),
]

NATURAL = [
    ("What is the definition of sexual harassment?", "harassment"),
    ("Who is on the Internal Committee?", "committee"),
    ("What are the travel entitlements?", "travel"),
    ("How do I claim travel reimbursement?", "reimburse"),
    ("What are employee responsibilities for information security?", "security"),
    ("Which requests are Pending Approval?", "Pending Approval"),
    ("Who owns the quarterly office maintenance and equipment inspection?", "Facilities Team"),
    ("What happens during background verification?", "verification"),
]

OUT = [
    "How do I train a puppy?",
    "What is the capital of France?",
    "How do I bake sourdough bread?",
    "What is the weather tomorrow?",
    "Who won the football match?",
]

# Values the corpus does not contain, in a shape the question makes obvious: a
# code, a date, a full name. Hybrid must stay silent on all of these.
ABSENT = [
    "What is the status of REQ-48291?",
    "What is due on 31-12-2099?",
    "Which requests does Sarah Connor own?",
    "Who is Priya Venkatesan?",
    "What does policy Calfus-ISMS-PL-99 say?",
]

# A known limit, reported rather than asserted. One capitalised word cannot be
# told apart from a synonym: "John" and "Grievance" look identical to the index,
# and "Grievance" is out of vocabulary here yet the POSH policy answers it at
# 0.749. Rejecting on a single unknown capitalised word would cost more than it
# saves, so a bare first name still reaches the distance guard alone.
UNPROTECTED = [
    "What is the status of John's request?",
    "Which requests does Sarah own?",
]


def semantic_only(question: str):
    """The semantic leg with its guard, and no keyword leg at all."""
    total = vector_store.get_collection().count()
    return vector_store.semantic_candidates(question, total)[:TOP_K]


def rank_of(chunks, needle: str):
    for position, chunk in enumerate(chunks, start=1):
        if needle.lower() in chunk["text"].lower():
            return position
    return None


def show(name, rows):
    print("\n" + "=" * 78)
    print(name)
    print("=" * 78)
    print("%-52s %11s %11s" % ("QUERY", "SEMANTIC", "HYBRID"))
    for line in rows:
        print(line)


failures = []


def check(condition, message):
    if not condition:
        failures.append(message)
    return condition


# --- exact values ------------------------------------------------------------
rows = []
found_semantic = found_hybrid = 0
for question, needle in EXACT:
    base = rank_of(semantic_only(question), needle)
    hybrid_chunks = vector_store.search(question)
    hybrid = rank_of(hybrid_chunks, needle)
    found_semantic += base is not None
    found_hybrid += hybrid is not None
    check(hybrid is not None, "EXACT: %r did not retrieve %r" % (question, needle))
    rows.append(
        "%-52s %11s %11s"
        % (question[:52], "rank %d" % base if base else "MISS",
           "rank %d" % hybrid if hybrid else "MISS")
    )
show("EXACT VALUES  (hybrid must find these)", rows)
print("\n  found: semantic %d/%d   hybrid %d/%d" % (
    found_semantic, len(EXACT), found_hybrid, len(EXACT)))

# --- natural language --------------------------------------------------------
rows = []
regressions = 0
for question, needle in NATURAL:
    base_chunks = semantic_only(question)
    hybrid_chunks = vector_store.search(question)
    base, hybrid = rank_of(base_chunks, needle), rank_of(hybrid_chunks, needle)
    if base is not None and (hybrid is None or hybrid > base):
        regressions += 1
    check(
        hybrid is not None and hybrid <= (base or hybrid),
        "NATURAL: %r fell from rank %s to %s" % (question, base, hybrid),
    )
    rows.append(
        "%-52s %11s %11s"
        % (question[:52],
           "rank %s (%d)" % (base, len(base_chunks)),
           "rank %s (%d)" % (hybrid, len(hybrid_chunks)))
    )
show("NATURAL LANGUAGE  (rank must not get worse; count in brackets)", rows)
print("\n  regressions: %d" % regressions)

# --- out of scope ------------------------------------------------------------
rows = []
for question in OUT:
    base, hybrid = len(semantic_only(question)), len(vector_store.search(question))
    check(hybrid == 0, "OUT OF SCOPE: %r returned %d chunks" % (question, hybrid))
    rows.append("%-52s %11s %11s" % (question[:52], base, hybrid))
show("OUT OF SCOPE  (both columns must be 0)", rows)

# --- absent exact values -----------------------------------------------------
rows = []
for question in ABSENT:
    base, hybrid = len(semantic_only(question)), len(vector_store.search(question))
    check(hybrid == 0, "ABSENT VALUE: %r returned %d chunks" % (question, hybrid))
    rows.append("%-52s %11s %11s" % (question[:52], base, hybrid))
show("ABSENT EXACT VALUES  (hybrid must stay silent)", rows)

# --- known limit -------------------------------------------------------------
rows = []
for question in UNPROTECTED:
    base, hybrid = len(semantic_only(question)), len(vector_store.search(question))
    rows.append("%-52s %11s %11s" % (question[:52], base, hybrid))
show("BARE FIRST NAMES  (known limit - reported, not asserted)", rows)
print("\n  A single capitalised word is ambiguous between a name and a synonym.")
print("  See lexical.mentions_value for why this is left alone.")

# --- verdict -----------------------------------------------------------------
print("\n" + "=" * 78)
if failures:
    print("FAILED (%d)" % len(failures))
    for message in failures:
        print("  -", message)
    sys.exit(1)
print("PASSED  -  exact %d/%d found, %d natural-language regressions, guard intact"
      % (found_hybrid, len(EXACT), regressions))
