"""The keyword half of retrieval: BM25 over the same chunks Chroma holds.

Why a second retriever at all?
    An embedding turns text into 384 numbers about its *meaning*, which is what
    makes "vacation days" find "annual leave". The same property makes it blind
    to symbols: "04-02-2026" and "22-08-2026" do not differ in meaning, so the
    model places every register row in one tight cluster. Measured on this
    corpus, the row holding the date asked about ranked 79th and 110th of 232 -
    no better than chance. BM25 counts words, so it ranked both first.

    The two fail in opposite directions, which is the reason to keep both.
    Asked for "time off entitlement", BM25 goes to the wrong policy because it
    cannot see that "time off" and "leave" mean the same thing.

Why hand-rolled rather than a library?
    It is one scoring formula over a few hundred short documents. Writing it
    here keeps the project dependency-free and local-first, and the index is
    small enough to rebuild from Chroma on demand rather than persist.
"""

import math
import re
from typing import Dict, List, Optional, Sequence, Set, Tuple

from backend.config import BM25_B, BM25_K1, RARE_DF_RATIO

STOPWORDS = frozenset(
    """
    a an the and or but if then than that this these those of on in to for from by with at as
    is are was were be been being am do does did doing have has had having
    i me my we us our you your he him his she her it its they them their
    what which who whom whose when where why how
    can could should would will shall may might must not no nor only very
    about into over under again further once here there all any both each few more most other some such
    """.split()
)
TOKEN = re.compile(r"[a-z0-9][a-z0-9\-/.]*")
PROPER_PHRASE = re.compile(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)+)\b")
def tokenize(text: str) -> List[str]:
    """Lowercase words, minus the stopwords, with values left intact."""
    return [
        token
        for token in TOKEN.findall(text.lower())
        if token not in STOPWORDS and not token.isspace()
    ]

def looks_like_value(token: str) -> bool:
    """Is this token an identifier rather than a word?

    Any digit makes it one: dates in either format, document codes, version
    numbers, amounts. Names are handled separately, as phrases.
    """
    return any(character.isdigit() for character in token)

def looks_like_identifier(token: str) -> bool:
    """A value that names one specific thing, rather than counting something.

    "Calfus-ISMS-PL-15", "04-02-2026" and "1.5" identify a document, a row and a
    clause. "45" and "2024" are quantities: a question may mention either
    without the corpus having to contain it, so they must not be treated as
    names of things - "what if I travel for 45 days?" is a fair question about
    the travel policy even though "45" appears nowhere in it.

    The difference is structure: an identifier carries a separator or mixes
    letters with digits.
    """
    if not any(character.isdigit() for character in token):
        return False
    return any(character in "-/." for character in token) or any(
        character.isalpha() for character in token
    )

def mentions_value(question: str) -> bool:
    """Does the question name something specific, whether or not we hold it?

    Deliberately separate from LexicalIndex.value_terms, which requires the
    value to be *present*. The two answer different questions and conflating
    them cost the guard its teeth: asked for "REQ-48291", a code the corpus
    does not contain, value_terms is empty, and treating that as "no value was
    named" let four unrelated register rows through. A question that names an
    identifier we do not hold should be answered "I don't know" - which needs a
    test on the shape of the question alone.

    A bare capitalised word is not enough, and that is not an oversight. "John"
    and "Vacation" are indistinguishable here, so rejecting on a single
    unknown capitalised word would break the embedding's best trick - finding
    "annual leave" from "vacation days". Names therefore only register as
    values when they come in pairs, "Sarah Connor" rather than "Sarah".
    """
    if any(looks_like_value(token) for token in tokenize(question)):
        return True
    return bool(PROPER_PHRASE.search(question))

class LexicalIndex:
    """BM25 over a fixed list of chunks, plus the vocabulary statistics the
    relevance gate needs.

    Built from whatever Chroma currently holds, so it can never disagree with
    the vector store about which chunks exist.
    """
    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        self.lowered = [text.lower() for text in self.documents]
        self.tokens = [tokenize(text) for text in self.documents]
        self.total = len(self.documents)
        self.lengths = [len(t) for t in self.tokens]
        self.average_length = (sum(self.lengths) / self.total) if self.total else 0.0

        self.frequencies: List[Dict[str, int]] = []
        self.document_frequency: Dict[str, int] = {}
        for tokens in self.tokens:
            counts: Dict[str, int] = {}
            for token in tokens:
                counts[token] = counts.get(token, 0) + 1
            self.frequencies.append(counts)
            for token in counts:
                self.document_frequency[token] = self.document_frequency.get(token, 0) + 1

        self.rare_ceiling = max(1, int(self.total * RARE_DF_RATIO))

    def is_rare(self, term: str) -> bool:
        """Present in the corpus, and in few enough chunks to pinpoint one."""
        count = self.document_frequency.get(term, 0)
        return 0 < count <= self.rare_ceiling

    def absent_identifiers(self, question: str) -> List[str]:
        """Identifiers the question names that appear in no chunk at all.

        Positive evidence that the thing asked about does not exist here, and the
        one signal the distance guard cannot produce. Asked about
        "Calfus-ISMS-PL-99", the embedding lands 0.266 from the HR Security
        Policy - all but identical to the 0.267 it gives the real
        "Calfus-ISMS-PL-15", because one digit is not a difference in meaning.
        No distance threshold can separate those, so answering from the policy
        that does exist would quietly attribute it the wrong document number.

        Only identifiers count, never bare numbers or capitalised words. Both of
        those are routinely absent from a corpus that answers the question
        anyway.
        """
        return [
            token
            for token in tokenize(question)
            if looks_like_identifier(token) and token not in self.document_frequency
        ]

    def phrase_count(self, phrase: str) -> int:
        """How many chunks contain this phrase literally."""
        needle = phrase.lower()
        return sum(1 for text in self.lowered if needle in text)

    def value_terms(self, question: str) -> List[str]:
        """The parts of a question that name a specific thing we actually hold.

        Only these let the keyword leg admit a chunk the semantic leg rejected,
        and that restriction is what stops the out-of-scope guard from being
        weakened. Two shapes qualify:

          - a token containing a digit  - "04-02-2026", "Calfus-ISMS-PL-15"
          - a capitalised multi-word phrase - "Sherrin Shariff"

        Each must also be rare. "How do I train a puppy?" offers neither: nothing
        carries a digit, and "train" - which does appear once - is an ordinary
        word, so the question stays unanswered rather than reaching a chunk on
        the strength of one incidental match.
        """
        found: List[str] = []

        for token in tokenize(question):
            if looks_like_value(token) and self.is_rare(token):
                found.append(token)

        for phrase in PROPER_PHRASE.findall(question):
            count = self.phrase_count(phrase)
            if 0 < count <= self.rare_ceiling:
                found.append(phrase.lower())
        return list(dict.fromkeys(found))

    def score_all(self, question: str) -> List[Tuple[float, int]]:
        """BM25 score for every chunk, best first, zeros dropped."""
        terms = tokenize(question)
        if not terms or not self.total:
            return []
        scored: List[Tuple[float, int]] = []
        for index, counts in enumerate(self.frequencies):
            total = 0.0
            for term in terms:
                frequency = counts.get(term)
                if not frequency:
                    continue
                appears_in = self.document_frequency[term]
                idf = math.log(1 + (self.total - appears_in + 0.5) / (appears_in + 0.5))
                length_penalty = BM25_K1 * (
                    1 - BM25_B + BM25_B * self.lengths[index] / self.average_length
                )
                total += idf * frequency * (BM25_K1 + 1) / (frequency + length_penalty)
            if total > 0:
                scored.append((total, index))

        scored.sort(key=lambda pair: -pair[0])
        return scored

    def candidates(self, question: str, limit: int) -> List[int]:
        """Chunk positions worth admitting, best BM25 score first.

        A chunk needs one of the question's value terms in it. Ranking is left
        to BM25, which weighs a rare term far above a common one, but admission
        is decided by the value term alone - a high score built entirely out of
        ordinary words is what put the Job Abandonment policy at the top of
        "time off entitlement", and the semantic leg answers that kind of
        question better anyway.
        """
        values = self.value_terms(question)
        if not values:
            return []
        wanted: List[int] = []
        for _, index in self.score_all(question):
            text = self.lowered[index]
            if any(value in text for value in values):
                wanted.append(index)
                if len(wanted) >= limit:
                    break
        return wanted

_index: Optional[LexicalIndex] = None
_index_size: int = -1

def get_index(documents: Sequence[str]) -> LexicalIndex:
    """The cached index, rebuilt when the corpus size changes.

    Cheap enough to build from scratch - a few hundred short chunks - so there
    is nothing to persist and nothing that can fall out of step with Chroma.
    """
    global _index, _index_size
    if _index is None or _index_size != len(documents):
        _index = LexicalIndex(documents)
        _index_size = len(documents)
    return _index

def reset() -> None:
    """Forget the cached index. Called after the vector store is rebuilt."""
    global _index, _index_size
    _index = None
    _index_size = -1
