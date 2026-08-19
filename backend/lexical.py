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
# A PDF table's line breaks survive the Markdown conversion as "<br>", which
# splits a name down the middle: "Arshi<br>Dutta".
LINE_BREAK = re.compile(r"<br\s*/?>", re.IGNORECASE)
def tokenize(text: str) -> List[str]:
    """Lowercase words, minus the stopwords, with values left intact.

    A slash is part of a token so that "04/02/2026" survives whole, but a
    Markdown table also crams cells together as "Singh/Arshi" - one token the
    corpus holds and no question would ever type. Between words the slash is
    therefore a separator; between digits it stays.
    """
    tokens: List[str] = []
    for token in TOKEN.findall(text.lower()):
        parts = [token] if any(c.isdigit() for c in token) else token.split("/")
        tokens.extend(part for part in parts if part and part not in STOPWORDS)
    return tokens

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

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}
WRITTEN_DATE = re.compile(
    r"\b(?:(\d{1,2})\s+([A-Za-z]{3,9})|([A-Za-z]{3,9})\s+(\d{1,2})),?\s+(\d{4})\b"
)
SEPARATED_DATE = re.compile(r"\b(\d{1,2})[/.](\d{1,2})[/.](\d{4})\b")

def date_variants(question: str) -> List[str]:
    """The dates in a question, rewritten in the two forms the documents use.

    A date is one token only when it is punctuated: "04-02-2026" survives
    tokenize intact, while "04 Feb 2026" becomes "04", "feb" and "2026" - three
    pieces the corpus never contains, because it always writes the date glued
    together. The two spellings mean the same thing and matched nothing in
    common, so the keyword leg found the first and went silent on the second.
    """
    found: List[str] = []

    for match in WRITTEN_DATE.finditer(question):
        day, year = match.group(1) or match.group(4), match.group(5)
        name = (match.group(2) or match.group(3))[:3].lower()
        if name in MONTHS:
            found.append("%02d-%02d-%s" % (int(day), MONTHS[name], year))
            found.append("%02d-%s-%s" % (int(day), name, year))

    for day, month, year in SEPARATED_DATE.findall(question):
        found.append("%02d-%02d-%s" % (int(day), int(month), year))

    return found

def word_pairs(words: Sequence[str]) -> List[str]:
    """Adjacent word pairs, skipping any pair that leans on a stopword.

    A document writing "The Internal Committee" would otherwise offer the pair
    "the internal", which is rare only by accident and identifies nothing.
    """
    return [
        f"{first} {second}"
        for first, second in zip(words, words[1:])
        if first not in STOPWORDS and second not in STOPWORDS
    ]

class LexicalIndex:
    """BM25 over a fixed list of chunks, plus the vocabulary statistics the
    relevance gate needs.

    Built from whatever Chroma currently holds, so it can never disagree with
    the vector store about which chunks exist.
    """
    def __init__(self, documents: Sequence[str]):
        self.documents = list(documents)
        # The literal checks - phrase counting and name detection - read the text
        # with a table's line breaks as spaces, so a name written across two
        # lines of one cell is still the one phrase the question asks for.
        readable = [LINE_BREAK.sub(" ", text) for text in self.documents]
        self.lowered = [text.lower() for text in readable]
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

        # Every name the corpus itself writes capitalised, kept as lowercase
        # adjacent word pairs. This is what lets a question find a name it did
        # not capitalise; see named_phrases.
        self.name_pairs: Set[str] = set()
        for text in readable:
            for phrase in PROPER_PHRASE.findall(text):
                self.name_pairs.update(word_pairs(phrase.lower().split()))

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

    def normalize_dates(self, question: str) -> str:
        """Add corpus-format spellings of any date the question writes out.

        Only forms that are actually present are added. A date we do not hold
        contributes nothing, so absent_identifiers still sees an unknown date
        as unknown rather than being handed a spelling that was invented here.
        """
        held = [v for v in date_variants(question) if v in self.document_frequency]
        return " ".join([question] + held) if held else question

    def named_phrases(self, question: str) -> List[str]:
        """Names the question mentions, however it happened to capitalise them.

        PROPER_PHRASE can only see "Arshi Dutta", and a question typed into a
        chat box is usually all lowercase: "who is arshi dutta" named a person
        the corpus holds, offered the keyword leg no anchor, and left the answer
        to whatever the embedding thought was nearest - a page of contents.

        A pair of adjacent words qualifies only if the corpus writes that same
        pair as a proper name, so this cannot let an ordinary phrase in: the
        evidence comes from the documents, not from how the question was typed.
        """
        words = re.findall(r"[a-z][a-z'\-]*", question.lower())
        return [
            phrase
            for phrase in word_pairs(words)
            if phrase in self.name_pairs
            and 0 < self.phrase_count(phrase) <= self.rare_ceiling
        ]

    def value_terms(self, question: str) -> List[str]:
        """The parts of a question that name a specific thing we actually hold.

        Only these let the keyword leg admit a chunk the semantic leg rejected,
        and that restriction is what stops the out-of-scope guard from being
        weakened. Two shapes qualify:

          - an identifier - "04-02-2026", "Calfus-ISMS-PL-15"
          - a capitalised multi-word phrase - "Sherrin Shariff"
          - the same phrase uncapitalised, when the corpus capitalises it -
            "arshi dutta", because a document writes "Arshi Dutta"

        An identifier, not any token carrying a digit. A bare "4" is rare enough
        to pass, and admission below is by substring, so it matched "1.14" and
        "Purpose 3" and put four unrelated chunks under "4 February 2026".

        Each must also be rare. "How do I train a puppy?" offers neither: nothing
        carries a digit, and "train" - which does appear once - is an ordinary
        word, so the question stays unanswered rather than reaching a chunk on
        the strength of one incidental match.
        """
        found: List[str] = []

        for token in tokenize(question):
            if looks_like_identifier(token) and self.is_rare(token):
                found.append(token)

        for phrase in PROPER_PHRASE.findall(question):
            count = self.phrase_count(phrase)
            if 0 < count <= self.rare_ceiling:
                found.append(phrase.lower())

        found.extend(self.named_phrases(question))
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
