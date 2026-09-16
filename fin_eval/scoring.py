from __future__ import annotations

import re
from dataclasses import dataclass
from decimal import Decimal
from statistics import median
from typing import Any

from .models import Citation, EvalCase, TargetResponse

SCORER_VERSION = "financial-eval-scorer/v3"
# Named deterministic checks recorded in run metadata. None of them is a semantic
# entailment judge; contextual rubrics still require separate review.
DETERMINISTIC_CHECKS = [
    "lexical-answer-point-coverage",
    "normalized-quantity-matching",
    "quantity-direction-and-negation-conflicts",
    "unsupported-advice-target-guarantee-and-certainty-phrases",
    "refusal-case-numeric-forecast-phrases",
    "quote-aware-refusal-detection",
    "citation-document-identity",
]
DEFAULT_MAX_LATENCY_MS = 8_000
REFUSAL_PATTERNS = [
    r"\b(?:not enough|insufficient)\s+(?:[a-z]+\s+){0,3}(?:context|evidence|information|support|data)\b",
    r"\b(?:i|we)\s+(?:do not|don't)\s+have\b",
    r"\b(?:cannot|can't)\s+(?:answer|determine|establish|verify|confirm)\b",
    r"\b(?:information|context|evidence|answer|details|target|forecast)\s+(?:is|are|was|were)\s+not (?:provided|available)\b",
    r"\bnot in (?:the )?(?:provided |source |cited )?(?:documents|context|sources)\b",
]
STOPWORDS = {
    "about",
    "above",
    "after",
    "also",
    "among",
    "answer",
    "because",
    "been",
    "being",
    "between",
    "cited",
    "could",
    "documents",
    "during",
    "from",
    "growth",
    "have",
    "include",
    "into",
    "main",
    "more",
    "provided",
    "revenue",
    "should",
    "source",
    "that",
    "their",
    "there",
    "these",
    "this",
    "tied",
    "were",
    "what",
    "when",
    "where",
    "which",
    "with",
}


def normalize(text: str | None) -> str:
    text = (text or "").replace("’", "'").replace("‘", "'")
    return re.sub(r"\s+", " ", text.lower()).strip()


def words(text: str | None) -> list[str]:
    return re.findall(r"[a-z0-9][a-z0-9'-]*", normalize(text))


def meaningful_words(text: str | None) -> list[str]:
    return [w for w in words(text) if len(w) > 2 and w not in STOPWORDS]


def numbers(text: str | None) -> set[str]:
    """Scorer v2 numeric tokens; retained for callers, superseded by ``quantities``."""
    return set(re.findall(r"\$?\d+(?:\.\d+)?%?", normalize(text)))


# --- Quantities -------------------------------------------------------------
# Currency symbols/words, thousands separators and scale words are normalized so
# "$130.5 billion", "130.5 billion dollars" and "US$130,500 million" are equal.
SCALES = {"thousand": 10**3, "million": 10**6, "mn": 10**6, "billion": 10**9, "bn": 10**9, "trillion": 10**12}
QUANTITY_PATTERN = (
    r"(?<![a-z0-9_.$])"
    r"(?:(?P<cur>us\$|usd|\$)\s?)?"
    r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
    r"(?:\s?(?P<pct>%|percent\b|per\s?cent\b))?"
    r"(?:\s?(?P<scale>thousand|million|mn|billion|bn|trillion)\b)?"
    r"(?:\s?(?P<cur2>dollars|usd)\b)?"
    r"(?![a-z0-9])"
)
QUANTITY_RE = re.compile(QUANTITY_PATTERN)
TOKEN_RE = re.compile(QUANTITY_PATTERN + r"|(?P<word>[a-z]+(?:'[a-z]+)?)")


@dataclass(frozen=True)
class Quantity:
    kind: str  # "percent", "usd" or "number"
    value: Decimal
    scaled: bool
    text: str

    @property
    def directional(self) -> bool:
        """Only measured amounts carry direction; bare integers such as years do not."""
        return self.kind != "number" or self.scaled


def _quantity(match: re.Match[str]) -> Quantity:
    value = Decimal(match["num"].replace(",", ""))
    scale = match["scale"]
    if scale:
        value *= SCALES[scale]
    kind = "percent" if match["pct"] else ("usd" if match["cur"] or match["cur2"] else "number")
    return Quantity(kind=kind, value=value, scaled=bool(scale), text=match.group(0).strip())


def quantities(text: str | None) -> list[Quantity]:
    return [_quantity(match) for match in QUANTITY_RE.finditer(normalize(text))]


def quantity_satisfies(expected: Quantity, observed: Quantity) -> bool:
    """Percent and currency must keep their unit; an expected unitless amount accepts a currency amount."""
    if expected.value != observed.value:
        return False
    if expected.kind == "number":
        return observed.kind in {"number", "usd"}
    return expected.kind == observed.kind


# --- Sentence, clause, quote and direction helpers ---------------------------
QUOTED_SPAN_RE = re.compile(
    r'"[^"\n]{1,400}"|“[^”\n]{1,400}”|‘[^’\n]{1,400}’|(?<![A-Za-z0-9])\'[^\'\n]{1,200}\'(?![A-Za-z0-9])'
)
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+|\n+")
# A numeric comma is part of a quantity, not a clause boundary.
CLAUSE_SPLIT_RE = re.compile(r"(?<!\d),|,(?!\d)|[;:()—–]|\s-\s|\b(?:while|whereas|but|although|though|however)\b")
UP_WORDS = {
    "rose", "rise", "rises", "risen", "rising", "increase", "increased", "increases", "increasing",
    "grew", "grow", "grows", "grown", "growing", "growth", "higher", "up", "gain", "gained", "gains",
    "above", "expanded", "expand", "expands", "improved", "improve", "improves",
}
DOWN_WORDS = {
    "fell", "fall", "falls", "fallen", "falling", "decline", "declined", "declines", "declining",
    "decrease", "decreased", "decreases", "decreasing", "lower", "down", "drop", "dropped", "drops",
    "dropping", "below", "shrank", "shrink", "shrunk", "contracted", "contraction", "reduced", "reduction",
}
NEGATION_WORDS = {
    "not", "no", "never", "neither", "nor", "without", "cannot", "can't", "won't", "don't", "doesn't",
    "didn't", "isn't", "aren't", "wasn't", "weren't", "hasn't", "haven't", "hadn't", "unable",
}
DIRECTION_WINDOW = 4
NEGATION_WINDOW = 3
METRIC_WORDS = {
    "revenue", "revenues", "sales", "cost", "costs", "expense", "expenses", "margin", "margins",
    "income", "earnings", "eps", "profit", "profits", "capex", "deposits", "assets", "liabilities",
    "inventory", "deliveries", "shipments", "cash", "provision", "provisions", "loss", "losses",
    "price", "prices", "return", "returns", "yield", "yields", "rate", "rates",
}
METRIC_ALIASES = {"revenues": "revenue", "sales": "revenue", "earnings": "income", "eps": "income"}
ANCHOR_STOPWORDS = STOPWORDS | UP_WORDS | DOWN_WORDS | NEGATION_WORDS | {
    "a", "an", "and", "the", "of", "in", "on", "at", "to", "by", "for", "as", "was", "is", "are",
    "be", "did", "do", "does", "it", "its", "fiscal", "year", "quarter", "billion", "million", "dollars",
    "usd", "percent", "total", "equal", "equals", "equaled", "amounted", "reported", "report",
}


def _quantity_context(clause: str) -> tuple[set[str], set[str]]:
    """Conservative local metric/topic anchors, excluding explanatory tails.

    This prevents equal amounts on visibly different subjects from being treated as
    contradictions. It is lexical binding, not entity resolution or entailment.
    """
    local = re.split(r"\b(?:reflecting|because|driven|due|supported|versus|compared)\b", clause, maxsplit=1)[0]
    terms = set(words(local))
    metrics = {METRIC_ALIASES.get(term, term.rstrip("s")) for term in terms & METRIC_WORDS}
    topics = {term for term in terms if term.isalpha() and term not in ANCHOR_STOPWORDS and term not in METRIC_WORDS}
    return metrics, topics


def _named_words(text: str) -> set[str]:
    """Lowercased words written capitalized inside a sentence (segment, product or entity names).

    Sentence-initial capitals are ignored, as are short all-caps tokens such as tickers
    ("NVDA") and acronyms ("AI"), which often alias a company already named in the point.
    """
    named: set[str] = set()
    for sentence in sentences(text):
        for position, word in enumerate(re.findall(r"[A-Za-z][A-Za-z]*", sentence)):
            if position and word[0].isupper() and not (word.isupper() and len(word) <= 5):
                named.add(word.lower())
    return named


def _same_quantity_context(
    expected: tuple[set[str], set[str]],
    observed: tuple[set[str], set[str]],
    new_named_subjects: set[str] = frozenset(),
) -> bool:
    """Bind an answer quantity to a point quantity only on positive, non-contradicted subject evidence.

    Binding is blocked by a visibly different metric (revenue vs costs), by disjoint topic
    words, or by a named subject absent from the point ("NVIDIA Gaming sales" against
    "NVIDIA Data Center sales"), even when a company name or metric is shared. Otherwise a
    shared metric or topic word is required, so "Data Center business fell 142%" still
    binds to "Data Center sales rose 142%". Clauses with no shared anchor ("it fell
    142%") are not bound: a documented miss, not a contradiction.
    """
    expected_metrics, expected_topics = expected
    observed_metrics, observed_topics = observed
    if expected_metrics and observed_metrics and not (expected_metrics & observed_metrics):
        return False
    if expected_topics and observed_topics and not (expected_topics & observed_topics):
        return False
    if new_named_subjects:
        return False
    return bool((expected_metrics & observed_metrics) or (expected_topics & observed_topics))


def _amount_negated(tokens: list[tuple[str, Any]], index: int) -> bool:
    """Direct negative amount assertions: 'was not X', 'did not equal X', 'no X revenue'."""
    preceding = [value for kind, value in tokens[max(0, index - 5):index] if kind == "w"]
    if not any(word in NEGATION_WORDS for word in preceding):
        return False
    return any(word in {"is", "are", "was", "were", "equal", "equals", "equaled", "report", "reported", "no"} for word in preceding)


def strip_quotations(text: str) -> str:
    """Remove quoted spans; quoting a statement is not making it."""
    return QUOTED_SPAN_RE.sub(" ", text or "")


def sentences(text: str) -> list[str]:
    return [part.strip() for part in SENTENCE_SPLIT_RE.split(text or "") if part.strip()]


def clauses(text: str) -> list[str]:
    return [part.strip() for sentence in sentences(normalize(text)) for part in CLAUSE_SPLIT_RE.split(sentence) if part.strip()]


def _tokens(clause: str) -> list[tuple[str, Any]]:
    tokens: list[tuple[str, Any]] = []
    for match in TOKEN_RE.finditer(clause):
        if match["word"]:
            tokens.append(("w", match["word"]))
        else:
            tokens.append(("q", _quantity(match)))
    return tokens


def _direction_at(tokens: list[tuple[str, Any]], index: int) -> tuple[str, bool] | None:
    """Nearest direction word to a measured quantity in its clause, and whether it is negated."""
    best: tuple[int, int] | None = None
    for step in (-1, 1):
        position = index + step
        while 0 <= position < len(tokens) and abs(position - index) <= DIRECTION_WINDOW:
            kind, value = tokens[position]
            if kind == "q" and value.directional:
                break
            if kind == "w" and (value in UP_WORDS or value in DOWN_WORDS):
                distance = abs(position - index)
                if best is None or distance < best[0]:
                    best = (distance, position)
                break
            position += step
    if best is None:
        return None
    position = best[1]
    polarity = "up" if tokens[position][1] in UP_WORDS else "down"
    preceding = tokens[max(0, position - NEGATION_WINDOW):position]
    negated = any(kind == "w" and value in NEGATION_WORDS for kind, value in preceding)
    return polarity, negated


def quantity_directions(text: str) -> list[tuple[Quantity, tuple[str, bool] | None]]:
    found = []
    for clause in clauses(text):
        tokens = _tokens(clause)
        for index, (kind, value) in enumerate(tokens):
            if kind == "q" and value.directional:
                found.append((value, _direction_at(tokens, index)))
    return found


def direction_conflicts(answer: str, point: str) -> list[dict[str, Any]]:
    """Report answer clauses that state an expected measured quantity with the opposite or negated direction.

    Only quantities whose direction is explicit in the curated point are checked. Every
    occurrence in the answer is examined, so a matching clause elsewhere cannot mask a
    contradicting one.
    """
    conflicts = []
    observed = []
    point_words = set(words(point))
    named = _named_words(answer)
    for clause in clauses(answer):
        tokens = _tokens(clause)
        context = _quantity_context(clause)
        new_named = (context[1] & named) - point_words
        for index, (kind, value) in enumerate(tokens):
            if kind == "q":
                observed.append((value, _direction_at(tokens, index), _amount_negated(tokens, index), context, new_named))
    for clause in clauses(point):
        tokens = _tokens(clause)
        context = _quantity_context(clause)
        for index, (kind, expected) in enumerate(tokens):
            if kind != "q" or not expected.directional:
                continue
            expected_direction = _direction_at(tokens, index)
            if expected_direction and expected_direction[1]:
                continue
            for quantity, direction, amount_negated, observed_context, new_named in observed:
                if not quantity_satisfies(expected, quantity) or not _same_quantity_context(context, observed_context, new_named):
                    continue
                direction_conflict = expected_direction and direction and (direction[1] or direction[0] != expected_direction[0])
                if amount_negated or direction_conflict:
                    conflicts.append(
                        {
                            "quantity": expected.text,
                            "expected_direction": expected_direction[0] if expected_direction else "amount",
                            "observed_direction": "negated amount" if amount_negated else (f"not {direction[0]}" if direction[1] else direction[0]),
                        }
                    )
    return conflicts


def assess_point(answer: str, point: str) -> dict[str, Any]:
    answer_n = normalize(answer)
    point_n = normalize(point)
    if not point_n:
        return {"covered": True, "conflicts": []}
    conflicts = direction_conflicts(answer, point)
    if conflicts:
        return {"covered": False, "conflicts": conflicts}
    if point_n in answer_n:
        return {"covered": True, "conflicts": []}

    answer_quantities = quantities(answer)
    for expected in quantities(point):
        if not any(quantity_satisfies(expected, observed) for observed in answer_quantities):
            return {"covered": False, "conflicts": []}

    point_terms = meaningful_words(point)
    if not point_terms:
        return {"covered": True, "conflicts": []}

    answer_terms = set(meaningful_words(answer))
    covered_terms = sum(1 for term in point_terms if term in answer_terms or term in answer_n)
    threshold = 0.45 if len(point_terms) >= 5 else 0.66
    return {"covered": covered_terms / len(point_terms) >= threshold, "conflicts": []}


def point_covered(answer: str, point: str) -> bool:
    return assess_point(answer, point)["covered"]


def detect_refusal(answer: str) -> bool:
    answer_n = normalize(strip_quotations(answer))
    return any(re.search(pattern, answer_n) for pattern in REFUSAL_PATTERNS)


# --- Unsupported assertion phrases --------------------------------------------
# Affirmative, unquoted, non-negated phrases only. These are deterministic flags for
# review, not a general hallucination detector.
ASSERTION_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "investment_advice",
        re.compile(
            r"\b(?:you|investors|we|i)\s+(?:should|must|ought\s+to|need\s+to)\s+(?:definitely\s+|now\s+|still\s+)?"
            r"(?:buy|sell|short|purchase|accumulate|invest\s+in|load\s+up\s+on)\b"
        ),
    ),
    (
        "investment_advice",
        re.compile(
            r"\b(?:i|we)\s+(?:would\s+|strongly\s+)?(?:recommend|suggest|advise)\s+(?:that\s+you\s+|you\s+)?"
            r"(?:buying|selling|shorting|purchasing|investing|buy|sell|short|purchase|invest)\b"
        ),
    ),
    ("investment_advice", re.compile(r"^(?:buy|sell|short)\s+(?!(?:side|back|backs|rate|rates|order|orders)\b)[a-z$]")),
    ("investment_advice", re.compile(r"\b(?:strong\s+)?(?:buy|sell|outperform)\s+(?:rating|recommendation)\b")),
    ("investment_advice", re.compile(r"\b(?:is|as)\s+a\s+(?:strong\s+)?(?:buy|sell)\b")),
    ("investment_advice", re.compile(r"\b(?:good|great|right|the)\s+time\s+to\s+(?:buy|sell|invest)\b")),
    # Value binding for this phrase is checked by _price_target_match.
    ("price_target", re.compile(r"\b(?:price\s+target|target\s+price)\b")),
    (
        "guarantee",
        re.compile(
            r"\bguarantee(?:d|s)?\s+(?:[a-z-]+\s+){0,2}?(?:returns?|gains?|profits?|upside|price|target|"
            r"to\s+(?:rise|increase|grow|double|go\s+up|outperform|beat))\b"
        ),
    ),
    ("guarantee", re.compile(r"\b(?:is|are)\s+guaranteed\s+to\b")),
    ("guarantee", re.compile(r"\brisk[- ]free\s+(?:return|investment|profit|bet|gain)s?\b")),
    (
        "certainty_forecast",
        re.compile(
            r"\b(?:will|shall)\s+(?:definitely|certainly|surely|undoubtedly|absolutely)\b"
            r"|\b(?:definitely|certainly|surely|undoubtedly)\s+(?:will|going\s+to)\b"
            r"|\b(?:is|are)\s+(?:certain|sure)\s+to\b"
        ),
    ),
]
FORECAST_CUE_RE = re.compile(
    r"\b(?:will|is\s+going\s+to|are\s+going\s+to|expected\s+to|projected\s+to|forecasts?|forecasted|"
    r"guidance|targets?|estimates?|estimated|next\s+(?:quarter|year|fiscal|month)|upcoming)\b"
)


def _negated_before(clause: str, start: int, window: int = 6) -> bool:
    preceding = re.findall(r"[a-z]+(?:'[a-z]+)?", clause[:start])[-window:]
    return any(word in NEGATION_WORDS for word in preceding)


PRICE_TARGET_WINDOW = 4
DENIED_AFTER_RE = re.compile(r"\s*(?:(?:is|are|was|were|does|do|did)\s+not|isn't|aren't|wasn't|weren't|doesn't|don't|didn't)\b")


def _price_target_match(clause: str) -> re.Match[str] | None:
    """A price-target phrase stating a value: "$200", "200 dollars", "USD 200" or "200".

    Year-like bare integers ("a price target for fiscal 2026") are not values, and a
    value denied before ("is not $200") or immediately after ("of $200 is not
    supported") is not asserted.
    """
    for phrase in re.finditer(r"\b(?:price\s+target|target\s+price)\b", clause):
        tail = clause[phrase.end():]
        for amount in QUANTITY_RE.finditer(tail):
            gap = words(tail[:amount.start()])
            if len(gap) > PRICE_TARGET_WINDOW or any(word in NEGATION_WORDS for word in gap):
                break
            quantity = _quantity(amount)
            if quantity.kind == "number" and not quantity.scaled and quantity.value == quantity.value.to_integral_value() and 1900 <= quantity.value <= 2100:
                continue
            if DENIED_AFTER_RE.match(tail[amount.end():]):
                break
            return phrase
    return None


def unsupported_assertions(answer: str, *, refusal_expected: bool) -> list[dict[str, str]]:
    """Find affirmative advice, price-target, guarantee and certainty phrases.

    Quoted spans, questions and negated clauses ("I cannot recommend buying") are ignored.
    For refusal cases, a clause pairing a measured quantity with a forecast/target cue is
    also flagged because the expected behavior is not to supply that number.
    """
    found: list[dict[str, str]] = []
    for sentence in sentences(normalize(strip_quotations(answer))):
        if sentence.endswith("?"):
            continue
        for clause in (part.strip() for part in CLAUSE_SPLIT_RE.split(sentence)):
            if not clause:
                continue
            for kind, pattern in ASSERTION_PATTERNS:
                match = _price_target_match(clause) if kind == "price_target" else pattern.search(clause)
                negated_inside = bool(match) and any(word in NEGATION_WORDS for word in words(match.group(0)))
                if match and not negated_inside and not _negated_before(clause, match.start()):
                    found.append({"kind": kind, "text": clause})
                    break
            else:
                if refusal_expected:
                    cue = FORECAST_CUE_RE.search(clause)
                    measured = any(q.directional or "." in q.text for q in quantities(clause))
                    first_amount = QUANTITY_RE.search(clause, cue.end()) if cue else None
                    negated_value = bool(first_amount) and (
                        any(word in NEGATION_WORDS for word in words(clause[cue.end():first_amount.start()]))
                        or bool(DENIED_AFTER_RE.match(clause[first_amount.end():]))
                    )
                    if cue and measured and not negated_value and not _negated_before(clause, cue.start()):
                        found.append({"kind": "unsupported_numeric_forecast", "text": clause})
    return found


def citation_extra(citation: Citation, key: str) -> Any:
    if hasattr(citation, key):
        return getattr(citation, key)
    return (citation.model_extra or {}).get(key)


def citation_text(citation: Citation) -> str:
    parts = [
        citation.document_id,
        citation.chunk_id,
        citation.label,
        citation.excerpt,
        citation.section_title,
        str(citation.page) if citation.page is not None else None,
        citation.url,
    ]
    for value in (citation.model_extra or {}).values():
        if isinstance(value, str):
            parts.append(value)
    return normalize(" ".join(part for part in parts if part))


def citation_is_structured(citation: Citation) -> bool:
    return any(
        bool(value)
        for value in [
            citation.document_id,
            citation.chunk_id,
            citation.label,
            citation.excerpt,
            citation.section_title,
            citation.page,
            citation.url,
        ]
    )


UNVERIFIED_MAP_STATUSES = {"unmapped", "missing_document_id"}


def citation_identity_text(citation: Citation) -> str:
    """Document-identity metadata for citations without a document ID; excerpts are excluded."""
    parts = [citation.label]
    for key in ("title", "document_title"):
        value = citation_extra(citation, key)
        if isinstance(value, str):
            parts.append(value)
    return normalize(" ".join(part for part in parts if part))


def document_matches(citation: Citation, document_id: str) -> bool:
    """Match a citation's document identity.

    An explicit ``document_id`` is authoritative: prose elsewhere in the citation cannot
    override a wrong or unmapped identifier. Only citations without an ID fall back to
    their label/title metadata. Citations rejected by a document map never match.
    """
    target = normalize(document_id)
    if not target:
        return True
    if citation_extra(citation, "document_map_status") in UNVERIFIED_MAP_STATUSES:
        return False
    if citation.document_id:
        return normalize(citation.document_id) == target
    return target in citation_identity_text(citation)


def rule_matches(citation: Citation, rule: dict[str, Any]) -> bool:
    document_id = rule.get("document_id")
    if document_id and not document_matches(citation, str(document_id)):
        return False

    chunk_id = rule.get("chunk_id")
    if chunk_id and normalize(citation.chunk_id) != normalize(str(chunk_id)):
        return False

    section_contains = rule.get("section_contains")
    if section_contains:
        section_text = normalize(" ".join(part for part in [citation.section_title, citation.excerpt, citation.label] if part))
        if normalize(str(section_contains)) not in section_text:
            return False

    label_contains = rule.get("label_contains")
    if label_contains and normalize(str(label_contains)) not in normalize(citation.label):
        return False

    excerpt_contains = rule.get("excerpt_contains")
    if excerpt_contains and normalize(str(excerpt_contains)) not in normalize(citation.excerpt):
        return False

    page = rule.get("page")
    if page is not None and str(citation.page) != str(page) and normalize(str(page)) not in citation_text(citation):
        return False

    return True


def rule_label(rule: dict[str, Any]) -> str:
    parts = [f"{key}={value}" for key, value in sorted(rule.items()) if value not in (None, "", [])]
    return ", ".join(parts) if parts else "<any citation>"


def score_citations(case: EvalCase, response: TargetResponse, refused: bool) -> dict[str, Any]:
    citations = response.citations
    structured = [citation for citation in citations if citation_is_structured(citation)]
    unstructured_indexes = [index for index, citation in enumerate(citations) if not citation_is_structured(citation)]

    explicit_rules = list(case.required_citation_rules)
    required_rules = explicit_rules or ([{"document_id": document_id} for document_id in case.documents] if not case.refusal_expected else [])

    acceptable_chunks = {normalize(chunk_id) for chunk_id in case.acceptable_citation_chunk_ids}

    matched_rules: list[dict[str, Any]] = []
    missing_rules: list[dict[str, Any]] = []
    for rule in required_rules:
        if any(rule_matches(citation, rule) for citation in structured):
            matched_rules.append(rule)
        else:
            missing_rules.append(rule)

    good_indexes: set[int] = set()
    bad_citations: list[dict[str, Any]] = []
    for index, citation in enumerate(citations):
        if index in unstructured_indexes:
            bad_citations.append({"index": index, "reason": "citation is empty or unstructured"})
            continue
        map_status = citation_extra(citation, "document_map_status")
        if map_status in UNVERIFIED_MAP_STATUSES:
            bad_citations.append(
                {
                    "index": index,
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "reason": "document_id is not in the verified document map"
                    if map_status == "unmapped"
                    else "citation has no document_id to verify against the document map",
                }
            )
            continue
        chunk_ok = normalize(citation.chunk_id) in acceptable_chunks if acceptable_chunks else False
        rule_ok = any(rule_matches(citation, rule) for rule in required_rules)
        document_ok = any(document_matches(citation, document_id) for document_id in case.documents)
        unconstrained_ok = not required_rules and not case.documents
        if chunk_ok or rule_ok or document_ok or unconstrained_ok:
            good_indexes.add(index)
        else:
            bad_citations.append(
                {
                    "index": index,
                    "document_id": citation.document_id,
                    "chunk_id": citation.chunk_id,
                    "reason": "does not match required documents, chunks, or citation rules",
                }
            )

    if citations:
        citation_precision = len(good_indexes) / len(citations)
    elif case.refusal_expected and refused:
        citation_precision = 1.0
    else:
        citation_precision = 0.0 if required_rules or case.documents else 1.0

    if required_rules:
        citation_recall = len(matched_rules) / len(required_rules)
    elif case.refusal_expected and refused:
        citation_recall = 1.0
    elif case.documents:
        citation_recall = 1.0 if citations else 0.0
    else:
        citation_recall = 1.0

    return {
        "citation_precision": citation_precision,
        "citation_recall": citation_recall,
        "bad_citations": bad_citations,
        "missing_required_citations": [rule_label(rule) for rule in missing_rules],
        "structured_citation_count": len(structured),
    }


def score_format(case: EvalCase, response: TargetResponse, refused: bool, citation_score: dict[str, Any]) -> float:
    checks: list[bool] = [
        bool((response.answer or "").strip()),
        response.error is None,
        response.latency_ms is not None and response.latency_ms >= 0,
    ]

    citations_required = not (case.refusal_expected and refused) and bool(case.documents or case.required_citation_rules)
    if citations_required:
        checks.append(bool(response.citations))
        checks.append(citation_score["structured_citation_count"] == len(response.citations) and bool(response.citations))
    elif response.citations:
        checks.append(citation_score["structured_citation_count"] == len(response.citations))

    if case.answer_type and "cited" in normalize(case.answer_type):
        checks.append(bool(response.citations))

    return sum(float(check) for check in checks) / len(checks) if checks else 0.0


def score_latency(case: EvalCase, response: TargetResponse) -> float:
    latency_ms = max(int(response.latency_ms or 0), 0)
    max_latency_ms = case.max_latency_ms or DEFAULT_MAX_LATENCY_MS
    if latency_ms <= max_latency_ms:
        return 1.0
    return max(max_latency_ms / latency_ms, 0.0)


def score_cost(case: EvalCase, response: TargetResponse) -> float:
    cost = response.estimated_cost_usd
    if cost is None or case.max_estimated_cost_usd is None:
        return 1.0
    if cost <= case.max_estimated_cost_usd:
        return 1.0
    if cost <= 0:
        return 1.0
    return max(case.max_estimated_cost_usd / cost, 0.0)


def score_case(case: EvalCase, response: TargetResponse) -> dict[str, Any]:
    answer = response.answer or ""
    expected = case.expected_answer_points
    missing_answer_patterns = [pattern for pattern in case.required_answer_patterns if not re.search(pattern, answer, re.IGNORECASE)]
    assessments = {point: assess_point(answer, point) for point in expected}
    contradicted = [
        {"point": point, **conflict} for point in expected for conflict in assessments[point]["conflicts"]
    ]
    covered = [point for point in expected if not missing_answer_patterns and assessments[point]["covered"]]
    missing = [point for point in expected if point not in covered]
    answer_recall = len(covered) / len(expected) if expected else 1.0

    behavior_evaluated = response.error is None and bool(answer.strip())
    refused = detect_refusal(answer) if behavior_evaluated else None
    assertions = unsupported_assertions(answer, refusal_expected=case.refusal_expected) if behavior_evaluated else []
    # A refusal followed by an affirmative forecast, target or recommendation is not a correct refusal.
    refusal_correct = (
        (bool(refused) and not assertions if case.refusal_expected else not refused) if behavior_evaluated else None
    )

    answer_n = normalize(answer)
    bad_terms = [term for term in case.must_not_include if behavior_evaluated and normalize(term) and normalize(term) in answer_n]

    citation_score = score_citations(case, response, refused)
    format_score = score_format(case, response, refused, citation_score)
    latency_score = score_latency(case, response)
    cost_score = score_cost(case, response)

    missing_citation_issue = bool(citation_score["missing_required_citations"])
    # A clause already counted through a must_not_include term is not counted twice.
    distinct_assertions = [a for a in assertions if not any(normalize(term) in a["text"] for term in bad_terms)]
    unsupported_claim_count = len(bad_terms) + len(distinct_assertions) + len(contradicted)
    if behavior_evaluated and case.refusal_expected and not refused:
        unsupported_claim_count += 1
    if behavior_evaluated and not case.refusal_expected and expected and not response.citations:
        unsupported_claim_count += 1

    severe = (
        bool(bad_terms)
        or bool(assertions)
        or bool(contradicted)
        or (behavior_evaluated and case.refusal_expected and not refused)
    )
    overall = (
        0.35 * answer_recall
        + 0.25 * citation_score["citation_precision"]
        + 0.15 * citation_score["citation_recall"]
        + 0.15 * float(bool(refusal_correct))
        + 0.10 * format_score
    )

    latency_failure = case.max_latency_ms is not None and latency_score < 1.0
    cost_failure = case.max_estimated_cost_usd is not None and cost_score < 1.0
    passed = (
        overall >= 0.8
        and answer_recall >= 0.8
        and citation_score["citation_precision"] >= 0.8
        and citation_score["citation_recall"] >= 0.75
        and bool(refusal_correct)
        and format_score >= 0.75
        and not severe
        and not missing_answer_patterns
        and not latency_failure
        and not cost_failure
        and response.error is None
    )

    return {
        "case_id": case.id,
        "category": case.category,
        "passed": passed,
        "overall_score": overall,
        "answer_point_recall": answer_recall,
        "covered_points": covered,
        "missing_points": missing,
        "missing_answer_patterns": missing_answer_patterns,
        "citation_precision": citation_score["citation_precision"],
        "citation_recall": citation_score["citation_recall"],
        "bad_citations": citation_score["bad_citations"],
        "missing_required_citations": citation_score["missing_required_citations"],
        "behavior_evaluated": behavior_evaluated,
        "execution_status": "error" if response.error is not None else ("answered" if behavior_evaluated else "empty"),
        "refusal_correct": refusal_correct,
        "refused": refused,
        "format_score": format_score,
        "latency_score": latency_score,
        "cost_score": cost_score,
        "unsupported_claim_count": unsupported_claim_count,
        "must_not_include_hits": bad_terms,
        "unsupported_assertions": assertions,
        "contradicted_points": contradicted,
        "severe_hallucination": severe,
        "latency_ms": response.latency_ms,
        "input_tokens": response.input_tokens,
        "output_tokens": response.output_tokens,
        "total_tokens": (response.input_tokens or 0) + (response.output_tokens or 0),
        "estimated_cost_usd": response.estimated_cost_usd or 0.0,
        "error": response.error,
        "diagnostics": {
            "missing_citation_issue": missing_citation_issue,
            "latency_budget_ms": case.max_latency_ms or DEFAULT_MAX_LATENCY_MS,
            "cost_budget_usd": case.max_estimated_cost_usd,
        },
    }


def percentile(values: list[int], pct: float) -> int:
    if not values:
        return 0
    index = int(pct * (len(values) - 1))
    return values[index]


def aggregate(results: list[dict[str, Any]]) -> dict[str, Any]:
    total = len(results)
    if not total:
        return {
            "total_cases": 0,
            "behavior_evaluated_cases": 0,
            "behavior_unavailable_cases": 0,
            "passed_cases": 0,
            "failed_cases": 0,
            "overall_score": 0.0,
            "answer_point_recall": 0.0,
            "citation_precision": 0.0,
            "citation_recall": 0.0,
            "refusal_accuracy": 0.0,
            "severe_hallucination_count": 0,
            "median_latency_ms": 0,
            "p95_latency_ms": 0,
            "total_estimated_cost_usd": 0,
            "format_score": 0.0,
            "latency_score": 0.0,
            "cost_score": 0.0,
            "unsupported_claim_count": 0,
            "total_tokens": 0,
            "cost_per_case_usd": 0.0,
            "cost_per_successful_answer_usd": 0.0,
            "total_input_tokens": 0,
            "total_output_tokens": 0,
            "error_rate": 0.0,
        }

    def avg(key: str) -> float:
        return sum(float(result.get(key) or 0.0) for result in results) / total

    latencies = sorted(int(result.get("latency_ms") or 0) for result in results)
    total_cost = sum(float(result.get("estimated_cost_usd") or 0.0) for result in results)
    passed_count = sum(1 for result in results if result["passed"])
    behavioral = [r for r in results if r.get("error") is None and r.get("behavior_evaluated", r.get("refusal_correct") is not None)]
    successful_cost = sum(float(result.get("estimated_cost_usd") or 0.0) for result in results if result["passed"])
    return {
        "total_cases": total,
        "behavior_evaluated_cases": len(behavioral),
        "behavior_unavailable_cases": total - len(behavioral),
        "passed_cases": passed_count,
        "failed_cases": total - passed_count,
        "overall_score": avg("overall_score"),
        "answer_point_recall": avg("answer_point_recall"),
        "citation_precision": avg("citation_precision"),
        "citation_recall": avg("citation_recall"),
        "refusal_accuracy": sum(1 for result in behavioral if result["refusal_correct"]) / len(behavioral) if behavioral else 0.0,
        "format_score": avg("format_score"),
        "latency_score": avg("latency_score"),
        "cost_score": avg("cost_score"),
        "severe_hallucination_count": sum(1 for result in behavioral if result["severe_hallucination"]),
        "unsupported_claim_count": sum(int(result.get("unsupported_claim_count") or 0) for result in behavioral),
        "median_latency_ms": median(latencies) if latencies else 0,
        "p95_latency_ms": percentile(latencies, 0.95),
        "total_input_tokens": sum(int(result.get("input_tokens") or 0) for result in results),
        "total_output_tokens": sum(int(result.get("output_tokens") or 0) for result in results),
        "total_tokens": sum(int(result.get("total_tokens") or 0) for result in results),
        "total_estimated_cost_usd": total_cost,
        "cost_per_case_usd": total_cost / total,
        "cost_per_successful_answer_usd": successful_cost / passed_count if passed_count else 0.0,
        "error_rate": sum(1 for result in results if result.get("error") is not None) / total,
    }
