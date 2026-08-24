"""
A1 -- the rules-first parser cascade.

    Tier 1  deterministic scan   no model call, cannot hallucinate a ticker
    Tier 2  the B0 LLM parser    unchanged, for whatever Tier 1 declines

Same interface as ParserAgent, so nothing downstream changes and the B0/A1
comparison is a swap of one object.

WHAT TIER 1 IS ALLOWED TO KNOW
------------------------------
Two things, and the distinction is the whole experiment:

  the ticker universe    LEGITIMATE. It is the frozen snapshot -- the assets
                         the system can price. Every deployment knows this
                         independently of what anyone asks it.

  the surface forms      NOT ASSUMED. Company names come from the provider's
                         own metadata (data/names.json), never from
                         data/vocab.json, which was transcribed from these
                         exact queries. A cascade built on vocab.json would
                         report ~100% coverage on this corpus by construction
                         and would be evidence of nothing.

So Tier 1 normalises the provider's official name -- "The Coca-Cola Company",
"UnitedHealth Group Incorporated" -- and tries to match query text against it.
Where that fails, it declines and the LLM runs. The queries it declines are a
real finding.

NO TEMPLATE STRIPPING
---------------------
The scoring code in agentops/parser_eval.py strips template scaffolding using
a list of phrasings enumerated from this corpus. Tier 1 deliberately does NOT
do that -- reusing that list would smuggle the corpus back in through a side
door. Instead it scans the whole query for entity and percentage events and
pairs them by position, which is phrasing-agnostic and works on sentences
nobody has seen.

DECLINING IS A FEATURE
----------------------
Tier 1 returns None whenever anything is ambiguous: an unrecognised name, a
temporal phrase it cannot convert, a percentage count that does not match the
holding count. Guessing would trade a visible LLM cost for an invisible
correctness cost, which is the opposite of the point.
"""

from __future__ import annotations

import json
import os
import re
import threading
import unicodedata

# Corporate suffixes and articles the provider includes and users do not.
# General English/corporate vocabulary, not a list of anything in the corpus.
_SUFFIXES = [
    "incorporated", "inc", "corporation", "corp", "company", "co",
    "limited", "ltd", "plc", "holdings", "holding", "group", "class a",
    "class b", "class c", "the", "&", "and", "sa", "nv", "ag",
]
_PUNCT = re.compile(r"[^\w\s&-]+")
_WS = re.compile(r"\s+")

# Words that many companies share, so a single-token match on one of them
# identifies an industry rather than an issuer. General vocabulary; not
# assembled from anything in queries.json.
_INDUSTRY_WORDS = {
    "systems", "platforms", "technologies", "technology", "solutions",
    "services", "communications", "international", "industries",
    "enterprises", "resources", "partners", "associates", "brands",
    "stores", "retail", "energy", "financial", "finance", "bank",
    "banking", "capital", "insurance", "health", "healthcare", "pharma",
    "pharmaceutical", "pharmaceuticals", "motors", "electric", "electronics",
    "global", "national", "american", "america", "united", "general",
    "products", "materials", "media", "entertainment", "digital", "data",
    "software", "hardware", "semiconductor", "semiconductors", "micro",
    "devices", "labs", "laboratories", "research", "development",
}


def _fold(s: str) -> str:
    """Casefold, strip accents and punctuation, collapse whitespace."""
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))
    s = _PUNCT.sub(" ", s.lower())
    return _WS.sub(" ", s).strip()


def _core_name(name: str) -> str:
    """Provider name -> its distinguishing core.

    'The Coca-Cola Company'            -> 'coca cola'
    'UnitedHealth Group Incorporated'  -> 'unitedhealth'
    'JPMorgan Chase & Co.'             -> 'jpmorgan chase'
    """
    toks = _fold(name).replace("-", " ").split()
    while toks and toks[0] in _SUFFIXES:
        toks.pop(0)
    while toks and toks[-1] in _SUFFIXES:
        toks.pop()
    return " ".join(toks)


def _squash(s: str) -> str:
    """Drop spaces and hyphens entirely, so 'JP Morgan' == 'JPMorgan'."""
    return re.sub(r"[\s-]+", "", _fold(s))


class NameIndex:
    """Provider names -> tickers, plus the snapshot's ticker universe."""

    def __init__(self, names_path: str, snapshot_dir: str) -> None:
        with open(names_path, encoding="utf-8") as fh:
            payload = json.load(fh)
        self.source = payload.get("source", "unknown")
        self.names_sha256 = payload.get("names_sha256", "")

        manifest = os.path.join(snapshot_dir, "MANIFEST.json")
        with open(manifest, encoding="utf-8") as fh:
            m = json.load(fh)
        universe = m.get("tickers")
        self.universe = set(universe.keys() if isinstance(universe, dict)
                            else universe)

        # squashed form -> (ticker, n_tokens). n_tokens is carried because a
        # single-token alias and a multi-token one need different boundary
        # rules at match time -- see match_names.
        self.by_name: dict[str, tuple[str, int]] = {}
        cores: dict[str, str] = {}
        for tick, rec in payload["names"].items():
            if tick not in self.universe:
                continue
            for field in ("longName", "shortName"):
                raw = (rec.get(field) or "").strip()
                if not raw:
                    continue
                core = _core_name(raw)
                if core:
                    self.by_name.setdefault(_squash(core), (tick, len(core.split())))
                    cores.setdefault(core, tick)

        # Users shorten company names two different ways, and both need
        # indexing or the fast path declines things it should handle.
        #
        #   DROP TRAILING WORDS   "JP Morgan Chase" -> "JP Morgan"
        #                         handled by indexing leading token prefixes
        #   KEEP ONE DISTINCTIVE  "The Walt Disney Company" -> "Disney"
        #                         handled by indexing individual tokens
        #
        # Disney is the second token, so prefixes alone miss it; JPMorgan
        # written closed-up needs the two-token prefix, so tokens alone miss
        # it. Both rules are properties of the name table, not of any query --
        # run them on a different universe and they derive a different set.
        #
        # Every candidate must be UNIQUE across the universe: a form naming
        # two tickers names neither. Tokens additionally need >= 4 characters
        # ("co", "com", "of" carry no identity) and must not be industry words.
        prefix_counts: dict[str, set[str]] = {}
        token_counts: dict[str, set[str]] = {}
        for core, tick in cores.items():
            toks = core.split()
            for n in range(2, len(toks)):          # proper leading prefixes
                prefix_counts.setdefault(" ".join(toks[:n]), set()).add(tick)
            for tok in toks:
                if len(tok) >= 4 and tok not in _INDUSTRY_WORDS:
                    token_counts.setdefault(tok, set()).add(tick)

        self.token_aliases: dict[str, str] = {}
        for form, ticks in list(prefix_counts.items()) + list(token_counts.items()):
            if len(ticks) != 1:
                continue
            sq = _squash(form)
            if sq in self.by_name:
                continue
            t = next(iter(ticks))
            self.by_name[sq] = (t, len(form.split()))
            self.token_aliases[sq] = t

        # Longest first, so "Johnson & Johnson" wins over any shorter form.
        self._ordered = sorted(self.by_name, key=len, reverse=True)

    @staticmethod
    def _boundaries_ok(text: str, start: int, end: int, single: bool) -> bool:
        """Is text[start:end] a whole name rather than a fragment of a word?

        This is the guard that stops substring matching from inventing
        holdings. Without it, matching a squashed haystack finds "intel"
        inside "intelligent", "meta" inside "metadata", "amazon" inside
        "amazonian" -- every one a confident, silent, wrong parse on a
        sentence nobody wrote. The supplied corpus never triggers it, which
        is exactly why it has to be tested against sentences that do.

        Multi-token forms tolerate an adjacent hyphen, because squashing is
        what lets "Coca-Cola" match "The Coca-Cola Company" at all. Single
        tokens do not: they have no internal separator to justify it, and
        allowing one turns "oracle-like" into a position in ORCL.
        """
        before = text[start - 1] if start > 0 else ""
        after = text[end] if end < len(text) else ""
        for ch in (before, after):
            if not ch:
                continue
            if ch.isalnum():
                return False
            if ch == "-" and single:
                return False
        return True

    def match_names(self, text: str) -> list[tuple[int, int, str]]:
        """Find provider-name mentions. Returns (start, end, ticker) spans.

        Matching runs on a copy of the query with spaces and hyphens removed
        and an index back to original offsets, so 'JP Morgan' and 'JPMorgan'
        both hit -- then every candidate is checked against the ORIGINAL text
        for word boundaries, so fragments inside longer words are rejected.
        """
        squashed, back = [], []
        for i, ch in enumerate(text):
            f = _fold(ch)
            if not f or f.isspace() or ch == "-":
                continue
            squashed.append(f)
            back.append(i)
        hay = "".join(squashed)

        spans: list[tuple[int, int, str]] = []
        taken = [False] * len(hay)
        for needle in self._ordered:
            if len(needle) < 3:
                continue
            tick, n_tokens = self.by_name[needle]
            start = 0
            while True:
                j = hay.find(needle, start)
                if j < 0:
                    break
                k = j + len(needle)
                o_start, o_end = back[j], back[k - 1] + 1
                if (not any(taken[j:k])
                        and self._boundaries_ok(text, o_start, o_end,
                                                single=(n_tokens == 1))):
                    for x in range(j, k):
                        taken[x] = True
                    spans.append((o_start, o_end, tick))
                start = j + 1
        return sorted(spans)


# ---- lookback ------------------------------------------------------------
# The day conversions are Canyon Code's, taken from the SHIPPED
# expected_lookback_days labels in queries.json: years * 365 plus leftover
# months * 30, which reproduces every value they ship (6mo=180, 12mo=365,
# 18mo=545, 2y=730, 5y=1825). Using their convention is not circular -- it is
# their ground truth, not something derived from the query text.
_WORD_NUM = {
    "a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5,
    "six": 6, "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11,
    "twelve": 12, "eighteen": 18, "twenty": 20, "thirty": 30, "sixty": 60,
    "ninety": 90,
}
_UNIT_MONTHS = {"month": 1, "months": 1, "quarter": 3, "quarters": 3,
                "year": 12, "years": 12}
_UNIT_DAYS = {"day": 1, "days": 1, "week": 7, "weeks": 7}

# The quantifier is optional: "over the last month" means one month and is
# just as common as "over the last 3 months". Requiring a number made every
# such query decline, which the offline evaluation caught before any GPU time
# was spent on it.
_WINDOW = re.compile(
    r"\b(?:over|for|in|during|across)\s+the\s+"
    r"(?:past|last|trailing|previous)\s+"
    r"(?:([a-z0-9]+)\s+)?"
    r"(day|days|week|weeks|month|months|quarter|quarters|year|years)\b",
    re.I)
# A temporal phrase Tier 1 must NOT silently ignore. If one of these appears
# and _WINDOW did not fire, the query states a window in a form we cannot
# convert -- decline rather than emit lookback_days=None, which would be a
# confident claim that the user stated nothing.
_WINDOW_HINT = re.compile(
    r"\b(past|last|trailing|previous|since|ytd|year[- ]to[- ]date)\b", re.I)


def _months_to_days(months: int) -> int:
    return (months // 12) * 365 + (months % 12) * 30


def parse_lookback(query: str) -> tuple[int | None, bool]:
    """Returns (lookback_days_or_None, confident).

    confident=False means decline the whole query: something temporal is here
    that we could not read, and reporting None would assert the user stated no
    window at all.
    """
    m = _WINDOW.search(query)
    if m:
        raw, unit = (m.group(1) or "").lower(), m.group(2).lower()
        if not raw:
            n = 1                       # "over the last month" == one month
        elif raw.isdigit():
            n = int(raw)
        else:
            n = _WORD_NUM.get(raw)
        if n is None:
            return None, False
        if unit in _UNIT_DAYS:
            return n * _UNIT_DAYS[unit], True
        return _months_to_days(n * _UNIT_MONTHS[unit]), True
    if _WINDOW_HINT.search(query):
        return None, False
    return None, True


# ---- holdings ------------------------------------------------------------
_TICKER = re.compile(r"\b([A-Z]{1,5}(?:\.[A-Z])?)\b")

# ---- unknown-entity evidence --------------------------------------------
# Tier 1 knows thirty names. It does NOT know what a company looks like, so
# an issuer it cannot resolve is invisible to it rather than unresolved --
# and it will happily analyse the holdings it did recognise while dropping
# the one it did not:
#
#   "the return and volatility of Rivian, Berkshire Hathaway and ORCL"
#       -> resolved [BRK.B, ORCL],  Rivian silently gone
#
# That is a portfolio nobody asked about, produced confidently, with no model
# call left downstream to catch it -- the same shape as the baseline's
# Adobe->AAPL errors. The supplied corpus can never expose it, because every
# company in it is one of the thirty. The 91-query robustness set found it
# immediately, at a false-accept rate of 8.8%.
#
# THIS IS NOT A CLAIM THAT A CAPITALISED TOKEN IS A COMPANY. It is a
# conservative detector for *evidence* that the query names something Tier 1
# cannot see, and its false positives are deliberately acceptable:
#
#     false accept   -> wrong portfolio analysed  -> unbounded semantic cost
#     false decline  -> one LLM call              -> bounded, equals B0
#
# The two are not symmetric, so Tier 1 is tuned for precision, not coverage.
_CAP_TOKEN = re.compile(r"[A-Z][A-Za-z&'.\-]*")
# Capitalised words that carry no entity claim. Kept deliberately short: every
# addition buys coverage at the price of safety, and coverage is the cheap side.
_SAFE_CAPS = {"I", "I'm", "I've"}


def _sentence_initial(text: str, i: int) -> bool:
    """Is position i the first word of a sentence or clause?

    Capitalisation there is grammar, not reference. "Assess the risk." after a
    full stop says nothing about holdings.
    """
    j = i - 1
    while j >= 0 and text[j].isspace():
        j -= 1
    return j < 0 or text[j] in ".!?:;"


_LIST_TAIL = re.compile(r"\s*(?:and|&|plus)\s+", re.I)
_PCT_HEAD = re.compile(r"\d+(?:\.\d+)?\s*%")
_PCT_AFTER = re.compile(r"\s+\d+(?:\.\d+)?\s*%")
_CONNECTIVE = re.compile(r"\s*(?:in|of|to)?\s*", re.I)


def _in_holdings_list(query: str, s: int, e: int,
                      spans: list[tuple[int, int, str]]) -> bool:
    """Does this token sit where a HOLDING sits, whatever its position?

    The sentence-initial exemption below is a grammar rule, and grammar is not
    the only thing that puts a capital letter at the start of a clause. A
    company name does too:

        Rivian, AAPL and MSFT over the last month
        Portfolio: Rivian, AAPL and MSFT

    Both were accepted as {AAPL: 0.5, MSFT: 0.5} -- the unknown company
    dropped, the rest silently re-weighted to a portfolio nobody asked about.
    That is the precise failure A1 exists to prevent, and it survived because
    the exemption was written for one shape ("Assess the risk.") and applied to
    every shape.

    The signal is structural, not lexical -- no word list to fall out of date.
    A token is in holdings position when it is punctuated like a holding:
    followed by a list comma, joined by "and" to a recognised holding or a
    percentage, or adjacent to a percentage on either side.
    """
    tail = query[e:]

    # 1. a list comma directly after it: "Rivian, AAPL and MSFT"
    if tail[:1] == ",":
        return True

    # 2. joined by "and" to a recognised holding or to a percentage:
    #    "Snowflake and AAPL", "Snowflake and 40% AAPL"
    m = _LIST_TAIL.match(tail)
    if m:
        nxt = e + m.end()
        if any(cs == nxt for cs, _ce, _t in spans):
            return True
        if _PCT_HEAD.match(query[nxt:]):
            return True

    # 3. a percentage immediately after it -- but ONLY when that percentage
    #    does not belong to the holding that follows.
    #
    #    This distinction is the whole rule. "Rivian 40% and AAPL 60%" puts the
    #    percentage AFTER its holding, so Rivian is in holdings position. But
    #    "Evaluate 100% CRM" puts it BEFORE, and the percentage binds forward
    #    to CRM -- the leading word is a verb, not a company. Without the
    #    lookahead this fired on 44 of the 1,000 corpus queries, every one of
    #    them an instruction verb, and took Tier-1 coverage from 100% to 95.6%.
    pm = _PCT_AFTER.match(tail)
    if pm:
        after = e + pm.end()
        skip = _CONNECTIVE.match(query[after:])
        pos = after + (skip.end() if skip else 0)
        if not any(cs == pos for cs, _ce, _t in spans):
            return True

    return False


def unknown_entity_evidence(query: str, spans: list[tuple[int, int, str]],
                            index: "NameIndex") -> str | None:
    """A token suggesting a holding Tier 1 cannot resolve, or None."""
    covered = [(s, e) for s, e, _ in spans]
    for m in _CAP_TOKEN.finditer(query):
        s, e = m.start(), m.end()
        if any(cs <= s < ce for cs, ce in covered):
            continue
        tok = m.group(0).strip(".")
        if not tok or tok in _SAFE_CAPS:
            continue
        # Sentence-initial capitalisation is usually grammar -- unless the
        # token is punctuated like a holding, in which case position tells us
        # nothing and the name does. Declining costs one LLM call; accepting
        # costs a wrong portfolio delivered confidently.
        if _sentence_initial(query, s) and not _in_holdings_list(query, s, e, spans):
            continue
        # An all-caps token outside the universe is an unpriceable symbol --
        # exactly the GOOG-for-GOOGL case, and worth declining on.
        if tok.isupper() and tok in index.universe:
            continue
        return tok
    return None
_PCT = re.compile(r"(\d+(?:\.\d+)?)\s*%")
_EQUAL = re.compile(
    r"\b(equal(?:ly)?(?:\s+split|\s+parts|\s+weight(?:s|ed)?)?|"
    r"same\s+amount|evenly)\b", re.I)


def parse_holdings(query: str, index: NameIndex) -> dict[str, float] | None:
    """Scan for entities and percentages, pair them by position.

    Phrasing-agnostic on purpose: no template list, so a sentence shape the
    corpus never used is handled the same way.
    """
    spans: list[tuple[int, int, str]] = index.match_names(query)
    claimed = [(s, e) for s, e, _ in spans]

    for m in _TICKER.finditer(query):
        sym = m.group(1)
        if sym not in index.universe:
            continue
        if any(s <= m.start() < e for s, e in claimed):
            continue           # inside a company name already matched
        spans.append((m.start(), m.end(), sym))
    spans.sort()

    if not spans:
        return None

    unknown = unknown_entity_evidence(query, spans, index)
    if unknown is not None:
        return None

    # Duplicate mention of one ticker ("AAPL ... Apple") is ambiguous about
    # how many holdings were meant. Decline.
    tickers = [t for _, _, t in spans]
    if len(set(tickers)) != len(tickers):
        return None

    pcts = [(m.start(), float(m.group(1)) / 100.0) for m in _PCT.finditer(query)]

    if not pcts:
        # "lists holdings with no weights at all" -> split evenly, which is
        # what the shipped instruction specifies. An explicit "equal parts"
        # lands here too and gets the same answer.
        n = len(spans)
        return {t: 1.0 / n for _, _, t in spans}

    if len(pcts) != len(spans):
        return None            # partially weighted: audit, never guess

    # Stated weights must actually describe a portfolio. Portfolio weights sum
    # to one by definition, so "120% Visa and 30% Pfizer" is not a portfolio
    # with an unusual scale -- it is a request Tier 1 has misread or a user who
    # mistyped. Normalising it to 80/20 would invent an intent nobody
    # expressed.
    #
    # The band is 2 percentage points. Every percentage query in the supplied
    # corpus sums to exactly 100, so any band would pass it -- the width is set
    # by what a human plausibly writes, not by the corpus. Someone splitting
    # three ways types 33/33/33 and lands on 99; six holdings rounded to whole
    # percents can drift further. Two points covers that and still rejects
    # 120+30=150 by a factor of twenty-five.
    #
    # (1pp was the first attempt and failed its own regression: 0.33*3 is
    # 0.9899999... in binary floating point, so an exact 1pp boundary rejected
    # the very case it was written to allow.)
    stated = sum(v for _p, v in pcts)
    if abs(stated - 1.0) > 0.02:
        return None

    # Pair by ADJACENCY IN SEQUENCE, not by character distance.
    #
    # Distance-based pairing looks reasonable and is wrong. In "61% in Adobe,
    # 14% in Costco" the "14%" is physically closer to "Adobe" than "61%" is,
    # because "% in " sits between them. It silently transposed weights on 27%
    # of the corpus while getting every ticker right -- the worst failure shape
    # there is, since the answer stays plausible. The offline evaluation caught
    # it; a GPU run would have reported it as success.
    #
    # (The example is invented. It was the real corpus sentence until the
    # content audit found it here, which is how a comment ends up publishing a
    # dataset.)
    #
    # Walking the merged event stream instead handles both orders exactly:
    # a percent binds to the next entity if it arrives first ("61% in Adobe"),
    # or to the entity still waiting ("Boeing 25%").
    events = sorted(
        [(pos, "pct", val) for pos, val in pcts]
        + [(s, "ent", t) for s, _e, t in spans]
    )
    pairs: dict[str, float] = {}
    pending_pct: float | None = None
    waiting_ent: str | None = None
    for _pos, kind, payload in events:
        if kind == "pct":
            if waiting_ent is not None:
                pairs[waiting_ent] = payload
                waiting_ent = None
            elif pending_pct is None:
                pending_pct = payload
            else:
                return None          # two percentages, no holding between
        else:
            if pending_pct is not None:
                pairs[payload] = pending_pct
                pending_pct = None
            elif waiting_ent is None:
                waiting_ent = payload
            else:
                return None          # two holdings, no percentage between

    if pending_pct is not None or waiting_ent is not None:
        return None
    if len(pairs) != len(spans):
        return None
    total = sum(pairs.values())
    if total <= 0:
        return None
    return {t: w / total for t, w in pairs.items()}


class CascadeParserAgent(object):
    """Tier 1 rules, Tier 2 the unmodified LLM parser."""

    def __init__(self, names_path: str | None = None,
                 snapshot_dir: str | None = None) -> None:
        from parser_agent import ParserAgent
        root = os.environ.get("B0_ROOT") or os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__)))))
        self.index = NameIndex(
            names_path or os.path.join(root, "data", "names.json"),
            snapshot_dir or os.environ.get(
                "PRICE_SNAPSHOT_DIR", os.path.join(root, "data", "snapshot")),
        )
        self.llm_parser = ParserAgent()
        self.tools = [self.parse]
        self.stats = {"tier1": 0, "tier2": 0, "tier2_reasons": {}}
        # One Pipeline is shared across the worker pool, so these counters are
        # incremented concurrently. `+= 1` on a dict value is not atomic, and
        # an undercount here would overstate Tier 1 coverage -- the headline
        # A1 number. Cheap lock, uncontended relative to a model call.
        self._lock = threading.Lock()

    def reset_stats(self) -> None:
        """Zero the tier counters.

        Called after warm-up, alongside llm.COUNTER.reset(). Without it the
        two counters describe different windows -- LLM calls counted from the
        start of measurement, tier coverage counted from process start -- and
        a 100-query run reports "108 of 108", which is not a coverage figure
        for anything that was measured.
        """
        with self._lock:
            self.stats = {"tier1": 0, "tier2": 0, "tier2_reasons": {}}

    def _decline(self, reason: str) -> None:
        with self._lock:
            self.stats["tier2"] += 1
            self.stats["tier2_reasons"][reason] = \
                self.stats["tier2_reasons"].get(reason, 0) + 1

    def parse(self, query: str) -> dict:
        lookback, confident = parse_lookback(query)
        if not confident:
            self._decline("unreadable_window")
            return self.llm_parser.parse(query=query)

        holdings = parse_holdings(query, self.index)
        if not holdings:
            self._decline("unresolved_holdings")
            return self.llm_parser.parse(query=query)

        with self._lock:
            self.stats["tier1"] += 1
        return {"holdings": holdings, "lookback_days": lookback}
