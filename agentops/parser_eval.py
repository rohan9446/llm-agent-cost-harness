"""
Parser correctness.

A query can complete perfectly while the parser misread it. Ask for
"14% AAPL, 73% UNH, 8% AMD, 5% NFLX", get NVDA instead of NFLX, and the
workflow will happily analyse a portfolio nobody asked about and report
success. The Systems benchmark would then be measuring the cost of answering
the wrong question -- accurately.

So B0 reports parser accuracy alongside cost. Two tiers, kept apart because
their evidence is not equally strong:

  SHIPPED   holding count, phrasing, stated-vs-unstated lookback, and the
            lookback value. Canyon Code provides these in queries.json.

  DERIVED   ticker set and weights. Built from our own alias map, which was
            enumerated from this corpus and therefore covers it by
            construction. Reported as derived, never as ground truth.

Parser accuracy is measured and reported, not asserted as a validity gate --
a real error rate is itself a useful number, and A1 exists to move it.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from typing import Any

def load_vocab(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def vocab_sha256(path: str) -> str:
    """Checksum of the alias map the derived labels are built from.

    data/vocab.json IS the ruler for derived ticker and weight accuracy, in the
    same way advisor_eval.py is the ruler for briefing quality. It was not
    pinned anywhere: source_tree_sha256 hashes .py and .sh files only, so the
    alias map could be edited and the scorer re-run with every code hash in the
    manifest unchanged, and the reported accuracy would move with nothing in
    the artifact to show why.

    A1's name table is pinned for exactly this reason -- it decides what the
    PARSER resolves. This pins what the SCORER expects. Both halves of
    "100% ticker accuracy" now name the tables that produced it.
    """
    try:
        with open(path, "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


def scorer_sha256() -> str:
    """Checksum of this module, for the same reason advisor_eval has one."""
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""


# NO TEMPLATE LIST.
#
# An earlier version carried the corpus's sentence openings as a transcribed
# list of regexes. That published Canyon Code's dataset structure inside the
# very file meant to keep their material private -- and moving the list to a
# build script only moved the leak, because the build script is published too.
#
# The fix was to stop needing the list. Everything before the first thing that
# looks like a HOLDING -- a percentage, a known surface form, an equal-weight
# marker -- is scaffolding by definition, whatever words it happens to use.
# That rule reduces all 1,000 supplied queries to a holdings body, exactly as
# the nineteen hand-written patterns did, and it carries no corpus text at all.
#
# It is also simply better: a phrasing nobody has seen is handled the same way
# as one from the corpus, so the derived labels no longer silently drop queries
# when someone writes a sentence the list never anticipated. An earlier version
# of this project reported 98 "unresolvable" queries that turned out to be one
# missing pattern.
_TAILS = [
    r"\s*(over|for|in|during)\s+the\s+(past|last|trailing)\b.*$",
    r"\s*and summarize the concentration.*$",
    r"\s*Assess the risk\.?$",
    r"\s*Summarize the concentration\.?$",
    r"[.?]\s*$",
]
# Equal-weight markers.
#
# This was an alternation of four phrases copied out of the corpus, one of
# which ran to six words. That is the SAME leak as the template list this file
# is otherwise proud of having deleted -- smaller, and therefore missed by
# three rounds of review and by a grep that was looking for the big one. It
# survived because I fixed the finding rather than the class.
#
# Written generically it carries no corpus text and matches more: "equally
# weighted", "split evenly", "the same amount in each" and phrasings nobody
# has written yet all reduce to the same rule. Identical to the pattern
# parser_cascade.py already used, which is where this should have been copied
# from in the first place.
# The \b after the connective is load-bearing, and its absence cost two
# queries: "equal parts Intel, AMD, ..." matched "equal parts In" and left a
# body starting "tel", which resolved to nothing and dropped the whole query
# out of the derived score. Exactly the failure parser_cascade._boundaries_ok
# exists to prevent -- a pattern matching inside a word it was never meant to
# see. Caught by the full-corpus regression below, not by reading it.
_EQUAL = (r"\b(?:an?\s+)?(?:equal(?:ly)?(?:\s+split|\s+parts|\s+weight(?:s|ed)?)?"
          r"|same\s+amount|evenly)\b(?:\s+(?:of|in|across|between|among)\b"
          r"(?:\s+each\b)?(?:\s+of\b)?)?")


def _holding_start(query: str, aliases: dict[str, str]) -> int:
    """Index of the first token that looks like a holding, or 0."""
    surfaces = sorted(aliases, key=len, reverse=True)
    pat = re.compile(
        r"\d+(?:\.\d+)?\s*%|"
        r"\b(?:" + "|".join(re.escape(x) for x in surfaces) + r")\b|"
        + _EQUAL, re.I)
    m = pat.search(query)
    return m.start() if m else 0


def _strip(query: str, aliases: dict[str, str]) -> tuple[str, bool]:
    """Remove scaffolding. Returns (holdings text, equal_weighted)."""
    s = query[_holding_start(query, aliases):]
    for t in _TAILS:
        s = re.sub(t, "", s, flags=re.I)
    s = s.strip()
    equal = bool(re.search(_EQUAL, s, flags=re.I))
    s = re.sub(_EQUAL, "", s, flags=re.I)
    return s.strip(" .?:,"), equal


def _resolve(name: str, aliases: dict[str, str]) -> str | None:
    name = name.strip(" .?:%,").removeprefix("in ").strip()
    if not name:
        return None
    if name in aliases:
        return aliases[name]
    if name.upper() in aliases:
        return aliases[name.upper()]
    low = name.lower()
    for surface, tick in aliases.items():
        if surface.lower() == low:
            return tick
    return None


def expected_parse(query: str, aliases: dict[str, str]) -> dict | None:
    """Derived expected {ticker: weight} for one query.

    Returns None when any surface form fails to resolve, so unresolved queries
    are excluded from the derived score rather than counted as parser errors.
    Anything returning None belongs in the manual audit.

    Weights matter as much as tickers: 620 of the 1,000 queries are
    percentage-weighted, and a parser that swaps 90/10 for 10/90 gets the
    ticker set, the holding count and the lookback all correct while the
    workflow analyses a portfolio nobody asked about.
    """
    body, equal = _strip(query, aliases)
    if not body:
        return None

    parts = [p for p in re.split(r",\s*|\s+and\s+", body) if p.strip()]
    holdings: dict[str, float | None] = {}

    for part in parts:
        part = part.strip()
        pct = None
        m = re.search(r"(\d+(?:\.\d+)?)\s*%", part)
        if m:
            pct = float(m.group(1)) / 100.0
        name = re.sub(r"\d+(?:\.\d+)?\s*%", "", part).strip(" .?:,")
        tick = _resolve(name, aliases)
        if tick is None:
            return None
        holdings[tick] = pct

    if not holdings:
        return None

    stated = [w for w in holdings.values() if w is not None]
    if equal or not stated:
        # an equal-weight marker, or bare tickers with no percentages at all
        n = len(holdings)
        holdings = {t: 1.0 / n for t in holdings}
    elif len(stated) != len(holdings):
        return None                      # partially weighted: audit, don't guess
    else:
        total = sum(stated)
        if total <= 0:
            return None
        holdings = {t: w / total for t, w in holdings.items()}

    return holdings


def expected_tickers(query: str, aliases: dict[str, str]) -> set[str] | None:
    h = expected_parse(query, aliases)
    return set(h) if h else None


def _weights_ok(expected: dict[str, float], got: dict[str, float],
                tol: float = 1e-3) -> tuple[bool, float, float]:
    """Returns (all within tolerance, max abs error, L1 error)."""
    keys = set(expected) | set(got)
    errs = [abs(expected.get(k, 0.0) - float(got.get(k, 0.0))) for k in keys]
    return (all(e <= tol for e in errs), max(errs, default=0.0), sum(errs))


def _p95(xs: list[float]) -> float | None:
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(0.95 * (len(s) - 1))))]


def score(results: list[dict], vocab: dict[str, Any],
          failures: list[dict] | None = None,
          vocab_path: str = "") -> dict[str, Any]:
    """Per-query parser scoring, aggregated and split by phrasing class.

    `failures` matters more than it looks. A query can fail *because* the
    parser misread it -- a hallucinated ticker that is not in the frozen
    snapshot is a parse error that happens to crash downstream -- and scoring
    only over successful queries drops exactly those cases. The metric would
    then be most flattering on the parser's worst errors: the ones bad enough
    to take the workflow down with them. Failure rows carrying a `parsed`
    field are scored here alongside the successes, and `n_failed_scored`
    records how many.
    """
    aliases = vocab["aliases"]
    rows = []

    scored_input = [dict(r, _outcome="ok") for r in results]
    n_failed_scored = 0
    for f in (failures or []):
        # Only failures that got far enough to produce a parse can be scored.
        # An infrastructure failure before the parser ran has nothing to grade.
        if f.get("parsed") is None:
            continue
        scored_input.append(dict(f, _outcome="failed"))
        n_failed_scored += 1

    for r in scored_input:
        parsed = r.get("parsed") or {}
        label = r.get("label") or {}
        got_tickers = set((parsed.get("holdings") or {}).keys())
        got_lookback = parsed.get("lookback_days")
        exp_lookback = label.get("expected_lookback_days")

        exp = expected_parse(r.get("query", ""), aliases)
        exp_set = set(exp) if exp else None
        got_w = {k: float(v) for k, v in (parsed.get("holdings") or {}).items()}
        if exp is None:
            w_ok, w_max, w_l1 = None, None, None
        else:
            w_ok, w_max, w_l1 = _weights_ok(exp, got_w)

        rows.append({
            "query_id": r.get("query_id"),
            "outcome": r.get("_outcome", "ok"),
            "phrasing": label.get("phrasing"),
            # shipped
            "count_ok": len(got_tickers) == label.get("n_holdings"),
            "lookback_value_ok": got_lookback == exp_lookback,
            "lookback_stated_ok": (got_lookback is None) == (exp_lookback is None),
            # derived
            "ticker_set_ok": (None if exp_set is None else got_tickers == exp_set),
            "weights_ok": w_ok,
            "weight_max_abs_error": w_max,
            "weight_l1_error": w_l1,
            "weights_expected": exp,
            "weights_got": got_w,
            "n_holdings_expected": label.get("n_holdings"),
            "n_holdings_got": len(got_tickers),
            "tickers_expected": sorted(exp_set) if exp_set else None,
            "tickers_got": sorted(got_tickers),
        })

    def rate(key: str, subset=None) -> dict[str, Any]:
        pool = [x for x in (subset if subset is not None else rows)
                if x[key] is not None]
        if not pool:
            return {"n": 0, "accuracy": None}
        return {"n": len(pool),
                "accuracy": sum(1 for x in pool if x[key]) / len(pool)}

    by_phrasing = {}
    for ph in sorted({x["phrasing"] for x in rows if x["phrasing"]}):
        sub = [x for x in rows if x["phrasing"] == ph]
        by_phrasing[ph] = {
            "n": len(sub),
            "count_ok": rate("count_ok", sub),
            "lookback_value_ok": rate("lookback_value_ok", sub),
            "ticker_set_ok": rate("ticker_set_ok", sub),
            "weights_ok": rate("weights_ok", sub),
        }

    unresolvable = [x["query_id"] for x in rows if x["ticker_set_ok"] is None]

    return {
        "_note": "shipped labels are Canyon Code's; derived labels come from "
                 "our own alias map, which covers this corpus by construction "
                 "and proves nothing about generalization",
        "_scorer_sha256": scorer_sha256(),
        "_vocab_sha256": vocab_sha256(vocab_path) if vocab_path else "",
        "_ruler_note": ("derived ticker and weight accuracy is a property of "
                        "this alias map and this scorer; both are hashed so "
                        "two parser_eval.json files can be compared only when "
                        "they were produced by the same ruler"),
        "n_scored": len(rows),
        "n_failed_scored": n_failed_scored,
        "_failed_note": (
            "parser accuracy is computed over attempted queries, not "
            "successful ones: a parse error bad enough to crash the workflow "
            "is still a parse error, and excluding it would make the metric "
            "most optimistic exactly where the parser is worst"),
        "shipped": {
            "holding_count": rate("count_ok"),
            "lookback_value": rate("lookback_value_ok"),
            "lookback_stated_vs_unstated": rate("lookback_stated_ok"),
        },
        "derived": {
            "ticker_set": rate("ticker_set_ok"),
            "weights_within_1e-3": rate("weights_ok"),
            "weight_max_abs_error_p95": _p95(
                [x["weight_max_abs_error"] for x in rows
                 if x["weight_max_abs_error"] is not None]),
            "weight_l1_error_mean": (
                sum(x["weight_l1_error"] for x in rows
                    if x["weight_l1_error"] is not None)
                / max(1, sum(1 for x in rows if x["weight_l1_error"] is not None))),
            "n_unresolvable_by_alias_map": len(unresolvable),
            "unresolvable_query_ids": unresolvable[:20],
        },
        "by_phrasing": by_phrasing,
        "mismatches": [
            x for x in rows
            if not x["count_ok"] or not x["lookback_value_ok"]
            or x["ticker_set_ok"] is False or x["weights_ok"] is False
        ][:50],
    }
