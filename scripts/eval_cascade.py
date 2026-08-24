#!/usr/bin/env python3
"""
Tier 1 coverage and correctness, offline. No GPU, no model, no network.

Answers the two questions that decide whether A1 is worth running at all:

  COVERAGE     what fraction of queries does the deterministic tier resolve
               without a model call? That fraction is the parser cost that
               disappears.

  CORRECTNESS  where it does resolve, does it agree with the derived labels?

The second is the important one. A fast path that is cheap and wrong is worse
than no fast path, because it removes the model call that would have caught
the error. Disagreement here is a bug in Tier 1 or in the labels; either way
it must be looked at before any GPU time is spent.

Note the labels used for scoring (data/vocab.json) and the names Tier 1 matches
against (data/names.json) are independent derivations of the same fact -- one
transcribed by hand from the corpus, one fetched from the price provider. Their
agreement is evidence. It is not circular, because neither was built from the
other.

    python scripts/eval_cascade.py
    python scripts/eval_cascade.py --show-declines 20
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workflow", "portfolio", "agents"))

from agentops import parser_eval                                   # noqa: E402
from parser_cascade import NameIndex, parse_holdings, parse_lookback  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=os.path.join(ROOT, "data", "queries.json"))
    ap.add_argument("--names", default=os.path.join(ROOT, "data", "names.json"))
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "data", "snapshot"))
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data", "vocab.json"))
    ap.add_argument("--query-set", default=None,
                    help="restrict to a frozen query set, e.g. "
                         "data/query_sets/systems_100.json")
    ap.add_argument("--show-declines", type=int, default=10)
    ap.add_argument("--show-mismatches", type=int, default=10)
    a = ap.parse_args()

    if not os.path.exists(a.names):
        print(f"no name table at {a.names}\n"
              f"run: python scripts/fetch_names.py   (needs network)",
              file=sys.stderr)
        return 2

    queries = json.load(open(a.queries, encoding="utf-8"))
    if a.query_set:
        spec = json.load(open(a.query_set, encoding="utf-8"))
        ids = set(spec["ids"])
        queries = [q for q in queries if q["id"] in ids]

    index = NameIndex(a.names, a.snapshot)
    aliases = parser_eval.load_vocab(a.vocab)["aliases"]

    print(f"name table: {index.source}")
    print(f"  {len(index.by_name)} indexed forms over {len(index.universe)} "
          f"tickers, of which {len(index.token_aliases)} are single-token "
          f"shortenings")
    print()

    resolved, declined, mismatched = [], [], []
    lookback_wrong = []

    for q in queries:
        text = q["query"]
        lb, confident = parse_lookback(text)
        if not confident:
            declined.append((q, "unreadable_window"))
            continue
        holdings = parse_holdings(text, index)
        if not holdings:
            declined.append((q, "unresolved_holdings"))
            continue

        resolved.append(q)

        exp = parser_eval.expected_parse(text, aliases)
        if exp is not None:
            ok, max_err, _ = parser_eval._weights_ok(exp, holdings)
            if set(exp) != set(holdings) or not ok:
                mismatched.append((q, exp, holdings, max_err))
        if lb != q.get("expected_lookback_days"):
            lookback_wrong.append((q, lb))

    n = len(queries)
    print(f"{'='*70}")
    print(f"TIER 1 on {n} queries")
    print(f"{'='*70}")
    print(f"  resolved without a model call   {len(resolved):>4} "
          f"({100*len(resolved)/n:.1f}%)")
    print(f"  declined, falls through to LLM  {len(declined):>4} "
          f"({100*len(declined)/n:.1f}%)")
    reasons: dict[str, int] = {}
    for _, r in declined:
        reasons[r] = reasons.get(r, 0) + 1
    for r, c in sorted(reasons.items(), key=lambda kv: -kv[1]):
        print(f"      {r:<22} {c}")

    print(f"\n  where resolved, vs derived labels:")
    print(f"      holdings disagree           {len(mismatched)} "
          f"({100*len(mismatched)/max(1,len(resolved)):.1f}% of resolved)")
    print(f"      lookback disagrees          {len(lookback_wrong)} "
          f"({100*len(lookback_wrong)/max(1,len(resolved)):.1f}% of resolved)")

    if mismatched:
        print(f"\n  MISMATCHES (first {a.show_mismatches}) -- fix before running:")
        for q, exp, got, err in mismatched[:a.show_mismatches]:
            print(f"    [{q['id']}] {q['query'][:88]}")
            print(f"        expected {sorted(exp)}")
            print(f"        tier1    {sorted(got)}   max weight err {err:.3f}")

    if lookback_wrong:
        print(f"\n  LOOKBACK DISAGREEMENTS (first {a.show_mismatches}):")
        for q, lb in lookback_wrong[:a.show_mismatches]:
            print(f"    [{q['id']}] shipped={q.get('expected_lookback_days')} "
                  f"tier1={lb}  {q['query'][:70]}")

    if declined and a.show_declines:
        print(f"\n  DECLINED (first {a.show_declines}) -- these still cost an "
              f"LLM call, and that is correct:")
        for q, r in declined[:a.show_declines]:
            print(f"    [{q['id']}] ({r}) {q['query'][:84]}")

    print(f"\nIf coverage is high AND mismatches are zero, A1 removes "
          f"{100*len(resolved)/n:.0f}% of parser calls at no accuracy cost.")
    print("If coverage is high and mismatches are NOT zero, the fast path is "
          "buying speed with correctness -- stop and fix it.")

    # BOTH kinds of disagreement fail this gate.
    #
    # It used to be `return 0 if not mismatched else 1`, which printed lookback
    # disagreements in red and then exited 0 anyway. A wrong lookback is not a
    # lesser error than a wrong ticker set: it silently analyses the right
    # portfolio over the wrong window, which is the same shape of failure --
    # plausible output, different question answered. The script called itself
    # the correctness gate while gating half of what it measured, so 1,000
    # lookback errors and zero holdings errors would have exited PASS.
    failed = len(mismatched) + len(lookback_wrong)
    if failed:
        print(f"\nFAIL: {len(mismatched)} holdings and {len(lookback_wrong)} "
              f"lookback disagreement(s). Do not spend GPU time on this.")
    else:
        print(f"\nPASS: zero holdings and zero lookback disagreements across "
              f"{len(resolved)} resolved queries.")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
