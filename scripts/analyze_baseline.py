#!/usr/bin/env python3
"""
Offline analysis of a completed run. No GPU, no network, no model.

Answers the three questions that decide what to measure next, all from
artifacts already on disk:

  1. PHRASING          Which phrasing classes fail, using Canyon Code's own
                       `phrasing` label rather than substring detection. An
                       earlier version of this analysis detected equal-weight
                       queries by searching for "equal split" and friends, got
                       a denominator of 154 where the shipped label says ~201,
                       and reported a concentration ratio ~40% too high. The
                       shipped-versus-derived rule this project applies to
                       every other metric applies here too.

  2. SEMANTIC SUCCESS  How many queries COMPLETED while analysing the wrong
                       portfolio. "34 failures" flatters the baseline: a
                       SnapshotMiss is the system saying it does not know,
                       which is safe. A successful analysis of a portfolio
                       nobody asked for is the system being confidently wrong,
                       with correct-looking numbers attached. Only an
                       independent parser score can see these at all.

  3. ADVISOR LENGTH    What the Advisor actually generates against its
                       max_tokens budget. If completions stop far short of the
                       cap, then an A2 cap sweep is four identical rows and a
                       wasted GPU booking. If a tail sits exactly at the cap,
                       the baseline is already truncating briefings mid
                       sentence -- a latent quality defect it never reported.

Also reports whether the trace captures `finish_reason`, because truncation
cannot be measured without it and A2 needs that gate before it runs.

    python scripts/analyze_baseline.py results/B0-offline-n1000-c8-rep1
    python scripts/analyze_baseline.py results/B0-offline-n1000-c8-rep1 \
        --compare results/A1-offline-n1000-c8-rep1
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import parser_eval  # noqa: E402


def jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(xs: list[float], q: float):
    if not xs:
        return None
    s = sorted(xs)
    return s[min(len(s) - 1, int(round(q * (len(s) - 1))))]


def is_warmup(qid) -> bool:
    return isinstance(qid, str) and qid.startswith("warmup-")


# ---------------------------------------------------------------- 1. phrasing
def section_phrasing(results: list[dict], failures: list[dict]) -> None:
    """Failure rate by Canyon Code's shipped phrasing class.

    Both results and failures carry the shipped `label`, so the corpus share
    and the failure share come from the same source and cover exactly the
    attempted set -- no re-reading queries.json, no substring guessing.
    """
    print("\n" + "=" * 74)
    print("1. FAILURES BY SHIPPED PHRASING CLASS")
    print("=" * 74)

    attempted = [(r, "ok") for r in results] + [(f, "failed") for f in failures]
    by_class: dict[str, dict[str, int]] = collections.defaultdict(
        lambda: {"n": 0, "failed": 0, "parse_error": 0, "snapshot_miss": 0})

    unlabelled = 0
    for row, outcome in attempted:
        ph = (row.get("label") or {}).get("phrasing")
        if ph is None:
            unlabelled += 1
            ph = "(no shipped label)"
        e = by_class[ph]
        e["n"] += 1
        if outcome == "failed":
            e["failed"] += 1
            err = row.get("error", "")
            if "ParseError" in err:
                e["parse_error"] += 1
            elif "SnapshotMiss" in err:
                e["snapshot_miss"] += 1

    total = len(attempted)
    total_failed = sum(e["failed"] for e in by_class.values())
    print(f"  {'phrasing':<18}{'n':>6}{'share':>8}{'failed':>8}{'rate':>8}"
          f"{'parse':>7}{'ticker':>8}")
    for ph, e in sorted(by_class.items(), key=lambda kv: -kv[1]["failed"]):
        print(f"  {ph:<18}{e['n']:>6}{100*e['n']/total:>7.1f}%{e['failed']:>8}"
              f"{100*e['failed']/max(1,e['n']):>7.2f}%"
              f"{e['parse_error']:>7}{e['snapshot_miss']:>8}")

    if unlabelled:
        print(f"  NOTE {unlabelled} attempted queries carried no shipped "
              f"phrasing label")

    # Relative risk for the class carrying the most parse errors, stated with
    # its own event count so nobody reads six events as an established effect.
    pe_by_class = {ph: e["parse_error"] for ph, e in by_class.items()}
    total_pe = sum(pe_by_class.values())
    if total_pe:
        top = max(pe_by_class, key=lambda k: pe_by_class[k])
        n_top = by_class[top]["n"]
        pe_top = pe_by_class[top]
        rate_top = pe_top / max(1, n_top)
        rate_rest = (total_pe - pe_top) / max(1, total - n_top)
        print(f"\n  parse errors concentrate in '{top}': {pe_top}/{total_pe} "
              f"of all parse errors, from {100*n_top/total:.1f}% of the corpus")
        print(f"    rate in  '{top}': {100*rate_top:.2f}%")
        print(f"    rate elsewhere:   {100*rate_rest:.2f}%")
        if rate_rest > 0:
            print(f"    relative risk:    {rate_top/rate_rest:.1f}x")
        print(f"    on {total_pe} events total -- a signal worth a targeted "
              f"test, not an established effect")
    print(f"\n  total failed {total_failed}/{total} = "
          f"{100*total_failed/max(1,total):.2f}%")


# ------------------------------------------------- 2. operational vs semantic
def section_semantic(results: list[dict], failures: list[dict],
                     aliases: dict) -> None:
    """Queries that succeeded while analysing the wrong portfolio.

    This is the number a success rate cannot show. It exists only because
    parser correctness is scored independently of whether the workflow
    returned something.
    """
    print("\n" + "=" * 74)
    print("2. OPERATIONAL SUCCESS vs SEMANTIC CORRECTNESS")
    print("=" * 74)

    attempted = len(results) + len(failures)
    silent_ticker, silent_weights, silent_count, unscorable = [], [], [], 0

    for r in results:
        q = r.get("query", "")
        parsed = r.get("parsed") or {}
        got = {k: float(v) for k, v in (parsed.get("holdings") or {}).items()}
        label = r.get("label") or {}

        exp = parser_eval.expected_parse(q, aliases)
        if exp is None:
            unscorable += 1
        else:
            if set(exp) != set(got):
                silent_ticker.append(r)
            else:
                ok, _mx, _l1 = parser_eval._weights_ok(exp, got)
                if not ok:
                    silent_weights.append(r)
        if label.get("n_holdings") is not None and \
                len(got) != label["n_holdings"]:
            silent_count.append(r)

    wrong = {id(r) for r in silent_ticker + silent_weights + silent_count}
    n_wrong = len(wrong)
    op = len(results)
    sem = op - n_wrong

    print(f"  attempted                       {attempted}")
    print(f"  operational success             {op}  "
          f"({100*op/max(1,attempted):.1f}%)   the workflow returned something")
    print(f"  SEMANTIC success                {sem}  "
          f"({100*sem/max(1,attempted):.1f}%)   ...and it was the right portfolio")
    print(f"  silent wrong answers            {n_wrong}")
    print(f"      wrong ticker set            {len(silent_ticker)}")
    print(f"      right tickers, wrong weights{len(silent_weights):>4}")
    print(f"      wrong holding count         {len(silent_count)}")
    print(f"  loud failures                   {len(failures)}")
    if unscorable:
        print(f"  not scorable by alias map       {unscorable}")

    print(f"\n  A SnapshotMiss is the system saying it does not know, which is")
    print(f"  safe. These {n_wrong} are the system being confidently wrong, with")
    print(f"  correct-looking numbers attached, and no downstream check that")
    print(f"  could catch them.")

    for title, rows in (("wrong ticker set", silent_ticker),
                        ("wrong weights", silent_weights)):
        if not rows:
            continue
        print(f"\n  examples -- {title}:")
        for r in rows[:6]:
            exp = parser_eval.expected_parse(r.get("query", ""), aliases) or {}
            got = (r.get("parsed") or {}).get("holdings") or {}
            print(f"    [{r.get('query_id')}] {r.get('query','')[:74]}")
            print(f"        asked for {sorted(exp)}")
            print(f"        analysed  {sorted(got)}")


def section_entity_rate(results: list[dict], failures: list[dict],
                        aliases: dict) -> None:
    """Per-company error rate.

    A single corpus-wide failure rate hides the shape of the problem. If the
    errors sit on two of thirty companies, that is a different engineering
    problem -- and a different fix -- than a uniform rate.
    """
    print("\n" + "=" * 74)
    print("3. ERROR RATE BY COMPANY (mentions vs appearances in a bad parse)")
    print("=" * 74)

    surfaces = sorted(aliases, key=len, reverse=True)

    def mentioned(text: str) -> set[str]:
        hits = set()
        for s in surfaces:
            if re.search(rf"(?<![A-Za-z0-9]){re.escape(s)}(?![A-Za-z0-9])",
                         text):
                hits.add(aliases[s])
        return hits

    mentions: collections.Counter = collections.Counter()
    bad: collections.Counter = collections.Counter()

    for r in results:
        q = r.get("query", "")
        ms = mentioned(q)
        mentions.update(ms)
        exp = parser_eval.expected_parse(q, aliases)
        got = set(((r.get("parsed") or {}).get("holdings") or {}))
        if exp is not None and set(exp) != got:
            bad.update(ms)
    for f in failures:
        q = f.get("query", "")
        ms = mentioned(q)
        mentions.update(ms)
        bad.update(ms)

    print(f"  {'ticker':<8}{'mentions':>9}{'in bad parse':>14}{'rate':>8}")
    rows = sorted(mentions, key=lambda t: -bad[t] / max(1, mentions[t]))
    for t in rows[:10]:
        if not bad[t]:
            continue
        print(f"  {t:<8}{mentions[t]:>9}{bad[t]:>14}"
              f"{100*bad[t]/max(1,mentions[t]):>7.1f}%")
    clean = [t for t in mentions if not bad[t]]
    print(f"\n  {len(clean)} of {len(mentions)} companies appear in zero bad "
          f"parses")


# -------------------------------------------------------- 4. advisor lengths
def section_advisor(trace: list[dict], manifest: dict) -> None:
    """Does the Advisor's max_tokens budget bind, and is anything truncated?"""
    print("\n" + "=" * 74)
    print("4. ADVISOR COMPLETION LENGTH vs max_tokens  (decides whether A2 is")
    print("   an experiment or four identical rows)")
    print("=" * 74)

    cap = manifest.get("advisor_max_tokens") or 0
    rows = [r for r in trace
            if r.get("kind") == "llm" and r.get("agent") == "AdvisorAgent"
            and not r.get("error") and not is_warmup(r.get("query_id"))]
    lens = [r.get("completion_tokens", 0) for r in rows]
    if not lens:
        print("  no Advisor calls in this trace")
        return

    print(f"  max_tokens budget   {cap}")
    print(f"  calls               {len(lens)}")
    print(f"  mean                {sum(lens)/len(lens):.1f}")
    for q in (0.50, 0.90, 0.95, 0.99):
        print(f"  p{int(q*100):<18} {pct(lens, q)}")
    print(f"  max                 {max(lens)}")

    at_cap = sum(1 for x in lens if cap and x >= cap)
    print(f"\n  at or above the cap {at_cap}  "
          f"({100*at_cap/len(lens):.2f}%)")
    if at_cap:
        print("  -> the baseline is ALREADY truncating briefings. That is a")
        print("     latent quality defect the baseline never reported, and it")
        print("     is a finding independent of A2.")

    print("\n  candidate caps, and whether each would bind:")
    for c in (400, 256, 192, 160, 128, 96):
        n = sum(1 for x in lens if x > c)
        verdict = "NO-OP" if n == 0 else f"affects {n} calls ({100*n/len(lens):.1f}%)"
        print(f"    {c:>4}   {verdict}")
    print("\n  A cap that affects zero calls is not a treatment. Only caps that")
    print("  bind belong in the sweep; the rest would produce identical rows.")

    fr = collections.Counter(r.get("finish_reason") for r in rows)
    print(f"\n  finish_reason: {dict(fr)}")
    n_len = fr.get("length", 0)
    if n_len:
        print(f"  -> {n_len} completions ({100*n_len/len(lens):.2f}%) stopped "
              f"because they hit the budget, not because they were finished.")
        print("     Those briefings are cut mid-sentence. This is a baseline")
        print("     quality defect, and it is also A2's ready-made quality gate:")
        print("     the cap sweep can be scored on truncation rate directly")
        print("     rather than on a proxy.")
    elif None in fr:
        print("  -> finish_reason NOT captured for some calls; truncation would")
        print("     have to be inferred from length == cap. Fix before A2.")
    else:
        print("  -> every completion stopped on its own. No truncation in this")
        print("     run, and truncation rate is available as A2's quality gate.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--compare", default=None,
                    help="a second run dir; section 4 is printed for it too")
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data", "vocab.json"))
    a = ap.parse_args()

    d = a.run_dir
    manifest = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    results = jsonl(os.path.join(d, "results.jsonl"))
    failures = jsonl(os.path.join(d, "failures.jsonl"))
    trace = jsonl(os.path.join(d, "trace.jsonl"))
    aliases = parser_eval.load_vocab(a.vocab)["aliases"]

    print(f"\n{'#'*74}\n# {manifest['stage']}  {manifest['run_id']}\n"
          f"# parser={manifest.get('parser_kind','llm')}  "
          f"n={manifest['n_queries']}  C={manifest.get('concurrency')}\n{'#'*74}")

    section_phrasing(results, failures)
    section_semantic(results, failures, aliases)
    section_entity_rate(results, failures, aliases)
    section_advisor(trace, manifest)

    if a.compare:
        m2 = json.load(open(os.path.join(a.compare, "manifest.json"),
                            encoding="utf-8"))
        t2 = jsonl(os.path.join(a.compare, "trace.jsonl"))
        print(f"\n\n{'#'*74}\n# COMPARISON: {m2['stage']}  {m2['run_id']}\n{'#'*74}")
        section_advisor(t2, m2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
