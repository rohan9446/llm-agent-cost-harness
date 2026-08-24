#!/usr/bin/env python3
"""
A1 robustness: does the cascade ROUTE correctly on inputs it has never seen?

The obvious criticism of A1 is "you regexed a template-generated benchmark."
It is a fair criticism and 100% coverage on the supplied corpus cannot answer
it, because Tier 1 was debugged against that corpus. This set answers a
narrower question that a rules-first design actually has to get right:

    when Tier 1 cannot handle an input, does it DECLINE -- or does it
    confidently produce a wrong parse?

Those two failures are not symmetric, and the scoring reflects that:

  FALSE ACCEPT   Tier 1 resolved something it should have refused, or resolved
                 it wrongly. SERIOUS. There is no model call left to catch it,
                 so the error reaches the user as a confident answer to the
                 wrong question -- the same shape as B0's Adobe->AAPL silent
                 errors, one layer up.

  FALSE DECLINE  Tier 1 refused something it could have handled. BENIGN. It
                 costs one LLM call, which is exactly B0's price. The cascade
                 degrades to the baseline rather than to a wrong answer.

So a cascade that declines too often is merely less profitable. One that
accepts too readily is unsafe. The headline metric is the false-accept rate.

HOW THE SET IS BUILT, AND WHY THAT MATTERS
------------------------------------------
Phrasing is NOT authored here. Groups A and B take verbatim queries from
queries.json and substitute only the entity, so sentence shape stays Canyon
Code's and the single manipulated variable is the company. Groups C-E are
authored, because no corpus query contains a word-boundary trap -- those are
labelled `authored` in the file so a reader can weigh them separately.

The out-of-universe companies are real issuers absent from the frozen
snapshot. The trap words are ordinary English or real company names that
happen to contain an indexed alias as a substring.

    python scripts/perturbed_set.py --build
    python scripts/perturbed_set.py --score
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "workflow", "portfolio", "agents"))

OUT = os.path.join(ROOT, "data", "query_sets", "perturbed.json")

# Real issuers, none of them in the frozen 30. Tier 1 must decline these: the
# system cannot price them, so the honest response is to hand off, not guess.
OUT_OF_UNIVERSE = [
    "Ford", "Boeing", "Starbucks", "Nike", "Target", "IBM", "Uber",
    "Airbnb", "Shopify", "Palantir", "Snowflake", "Rivian", "Moderna",
    "Caterpillar", "Goldman Sachs", "Verizon", "Comcast", "Lowe's",
    "Delta Air Lines", "Union Pacific",
]

# Words that CONTAIN an indexed alias as a substring. Every one of these must
# decline. They are the direct probe for the bug that shipped 100% coverage
# while matching "intel" inside "intelligent".
TRAPS = [
    ("How did intelligent systems exposure do over the last month?", "intel/INTC"),
    ("What is the metadata risk in my book over the last month?", "meta/META"),
    ("Tell me about metaphysical uncertainty over the past year.", "meta/META"),
    ("Compare amazonian rainforest exposure over the last month.", "amazon/AMZN"),
    ("Is oracle-like forecasting reliable over the last 6 months?", "oracle/ORCL"),
    ("Describe the visage of my book over the past year.", "visa/V"),
    ("Summarise the Walton family estate over the last 5 years.", "walt/DIS"),
    ("Price pineapple futures over the past 3 months.", "apple/AAPL"),
    ("Show microscopic positions over the last month.", "micro-/MSFT"),
    ("Explain the chevrons on the risk chart over the past year.", "chevron/CVX"),
]

# Real companies whose names contain an indexed alias as a WHOLE WORD. These
# are the hard cases: word boundaries do not save us, because the token really
# is present. Included precisely because they are expected to fail.
NEAR_MISS_ISSUERS = [
    ("Weight me 60% Morgan Stanley and 40% Boeing over the past year.",
     "'Morgan' is a whole word but the issuer is not JPMorgan"),
    ("Value Adobe brick suppliers, evenly weighted, over the last month.",
     "'Adobe' is an ordinary English noun as well as a ticker"),
    ("Break down Delta Air Lines and Target over the past year.",
     "'Delta' is a common financial term"),
]

# Ambiguity Tier 1 must refuse rather than resolve. In-universe entities, so
# the only reason to decline is the ambiguity itself.
AMBIGUOUS = [
    ("Weight me 60% in Visa and Pfizer over the past year.",
     "one percentage, two holdings"),
    ("Value Apple 50%, Microsoft 30% and Netflix over the past year.",
     "two percentages, three holdings"),
    ("Break down Apple and Apple Inc. over the past year.",
     "same issuer named twice"),
    ("Weight me 120% Visa and 30% Pfizer over the past year.",
     "weights exceed 100%"),
]

UNREADABLE_WINDOW = [
    ("How has Visa done since the IPO?", "since the IPO"),
    ("Break down Apple and Intel year-to-date.", "year-to-date"),
    ("Value Netflix and Pfizer since the pandemic.", "since the pandemic"),
    ("Show Visa and Adobe over the trailing period.",
     "no quantity or unit"),
]


def _entity_spans(query: str, aliases: dict) -> list[tuple[int, int, str]]:
    """Where the corpus surface forms sit in a query, longest first."""
    spans = []
    for surface in sorted(aliases, key=len, reverse=True):
        for m in re.finditer(
                rf"(?<![A-Za-z0-9]){re.escape(surface)}(?![A-Za-z0-9])", query):
            if not any(s <= m.start() < e for s, e, _ in spans):
                spans.append((m.start(), m.end(), surface))
    return sorted(spans)


def build(n_swapped: int = 40, seed: int = 20260823) -> dict:
    rng = random.Random(seed)
    corpus = json.load(open(os.path.join(ROOT, "data", "queries.json"),
                            encoding="utf-8"))
    aliases = json.load(open(os.path.join(ROOT, "data", "vocab.json"),
                             encoding="utf-8"))["aliases"]

    items: list[dict] = []

    # GROUP A -- verbatim corpus phrasing, every entity swapped out of universe.
    # Tier 1 must decline: nothing here is priceable.
    pool = [q for q in corpus if len(_entity_spans(q["query"], aliases)) >= 2]
    rng.shuffle(pool)
    for q in pool[:n_swapped]:
        text, spans = q["query"], _entity_spans(q["query"], aliases)
        picks = rng.sample(OUT_OF_UNIVERSE, len(spans))
        for (s, e, _), repl in sorted(zip(spans, picks), reverse=True):
            text = text[:s] + repl + text[e:]
        items.append({
            "id": f"P-A{len(items):03d}", "query": text, "group": "A",
            "phrasing_source": "corpus verbatim, entities substituted",
            "expect": "decline",
            "why": "every holding is outside the priceable universe",
        })

    # GROUP B -- corpus phrasing, ONE entity swapped out of universe.
    # A partly-priceable request is still not answerable as asked.
    for q in pool[n_swapped:n_swapped + 15]:
        text, spans = q["query"], _entity_spans(q["query"], aliases)
        s, e, _ = spans[rng.randrange(len(spans))]
        text = text[:s] + rng.choice(OUT_OF_UNIVERSE) + text[e:]
        items.append({
            "id": f"P-B{len(items):03d}", "query": text, "group": "B",
            "phrasing_source": "corpus verbatim, one entity substituted",
            "expect": "decline",
            "why": "one holding is outside the priceable universe",
        })

    for text, why in TRAPS:
        items.append({"id": f"P-C{len(items):03d}", "query": text, "group": "C",
                      "phrasing_source": "authored", "expect": "decline",
                      "why": f"substring trap: {why}"})
    for text, why in NEAR_MISS_ISSUERS:
        items.append({"id": f"P-D{len(items):03d}", "query": text, "group": "D",
                      "phrasing_source": "authored", "expect": "decline",
                      "why": f"whole-word near miss: {why}"})
    for text, why in AMBIGUOUS:
        items.append({"id": f"P-E{len(items):03d}", "query": text, "group": "E",
                      "phrasing_source": "authored", "expect": "decline",
                      "why": f"ambiguous: {why}"})
    for text, why in UNREADABLE_WINDOW:
        items.append({"id": f"P-F{len(items):03d}", "query": text, "group": "F",
                      "phrasing_source": "authored", "expect": "decline",
                      "why": f"unreadable window: {why}"})

    # GROUP G -- the control. In-universe, corpus phrasing, untouched. If Tier 1
    # starts declining these, the set has measured a regression rather than
    # robustness, and every other number here is suspect.
    for q in pool[n_swapped + 15:n_swapped + 30]:
        items.append({"id": f"P-G{len(items):03d}", "query": q["query"],
                      "group": "G", "phrasing_source": "corpus verbatim",
                      "expect": "accept",
                      "why": "control: unmodified corpus query, must still resolve"})

    payload = {
        "_purpose": "A1 routing robustness. False accepts are the metric that "
                    "matters; false declines only cost a model call.",
        "_groups": {
            "A": "corpus phrasing, ALL entities out of universe -> decline",
            "B": "corpus phrasing, ONE entity out of universe -> decline",
            "C": "authored substring traps (intel in intelligent) -> decline",
            "D": "authored whole-word near misses (Morgan Stanley) -> decline",
            "E": "authored ambiguity (partial weights) -> decline",
            "F": "authored unreadable time windows -> decline",
            "G": "unmodified corpus queries -> accept (control)",
        },
        "seed": seed,
        "n": len(items),
        "items": items,
    }
    payload["sha256"] = hashlib.sha256(
        json.dumps([i["query"] for i in items], separators=(",", ":")).encode()
    ).hexdigest()
    return payload


def score(path: str, names: str, snapshot: str) -> int:
    from parser_cascade import NameIndex, parse_holdings, parse_lookback

    payload = json.load(open(path, encoding="utf-8"))
    index = NameIndex(names, snapshot)

    by_group: dict[str, dict[str, int]] = {}
    false_accepts, false_declines = [], []

    for it in payload["items"]:
        lb, confident = parse_lookback(it["query"])
        holdings = parse_holdings(it["query"], index) if confident else None
        accepted = bool(confident and holdings)
        want = it["expect"]
        g = by_group.setdefault(it["group"], {"n": 0, "fa": 0, "fd": 0, "ok": 0})
        g["n"] += 1
        if accepted and want == "decline":
            g["fa"] += 1
            false_accepts.append((it, sorted(holdings or {})))
        elif not accepted and want == "accept":
            g["fd"] += 1
            false_declines.append((it, None))
        else:
            g["ok"] += 1

    n = payload["n"]
    fa = sum(g["fa"] for g in by_group.values())
    fd = sum(g["fd"] for g in by_group.values())

    print(f"\n{'='*74}\nA1 ROUTING ON {n} PERTURBED QUERIES  "
          f"(sha256 {payload['sha256'][:16]})\n{'='*74}")
    print(f"  {'group':<7}{'n':>5}{'correct':>9}{'FALSE ACC':>11}{'false dec':>11}"
          f"   what it probes")
    for gname in sorted(by_group):
        g = by_group[gname]
        print(f"  {gname:<7}{g['n']:>5}{g['ok']:>9}{g['fa']:>11}{g['fd']:>11}"
              f"   {payload['_groups'][gname][:38]}")
    print(f"\n  FALSE ACCEPT rate  {fa}/{n} = {100*fa/n:.1f}%   "
          f"<-- the number that matters")
    print(f"  false decline rate {fd}/{n} = {100*fd/n:.1f}%   "
          f"(costs one LLM call each; degrades to B0, not to a wrong answer)")

    if false_accepts:
        print(f"\n  FALSE ACCEPTS -- Tier 1 answered where it should have "
              f"handed off:")
        for it, got in false_accepts:
            print(f"    [{it['id']}] {it['query'][:70]}")
            print(f"        resolved {got}   ({it['why']})")
    if false_declines:
        print(f"\n  false declines -- these fall through to the LLM, which is "
              f"correct behaviour, just not free:")
        for it, _ in false_declines[:8]:
            print(f"    [{it['id']}] {it['query'][:70]}")

    print(f"\n  Read this as a SAFETY result, not a coverage one. A cascade that "
          f"declines\n  too often is less profitable; one that accepts too "
          f"readily is unsafe.")

    # The inputs stay private; the SCORES are the finding, so they get written
    # where a reader without the corpus can see them. Previously this printed
    # to a terminal and the published summary carried only the composition --
    # which meant the robustness claim in the README had no artifact behind it.
    summary_path = os.path.join(os.path.dirname(path), "perturbed.summary.json")
    existing = {}
    if os.path.exists(summary_path):
        with open(summary_path, encoding="utf-8") as fh:
            existing = json.load(fh)
    existing.update({
        "n": n,
        "sha256": payload["sha256"],
        "groups": payload.get("_groups"),
        "composition": {g: by_group[g]["n"] for g in sorted(by_group)},
        "scores": {
            "_what": ("routing safety, not coverage. A false ACCEPT is Tier 1 "
                      "answering a query it should have handed to the model -- "
                      "its cost is a wrong answer delivered confidently, and it "
                      "is unbounded. A false DECLINE costs one LLM call and "
                      "degrades to the B0 path. The asymmetry is why these are "
                      "reported separately rather than as one accuracy."),
            "false_accepts": fa,
            "false_accept_rate": fa / n,
            "false_declines": fd,
            "false_decline_rate": fd / n,
            "by_group": {g: dict(by_group[g]) for g in sorted(by_group)},
        },
        "_inputs": ("not published -- group G is verbatim corpus text and "
                    "groups A and B keep corpus sentence structure with "
                    "entities substituted. Rebuild with: "
                    "python scripts/perturbed_set.py --build"),
    })
    with open(summary_path, "w", encoding="utf-8") as fh:
        json.dump(existing, fh, indent=2)
    print(f"\n  wrote {os.path.relpath(summary_path, ROOT)} "
          f"(scores only; the queries stay local)")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--names", default=os.path.join(ROOT, "data", "names.json"))
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "data", "snapshot"))
    a = ap.parse_args()

    if a.build:
        payload = build()
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        with open(a.out, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        print(f"wrote {a.out}  ({payload['n']} queries, "
              f"sha256 {payload['sha256'][:16]})")
        for g, desc in payload["_groups"].items():
            k = sum(1 for i in payload["items"] if i["group"] == g)
            print(f"  {g}  {k:>3}  {desc}")
    if a.score:
        # redact_artifacts.py moves the runnable set to a gitignored name, so
        # --score after a redaction pass would otherwise fail on a file the
        # policy deliberately removed.
        path = a.out
        if not os.path.exists(path):
            local = path.replace(".json", ".local.json")
            if os.path.exists(local):
                print(f"using local copy {os.path.relpath(local, ROOT)}")
                path = local
            else:
                raise SystemExit(
                    f"no perturbed set at {path} or its .local.json -- "
                    f"build one first: python scripts/perturbed_set.py --build")
        return score(path, a.names, a.snapshot)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
