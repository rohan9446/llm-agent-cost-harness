#!/usr/bin/env python3
"""
Remove supplied query text from published artifacts.

WHY THIS EXISTS
---------------
The .gitignore in this repository states that Canyon Code's query corpus is not
ours to republish, and excludes data/queries.json and results.jsonl on exactly
that basis. It then failed to exclude two artifacts carrying the same text:

  failures.jsonl        run_bench records the full query on every failure row,
                        because that is what makes a failure debuggable locally
  report.json           reliability.*_failure_detail embeds the query so the
                        printed report can show what broke

A third artifact leaked more directly: data/query_sets/perturbed.json contains
a control group of unmodified corpus queries, and two further groups built by
substituting entities into corpus sentences -- so the PHRASING is Canyon
Code's even where the companies are not.

The policy and the artifacts contradicted each other inside the same commit.
This script resolves that in favour of the policy.

WHAT IS KEPT
------------
Everything needed to understand a failure except the sentence itself:

    query_id        which query, so it can be looked up in a private copy
    error           the exception and its message
    failure_class   workflow vs infrastructure
    parsed          what the parser produced -- the actual finding
    label           shipped metadata (n_holdings, phrasing, lookback)

A reader with authorised access to queries.json can reconstruct the full
picture from query_id. A reader without it still learns the error taxonomy,
which is the part that carries the result.

    python scripts/redact_artifacts.py --check     # report, change nothing
    python scripts/redact_artifacts.py --apply
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)

# Keys whose values ARE supplied query text.
QUERY_KEYS = ("query", "queries", "query_text")

# ...and the harder case: query text embedded INSIDE another string.
#
# ParseError used to interpolate the query into its message, so the sentence
# travelled inside `error` -- a field no key-based redaction would ever touch.
# The first version of this script scrubbed keys and declared the artifacts
# clean while every ParseError row still carried its query. parser_agent.py no
# longer embeds it, but artifacts already written do, so they need scrubbing
# too.
#
# The rule is blunt on purpose: any single-quoted span of 20+ characters inside
# an error message is replaced. That over-redacts the model's own output, which
# is a fair trade -- an error message is not where a corpus should be
# recoverable from, and the error CLASS is what carries the finding.
_QUOTED_SPAN = re.compile(r"'[^']{20,}'")


def _scrub_text(s: str) -> str:
    return _QUOTED_SPAN.sub("'<redacted>'", s)


def _strip(obj):
    """Drop query-text keys AND scrub quoted spans out of every string."""
    if isinstance(obj, dict):
        out = {}
        for k, v in obj.items():
            if k in QUERY_KEYS and isinstance(v, str):
                out[k + "_redacted"] = True
                continue
            out[k] = _strip(v)
        return out
    if isinstance(obj, list):
        return [_strip(x) for x in obj]
    if isinstance(obj, str):
        return _scrub_text(obj)
    return obj


def _count_query_strings(obj) -> int:
    """Query text present as a value, or embedded in any other string."""
    n = 0
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in QUERY_KEYS and isinstance(v, str) and v.strip():
                n += 1
            else:
                n += _count_query_strings(v)
    elif isinstance(obj, list):
        for x in obj:
            n += _count_query_strings(x)
    elif isinstance(obj, str):
        n += len(_QUOTED_SPAN.findall(obj))
    return n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true",
                    help="rewrite the files; without it, only report")
    ap.add_argument("--check", action="store_true")
    a = ap.parse_args()
    apply = a.apply and not a.check

    total = 0
    touched = []

    # ---- report.json: reliability.*_failure_detail[].query ------------------
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "report.json"))):
        with open(p, encoding="utf-8") as fh:
            doc = json.load(fh)
        n = _count_query_strings(doc)
        if not n:
            continue
        total += n
        touched.append((p, n))
        if apply:
            doc = _strip(doc)
            doc.setdefault("_redaction", (
                "Supplied query text removed. Failures are identified by "
                "query_id; the error class, the parse and the shipped labels "
                "are retained. See scripts/redact_artifacts.py."))
            with open(p, "w", encoding="utf-8") as fh:
                json.dump(doc, fh, indent=2, default=str)

    # ---- failures.jsonl: one query per row ---------------------------------
    for p in sorted(glob.glob(os.path.join(ROOT, "results", "*", "failures.jsonl"))):
        rows = []
        n = 0
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                n += _count_query_strings(row)
                rows.append(_strip(row))
        if not n:
            continue
        total += n
        touched.append((p, n))
        if apply:
            with open(p, "w", encoding="utf-8") as fh:
                for row in rows:
                    fh.write(json.dumps(row, default=str) + "\n")

    # ---- absolute home paths in EVERY published artifact --------------------
    #
    # /home/<user>/... is not a secret and the audit rightly only warns about
    # it. It is scrubbed anyway, because a warning that fires on every run is a
    # warning nobody reads -- and the next path to appear there might not be
    # generic. What matters is the CONFIGURATION, which survives the
    # substitution intact.
    #
    # THIS SCANNED manifest.json ONLY, ON ITS FIRST OUTING.
    # The manifests were scrubbed, the audit re-run, and eight report.json
    # files still carried the home directory inside SnapshotMiss error strings
    # -- the same shape as the query leak that started this whole exercise: a
    # path in a key I thought of, and the same path inside a message I did not.
    # A glob over every published artifact costs nothing and does not require
    # me to have guessed right about where paths live.
    home = os.path.expanduser("~")
    published = sorted(
        p for pat in ("*.json", "*.jsonl")
        for p in glob.glob(os.path.join(ROOT, "results", "*", pat)))
    n_paths = 0
    for p in published:
        with open(p, encoding="utf-8") as fh:
            raw = fh.read()
        if home not in raw:
            continue
        c = raw.count(home)
        n_paths += c
        touched.append((p, c))
        if apply:
            with open(p, "w", encoding="utf-8") as fh:
                fh.write(raw.replace(home, "~"))
    total += n_paths

    # ---- perturbed.json: the whole file is corpus-derived -------------------
    #
    # Not redactable. Group G is verbatim corpus text, and groups A and B keep
    # corpus SENTENCE STRUCTURE with entities swapped -- stripping the queries
    # would leave a file of ids and labels that cannot be re-run, which is
    # worse than not publishing it. The right split is to publish the SCORES
    # and keep the INPUTS private, so the robustness result stands without the
    # corpus travelling with it.
    pert = os.path.join(ROOT, "data", "query_sets", "perturbed.json")
    if os.path.exists(pert):
        with open(pert, encoding="utf-8") as fh:
            doc = json.load(fh)
        n = _count_query_strings(doc)
        total += n
        touched.append((pert, n))
        if apply:
            # Keep a runnable copy locally under a gitignored name.
            local = os.path.join(ROOT, "data", "query_sets", "perturbed.local.json")
            os.replace(pert, local)
            summary = {
                "_note": (
                    "The perturbed robustness set is NOT published. Groups A "
                    "and B reuse Canyon Code's sentence structure with "
                    "entities substituted, and group G is verbatim corpus "
                    "text, so the file cannot be redacted without becoming "
                    "unrunnable. The composition and the scores are published "
                    "here; the inputs stay private. Rebuild an equivalent set "
                    "with: python scripts/perturbed_set.py --build"),
                "n": doc.get("n"),
                "sha256": doc.get("sha256"),
                "groups": doc.get("_groups"),
                "composition": {
                    g: sum(1 for i in doc.get("items", []) if i.get("group") == g)
                    for g in sorted({i.get("group") for i in doc.get("items", [])})
                },
                "local_copy": "data/query_sets/perturbed.local.json (gitignored)",
            }
            with open(os.path.join(ROOT, "data", "query_sets",
                                   "perturbed.summary.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)

    print(f"{'REDACTED' if apply else 'WOULD REDACT'}: "
          f"{total} query strings and home paths across {len(touched)} files")
    for p, n in touched:
        print(f"  {n:>4}  {os.path.relpath(p, ROOT)}")
    if not apply and total:
        print("\nre-run with --apply to rewrite them")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
