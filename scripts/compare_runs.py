#!/usr/bin/env python3
"""
Are these runs a valid comparison?

WHY THIS EXISTS
---------------
Every headline number in this project is a DIFFERENCE between two runs:

    A1 is 28.4% cheaper than B0
    A2-short is 38.7% cheaper than B0
    A2-short's briefings are non-inferior to A1's

An external audit found that the third of those had almost nothing binding the
two arms together -- the gate compared the scores and checked they came from
the same scorer version, and nothing established that the two runs had used the
same corpus, model or price snapshot. That was fixed inside check_run.

The first two are worse, and nobody had flagged them, because they have no gate
at all. Three directories sit in results/, the README subtracts two numbers,
and the only thing asserting that those runs differ solely in the parser is
that I ran them and remember. Subtraction does not care whether the operands
describe the same experiment.

    python scripts/compare_runs.py results/B0-... results/A1-... --varying parser_kind
    python scripts/compare_runs.py --headline

Exit 0 = the comparison is sound. Exit 1 = it is not, or cannot be established
from the artifacts, which for the archived runs is the honest answer until they
are re-measured with the provenance fields.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import validity                      # noqa: E402

# The comparison the README's headline table makes, named here so it can be
# checked rather than assumed. Order matters only for readability.
#
# rep3, not rep1. The rep1 runs predate query_content_sha256 and can only ever
# answer CANNOT ESTABLISH -- which was the honest verdict while they were the
# only runs, and is now just a stale pointer. The rep3 runs carry the binding
# fields, so this comparison is decidable from the artifacts.
HEADLINE = [
    ("results/B0-offline-n1000-c8-rep3", ()),
    ("results/A1-offline-n1000-c8-rep3", ("parser_kind",)),
    ("results/A2-short-offline-n1000-c8-rep3",
     ("parser_kind", "advisor_style", "advisor_max_tokens",
      "advisor_temperature")),
]


def _json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def compare(paths: list[str], varying: tuple[str, ...]) -> int:
    runs = []
    for p in paths:
        m = _json(os.path.join(p, "manifest.json"))
        if m is None:
            print(f"no manifest.json in {p}", file=sys.stderr)
            return 1
        runs.append((os.path.basename(p.rstrip("/")), m))

    print(f"comparing {len(runs)} runs")
    for label, m in runs:
        print(f"  {label}  stage={m.get('stage')} parser={m.get('parser_kind')} "
              f"advisor={m.get('advisor_style')}")
    if varying:
        print(f"  deliberately varying: {', '.join(varying)}")
    print()

    checks = validity.comparability(runs, varying=varying)
    for c in checks:
        print(c.line())

    failed = [c for c in checks if c.verifiable and not c.ok]
    gaps = [c for c in checks if not c.verifiable]
    print()
    if failed:
        print(f"NOT COMPARABLE: {len(failed)} field(s) differ. Any difference "
              f"between these runs is confounded by them.")
        return 1
    if gaps:
        print(f"CANNOT ESTABLISH: {len(gaps)} binding field(s) are not recorded "
              f"on these runs.\nThey predate the provenance fields, so the "
              f"comparison rests on operator memory rather than on the "
              f"artifacts. Re-measure to close it.")
        return 1
    print("COMPARABLE: these runs differ only in the named treatment.")
    return 0


def repeats(paths: list[str]) -> int:
    """Repeats of ONE configuration -- stricter than a cross-stage comparison.

    A median over three repeats claims they are three samples of the same
    thing. That is a stronger statement than "these two runs differ only in
    the parser", and it needs the SOURCE TREE to match as well: repeats made
    before and after a code change are not repeated measurements, they are one
    measurement of each of two systems.

    This was not hypothetical. The B0 median landed on the one repeat whose
    source hash differed from its neighbours, because a patch went out between
    the two batches -- a change to a reporting constant that cannot affect a
    measurement, which is exactly the kind of difference that gets waved
    through until the habit of waving things through costs something.
    """
    runs = []
    for p in paths:
        m = _json(os.path.join(p, "manifest.json"))
        if m is None:
            print(f"no manifest.json in {p}", file=sys.stderr)
            return 1
        runs.append((os.path.basename(p.rstrip("/")), m))

    print(f"checking {len(runs)} repeats of one configuration")
    checks = validity.comparability(runs, varying=())
    trees = [(lbl, (m or {}).get("source_tree_sha256") or "") for lbl, m in runs]
    present = [t for _, t in trees if t]
    if not present:
        checks.append(validity.Check(
            "repeats::source_tree", False,
            "no repeat records source_tree_sha256", verifiable=False))
    elif len(present) != len(trees) or len(set(present)) != 1:
        checks.append(validity.Check(
            "repeats::source_tree", False,
            "; ".join(f"{lbl}={(t or 'MISSING')[:12]}" for lbl, t in trees)
            + " -- repeats made on different code are not repeated "
              "measurements of one system"))
    else:
        checks.append(validity.Check(
            "repeats::source_tree", True,
            f"all {len(trees)} repeats ran on source tree {present[0][:12]}"))

    for c in checks:
        print(c.line())
    failed = [c for c in checks if c.verifiable and not c.ok]
    gaps = [c for c in checks if not c.verifiable]
    print()
    if failed:
        print(f"NOT A CLEAN REPEAT SET: {len(failed)} field(s) differ. A median "
              f"over these pools samples of different systems.")
        return 1
    if gaps:
        print(f"CANNOT ESTABLISH: {len(gaps)} field(s) unrecorded.")
        return 1
    print("CLEAN REPEAT SET: a median over these is a median of one system.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*")
    ap.add_argument("--repeats", action="store_true",
                    help="treat the runs as repeats of ONE configuration: "
                         "everything must match, including the source tree")
    ap.add_argument("--varying", default="",
                    help="comma-separated fields this comparison intends to "
                         "change (e.g. parser_kind,advisor_style)")
    ap.add_argument("--headline", action="store_true",
                    help="check the comparison the README's table makes")
    a = ap.parse_args()

    if a.headline:
        rc = 0
        base = HEADLINE[0][0]
        for path, varying in HEADLINE[1:]:
            print("=" * 70)
            rc |= compare([os.path.join(ROOT, base), os.path.join(ROOT, path)],
                          tuple(varying))
            print()
        return rc

    if len(a.runs) < 2:
        print("give at least two run directories, or --headline", file=sys.stderr)
        return 2
    if a.repeats:
        return repeats(a.runs)
    varying = tuple(x.strip() for x in a.varying.split(",") if x.strip())
    return compare(a.runs, varying)


if __name__ == "__main__":
    raise SystemExit(main())
