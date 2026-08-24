#!/usr/bin/env python3
"""
Re-score every run's briefings with the CURRENT scorer.

WHY THIS EXISTS
---------------
Two repeats of A2-short, same prompt and same workload, reported 23.6% and 2.8%
of briefings containing an ungrounded figure. Nothing about a brevity
instruction can move a fabrication rate by 8x between repeats of itself. The
arms did not differ; the SCORERS did.

advisor_eval.py had a tokeniser bug -- "3-4" read as 3 and MINUS 4, and the
invented -4 scored as a fabricated figure in every briefing containing a
hyphenated range. It was found and fixed partway through the A2 work. Runs
scored before the fix and runs scored after it were then compared to each other
as though the numbers meant the same thing, and one of them was used to fail a
non-inferiority gate.

This is the same shape as every other finding in this project: the artifact
recorded what was measured and not what measured it. A run's manifest pins the
model, the snapshot, the query set and now the source tree -- and said nothing
about the version of the code that produced its quality scores.

Two fixes, both here:

  1. Every advisor_eval.json now carries _scorer_sha256, so two files can be
     compared only when it is the same number, and the harness can say so.
  2. This script recomputes them all from results.jsonl, which was frozen at
     run time. No GPU, no model, no re-measurement -- the briefings are the
     briefings. Only the ruler changes, and it changes for every run at once.

The original file is preserved as advisor_eval.asrun.json the first time a run
is rescored, because "what the tooling said at the time" is itself evidence and
overwriting it would hide the very inconsistency this script exists to correct.

    python scripts/rescore_advisor.py            # report, change nothing
    python scripts/rescore_advisor.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import advisor_eval                  # noqa: E402


def _jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    return out


def _json(path: str):
    if not os.path.exists(path):
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


def _pct(x) -> str:
    return "—" if x is None else f"{100*x:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true",
                    help="rewrite advisor_eval.json (keeps the original once "
                         "as advisor_eval.asrun.json)")
    a = ap.parse_args()

    scorer = advisor_eval.scorer_sha256()
    print(f"current scorer sha256: {scorer[:16]}\n")

    hdr = (f"{'run':<34}{'grounded':>19}{'any ungrounded':>24}"
           f"{'scorer':>10}")
    print(hdr)
    print("-" * len(hdr))

    changed = 0
    stale = 0
    results_root = os.path.join(ROOT, "results")
    for name in sorted(os.listdir(results_root)):
        d = os.path.join(results_root, name)
        if not os.path.isdir(d):
            continue
        rows = _jsonl(os.path.join(d, "results.jsonl"))
        if not rows:
            continue

        old = _json(os.path.join(d, "advisor_eval.json")) or {}
        new = advisor_eval.score_run(rows, trace=_jsonl(os.path.join(d, "trace.jsonl")))
        if not new.get("n"):
            continue

        old_sc = old.get("_scorer_sha256")
        same = old_sc == scorer
        if old and not same:
            stale += 1

        og, ng = old.get("grounded_fraction_mean"), new.get("grounded_fraction_mean")
        ou = old.get("briefings_with_an_ungrounded_number")
        nu = new.get("briefings_with_an_ungrounded_number")
        moved = (og is not None and ng is not None and abs(og - ng) > 1e-9)
        if moved:
            changed += 1

        print(f"{name:<34}{_pct(og):>9} -> {_pct(ng):<7}"
              f"{_pct(ou):>13} -> {_pct(nu):<9}"
              f"{('same' if same else 'STALE' if old else 'none'):>10}")

        if a.write:
            asrun = os.path.join(d, "advisor_eval.asrun.json")
            if old and not os.path.exists(asrun):
                with open(asrun, "w", encoding="utf-8") as fh:
                    json.dump({
                        "_note": ("the scores as written by the run itself, "
                                  "kept because a superseded measurement is "
                                  "evidence about the tooling even when it is "
                                  "no longer evidence about the model"),
                        **old}, fh, indent=2, default=str)
            with open(os.path.join(d, "advisor_eval.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(new, fh, indent=2, default=str)

    print()
    if stale:
        print(f"{stale} run(s) had been scored by a DIFFERENT version of "
              f"advisor_eval.py than the one here. Any cross-arm quality "
              f"comparison using those files compared two rulers.")
    print(f"{changed} run(s) change under the current scorer.")
    if not a.write:
        print("\nnothing written -- re-run with --write")
    else:
        print("\nrewritten. Re-run scripts/recheck_runs.py to re-judge the "
              "gates on a single consistent scorer.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
