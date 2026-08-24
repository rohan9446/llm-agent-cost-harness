#!/usr/bin/env python3
"""
Re-score every run's parser accuracy with the current scorer and alias map.

The advisor scorer needed this and so does this one, for the same reason: a
parser_eval.json written before parser_eval.py stamped its inputs cannot be
compared to one written after, and nothing in the file said which it was.

Two things changed under these artifacts since they were produced:

  _scorer_sha256   parser_eval.py itself -- including the equal-weight pattern
                   rewrite, which fixed two queries that the old alternation
                   silently dropped out of the derived score entirely
  _vocab_sha256    data/vocab.json, the alias map the derived ticker and weight
                   labels are built from

Both are recomputed here from results.jsonl and failures.jsonl, which were
frozen at run time. No GPU, no model, no re-measurement: the parses are the
parses, only the ruler changes, and it changes for every run at once.

The original is preserved as parser_eval.asrun.json the first time a run is
rescored.

    python scripts/rescore_parser.py            # report, change nothing
    python scripts/rescore_parser.py --write
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import parser_eval                  # noqa: E402

VOCAB = os.path.join(ROOT, "data", "vocab.json")


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


def _acc(d: dict, section: str, key: str):
    v = ((d.get(section) or {}).get(key) or {}).get("accuracy")
    return v


def _pct(x) -> str:
    return "—" if x is None else f"{100*x:.1f}%"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    if not os.path.exists(VOCAB):
        print(f"no alias map at {VOCAB}", file=sys.stderr)
        return 1
    vocab = parser_eval.load_vocab(VOCAB)
    scorer, vsha = parser_eval.scorer_sha256(), parser_eval.vocab_sha256(VOCAB)
    print(f"scorer {scorer[:16]}   vocab {vsha[:16]}\n")

    hdr = f"{'run':<34}{'ticker set':>21}{'weights':>21}{'ruler':>9}"
    print(hdr)
    print("-" * len(hdr))

    stale = changed = 0
    results_root = os.path.join(ROOT, "results")
    for name in sorted(os.listdir(results_root)):
        d = os.path.join(results_root, name)
        if not os.path.isdir(d):
            continue
        rows = _jsonl(os.path.join(d, "results.jsonl"))
        if not rows:
            continue
        fails = _jsonl(os.path.join(d, "failures.jsonl"))

        old = _json(os.path.join(d, "parser_eval.json")) or {}
        new = parser_eval.score(rows, vocab, failures=fails, vocab_path=VOCAB)

        same = (old.get("_scorer_sha256") == scorer
                and old.get("_vocab_sha256") == vsha)
        if old and not same:
            stale += 1
        ot, nt = _acc(old, "derived", "ticker_set"), _acc(new, "derived", "ticker_set")
        ow, nw = (_acc(old, "derived", "weights_within_1e-3"),
                  _acc(new, "derived", "weights_within_1e-3"))
        if ot is not None and nt is not None and abs(ot - nt) > 1e-9:
            changed += 1

        print(f"{name:<34}{_pct(ot):>9} -> {_pct(nt):<9}"
              f"{_pct(ow):>9} -> {_pct(nw):<9}"
              f"{('same' if same else 'STALE' if old else 'none'):>9}")

        if a.write:
            asrun = os.path.join(d, "parser_eval.asrun.json")
            if old and not os.path.exists(asrun):
                with open(asrun, "w", encoding="utf-8") as fh:
                    json.dump({"_note": "scores as written by the run itself, "
                                        "under an unrecorded scorer and alias "
                                        "map", **old}, fh, indent=2, default=str)
            with open(os.path.join(d, "parser_eval.json"), "w",
                      encoding="utf-8") as fh:
                json.dump(new, fh, indent=2, default=str)

    print()
    if stale:
        print(f"{stale} run(s) were scored by an unrecorded or different "
              f"scorer/alias map.")
    print(f"{changed} run(s) change under the current ruler.")
    print("\nnothing written -- re-run with --write" if not a.write
          else "\nrewritten. Re-run scripts/recheck_runs.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
