#!/usr/bin/env python3
"""
Does supplied query text survive anywhere in git HISTORY?

WHY THIS EXISTS
---------------
pre_commit_audit.sh answers "would this commit leak?". That is the wrong tense
for a repository that has already been pushed four times while the corpus was
being published. Removing a file from the working tree removes it from the next
commit and from nothing else: every blob ever committed is still in the object
store, still reachable through `git log -p`, still served by GitHub's raw
endpoint at the old commit SHA, and still in every clone anyone made.

The working-tree audit and the history audit are the same class of distinction
as filenames-versus-contents, one level up. Both were assumed rather than
checked, and both turned out to be hiding something.

This walks every blob in every reachable commit and applies the same 6-word
shingle test the pre-commit audit uses on the working tree.

    python scripts/history_audit.py

Exit 0 = history is clean. Exit 1 = it is not, and the only real fix is to
rewrite history and force-push; deleting the file in a new commit does nothing.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
CORPUS = os.path.join(ROOT, "data", "queries.json")
K = 6


def _words(s: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", s.lower())


def _run(argv: list[str], binary: bool = False):
    return subprocess.run(argv, cwd=ROOT, capture_output=True,
                          text=not binary, check=False)


def main() -> int:
    if not os.path.isdir(os.path.join(ROOT, ".git")):
        print("not a git repository -- nothing to audit")
        return 0
    if not os.path.exists(CORPUS):
        print("data/queries.json not present -- cannot check history against "
              "the corpus.\nRun this on the machine that holds the corpus, or "
              "you are not auditing anything.")
        return 1

    with open(CORPUS, encoding="utf-8") as fh:
        corpus = json.load(fh)
    shingles: dict[str, object] = {}
    for q in corpus:
        w = _words(q.get("query", ""))
        for i in range(max(0, len(w) - K + 1)):
            shingles.setdefault(" ".join(w[i:i + K]), q.get("id"))
    if not shingles:
        print("corpus produced no shingles -- refusing to report clean")
        return 1

    out = _run(["git", "rev-list", "--all", "--objects"])
    if out.returncode != 0:
        print("git rev-list failed:", out.stderr.strip())
        return 1

    entries = []
    for line in out.stdout.splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) == 2:
            entries.append((parts[0], parts[1]))

    print(f"scanning {len(entries)} objects across all reachable commits")

    hits: dict[str, set] = {}
    checked = 0
    for sha, path in entries:
        if not path or path.endswith("/"):
            continue
        kind = _run(["git", "cat-file", "-t", sha]).stdout.strip()
        if kind != "blob":
            continue
        blob = _run(["git", "cat-file", "blob", sha], binary=True).stdout
        if not blob or b"\0" in blob[:4096]:
            continue
        checked += 1
        w = _words(blob.decode("utf-8", "replace"))
        for i in range(max(0, len(w) - K + 1)):
            sh = " ".join(w[i:i + K])
            if sh in shingles:
                hits.setdefault(path, set()).add(sha[:12])
                break

    print(f"read {checked} text blobs")

    if not hits:
        print("\nCLEAN: no 6-word run from any supplied query appears in any "
              "blob in history.")
        return 0

    print(f"\nLEAK IN HISTORY: {len(hits)} path(s) carry supplied query text "
          f"in at least one past commit.\n")
    for path in sorted(hits):
        blobs = sorted(hits[path])
        print(f"  {path}")
        print(f"    in {len(blobs)} blob version(s), e.g. {', '.join(blobs[:3])}")

    print("\nDeleting these files in a new commit DOES NOT fix this. The blobs\n"
          "stay reachable at their old commit SHAs, including through GitHub's\n"
          "raw endpoint. The fix is to replace history:\n")
    print("  git checkout --orphan clean-main")
    print("  git add -A && ./scripts/pre_commit_audit.sh   # must pass")
    print("  git commit -m 'measurement harness and results'")
    print("  git branch -D main && git branch -m main")
    print("  git push --force origin main")
    print("\nThen re-run this script against a FRESH clone, not this tree.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
