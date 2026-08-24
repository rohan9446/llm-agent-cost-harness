#!/usr/bin/env python3
"""
Re-judge archived runs under the CURRENT validity rules.

WHY THIS EXISTS
---------------
Every run in results/ passed every check that existed when it was made. Then
an audit found the checks were weaker than the claims resting on them, and
several were strengthened:

  llm_calls_all_succeeded_first_time   a retried call spent measured wall-clock
  query_content_pinned                 ids pin which queries, not what they said
  source_tree_pinned                   which code produced these numbers
  advisor_* (three)                    A2's quality gates, as gates

"All runs passed validity" is now a weaker sentence than it looks, because it
means "passed the rules of the day". This script says the harder thing: here is
what the rules say about these runs NOW.

It re-runs check_run against the artifacts already on disk. Nothing is
recomputed from the model and nothing is backfilled -- in particular
source_tree_sha256 is NOT filled in retroactively. It could be: the code is
still here and hashing it would turn the check green in one line. It would also
be a hash of today's tree stamped onto yesterday's numbers, which is the exact
move this project exists to refuse. A run made before a provenance field
existed does not have that provenance, and the honest output is a check that
fails with "field did not exist at run time" beside it.

    python scripts/recheck_runs.py
    python scripts/recheck_runs.py --write
    python scripts/recheck_runs.py --write \\
        --advisor-reference results/A1-offline-n1000-c8-rep2
"""

from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import validity                     # noqa: E402
from agentops.validity import RunManifest         # noqa: E402

# Checks that did not exist when the published runs were made. A failure here
# is a gap in the record, not a defect in the measurement -- and the difference
# is stated rather than left for the reader to infer.
ADDED_AFTER_THE_RUNS = {
    "query_content_pinned",
    "source_tree_pinned",
    "llm_calls_all_succeeded_first_time",
}


def _jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _json(path: str) -> dict | None:
    if not os.path.exists(path):
        return None
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _resolve_ref(ref: str, run_dir: str) -> str | None:
    """Where a reference run actually lives. Same search the checks use."""
    if not ref:
        return None
    siblings = os.path.dirname(os.path.abspath(run_dir))
    for cand in (os.path.join(siblings, os.path.basename(ref.rstrip("/"))),
                 os.path.join(ROOT, ref)):
        if os.path.isdir(cand):
            return cand
    return None


def diagnose(check_name: str, manifest: RunManifest, counters: dict,
             n_attempted: int, run_dir: str = "") -> str | None:
    """Say WHY a check fails, where the signature is recognisable.

    A re-judge that only reports failures makes a reader guess whether a run is
    broken or merely old. These two failures have exact arithmetic signatures,
    so the tool can tell them apart instead.
    """
    tiers = counters.get("parser_tiers") or {}
    t1, t2 = int(tiers.get("tier1", 0)), int(tiers.get("tier2", 0))
    warm = int(manifest.warmed_workers or 0)

    if check_name == "cascade_saw_every_query" and t1 + t2 == n_attempted + warm:
        return (f"tier counts exceed attempted queries by exactly {warm}, the "
                f"warm-up count -- this run predates pipe.reset_parser_stats(), "
                f"which is the bug this check was written to catch. The check "
                f"is working; the run is superseded.")
    if check_name == "name_table_recorded" and not manifest.name_table_sha256:
        return ("manifest has no name_table_sha256 -- predates that field. "
                "Superseded by the runs that have it.")
    if check_name == "advisor_scored_by_the_same_ruler":
        return ("these advisor_eval.json files were written by different "
                "versions of advisor_eval.py, so the comparison is between two "
                "rulers rather than two arms. Fix with: "
                "python scripts/rescore_advisor.py --write, then re-run this. "
                "No GPU needed -- the briefings are already on disk.")
    if check_name == "advisor_reference_available":
        if not manifest.advisor_reference:
            return ("no reference declared at run time -- this arm was compared "
                    "by hand. Pass --advisor-reference to have the harness make "
                    "the comparison instead.")
        ref_dir = _resolve_ref(manifest.advisor_reference, run_dir)
        if not ref_dir:
            return f"{manifest.advisor_reference} is not a run directory."
        if not os.path.exists(os.path.join(ref_dir, "advisor_eval.json")):
            return (f"{manifest.advisor_reference} exists but has no "
                    f"advisor_eval.json -- it predates advisor scoring, so it "
                    f"cannot serve as a quality reference. Point at a later "
                    f"repeat of the same stage that does have one.")
    return None


def recheck(run_dir: str, write: bool,
            reference_override: str | None = None) -> tuple[int, int, int]:
    """Returns (n_pass, n_fail_real, n_fail_missing_field)."""
    man = _json(os.path.join(run_dir, "manifest.json"))
    counters = _json(os.path.join(run_dir, "counters.json"))
    if not man or not counters:
        print(f"  {os.path.basename(run_dir)}: no manifest/counters, skipped")
        return (0, 0, 0)

    fields = {f for f in RunManifest.__dataclass_fields__}
    manifest = RunManifest(**{k: v for k, v in man.items() if k in fields})

    results = _jsonl(os.path.join(run_dir, "results.jsonl"))
    failures = _jsonl(os.path.join(run_dir, "failures.jsonl"))
    if not results:
        # results.jsonl is gitignored (it carries supplied query text), so a
        # clone will not have it. n_queries minus failures is the count the
        # checks need; the per-result price-source check cannot run without the
        # rows and is reported as such rather than silently passing.
        n_ok = int(man.get("n_queries", 0)) - len(failures)
        results = [{"query_id": f"unavailable-{i}", "metrics": {}}
                   for i in range(max(0, n_ok))]
        print(f"  (results.jsonl absent -- reconstructed {len(results)} rows "
              f"from the manifest; prices_from_snapshot cannot be re-verified)")

    advisor = _json(os.path.join(run_dir, "advisor_eval.json"))
    ref = None

    # A reference supplied HERE is the reviewer stating the comparison, not the
    # run claiming it. That distinction is worth keeping: the A2 arms were
    # compared against A1 by hand, and the whole point of gating them is that
    # the verdict should not depend on my hand. Supplying the reference at
    # recheck time lets the harness reach the verdict from artifacts that were
    # frozen at run time -- the briefing scores -- while the manifest still
    # honestly says no reference was declared. Recorded as such in the output.
    supplied = False
    if reference_override and not manifest.advisor_reference:
        if manifest.advisor_style != "handout":
            manifest.advisor_reference = reference_override
            supplied = True

    if manifest.advisor_reference:
        # The reference is recorded as a repo-relative path, but a results tree
        # can be re-checked from anywhere (a copy, an archive, a clone). Look
        # beside this run first, then relative to the repo root.
        siblings = os.path.dirname(os.path.abspath(run_dir))
        for cand in (os.path.join(siblings,
                                  os.path.basename(manifest.advisor_reference.rstrip("/"))),
                     os.path.join(ROOT, manifest.advisor_reference)):
            ref = _json(os.path.join(cand, "advisor_eval.json"))
            if ref:
                break

    checks = validity.check_run(
        manifest, {**counters, "parser_tiers": counters.get("parser_tiers") or {}},
        results, failures, advisor=advisor, advisor_reference=ref)

    n_pass = sum(1 for c in checks if c.ok)
    real, missing = [], []
    for c in checks:
        if c.ok:
            continue
        (missing if c.name in ADDED_AFTER_THE_RUNS else real).append(c)

    n_attempted = len(results) + len(failures)
    print(f"\n{os.path.basename(run_dir)}  stage={manifest.stage} "
          f"parser={manifest.parser_kind} advisor={manifest.advisor_style}"
          + ("  [reference supplied at recheck]" if supplied else ""))
    print(f"  {n_pass}/{len(checks)} pass")
    for c in real:
        print(f"  FAIL   {c.name}: {c.detail[:130]}")
        why = diagnose(c.name, manifest, counters, n_attempted, run_dir)
        if why:
            print(f"         -> {why}")
    for c in missing:
        print(f"  gap    {c.name}: field not recorded at run time "
              f"-- not backfilled on purpose")

    if write:
        p = os.path.join(run_dir, "validity.recheck.json")
        with open(p, "w", encoding="utf-8") as fh:
            json.dump({
                "_note": ("re-judged under the validity rules current at the "
                          "time of this file, not the rules the run was made "
                          "under. Checks listed in _added_after_the_runs "
                          "postdate the measurement; nothing was backfilled."),
                "_added_after_the_runs": sorted(ADDED_AFTER_THE_RUNS),
                "_advisor_reference_supplied_at_recheck": (
                    manifest.advisor_reference if supplied else None),
                "_diagnosis": {c.name: diagnose(c.name, manifest, counters,
                                                n_attempted, run_dir)
                               for c in real
                               if diagnose(c.name, manifest, counters,
                                           n_attempted, run_dir)},
                "checks": [c.__dict__ for c in checks],
            }, fh, indent=2)

    return (n_pass, len(real), len(missing))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("runs", nargs="*", help="run directories; default all")
    ap.add_argument("--write", action="store_true",
                    help="write validity.recheck.json into each run directory")
    ap.add_argument("--advisor-reference", default=None,
                    help="reference run for arms whose manifest declares none "
                         "(the A2 arms predate the flag). Applied only to runs "
                         "that changed the Advisor, and recorded in the output "
                         "as supplied at recheck rather than declared at run "
                         "time.")
    a = ap.parse_args()

    runs = a.runs or sorted(
        d for d in (os.path.join(ROOT, "results", x)
                    for x in os.listdir(os.path.join(ROOT, "results")))
        if os.path.isdir(d))
    if not runs:
        print("no run directories found")
        return 1

    print(f"re-judging {len(runs)} run(s) under current rules")
    tot_real = tot_missing = 0
    for d in runs:
        _, real, missing = recheck(d, a.write, a.advisor_reference)
        tot_real += real
        tot_missing += missing

    print(f"\n{'='*70}")
    if tot_real:
        print(f"{tot_real} check(s) fail on their own terms -- these are "
              f"substantive and the runs concerned should not be reported "
              f"without addressing them.")
    else:
        print("No archived run fails a check it could have passed.")
    print(f"{tot_missing} gap(s): provenance fields added after these runs were "
          f"made. Closing them means re-running, not editing a manifest.")
    return 1 if tot_real else 0


if __name__ == "__main__":
    raise SystemExit(main())
