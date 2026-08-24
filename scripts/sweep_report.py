#!/usr/bin/env python3
"""
Collect a concurrency sweep into one table, one CSV and one honest conclusion.

Cost per query falls with concurrency almost by definition -- the same wall
clock serves more work -- so a curve that only shows cost falling is not a
finding, it is arithmetic. What makes it a result is the price paid for it:
tail latency. This script reports both against each other and locates the knee,
which is the only part a reader can act on.

    python scripts/sweep_report.py --repeat 1
    python scripts/sweep_report.py --repeat 1 --usd-per-gpu-hour 1.39 \
        --rate-source "RunPod A100 PCIe 80GB on-demand, retrieved 2026-08-22"
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def find_runs(stage: str, repeat: int) -> list[tuple[int, str]]:
    pat = os.path.join(ROOT, "results", f"{stage}-offline-n*-c*-rep{repeat}")
    out = []
    for d in glob.glob(pat):
        m = re.search(r"-c(\d+)-rep\d+$", d)
        if m:
            out.append((int(m.group(1)), d))
    return sorted(out)


def find_calibration(stage: str) -> str | None:
    cands = sorted(glob.glob(os.path.join(ROOT, "results", f"{stage}cal*-c1-*")))
    return cands[-1] if cands else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="B0")
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--usd-per-gpu-hour", type=float, default=1.39)
    ap.add_argument("--rate-source",
                    default="RunPod A100 PCIe 80GB on-demand, retrieved 2026-08-22")
    ap.add_argument("--slo-p95", type=float, default=5.0,
                    help="end-to-end p95 budget in seconds; the knee is the "
                         "highest concurrency still inside it")
    ap.add_argument("--no-recost", action="store_true",
                    help="use each run's existing report.json instead of "
                         "re-running report.py at a common rate")
    a = ap.parse_args()

    runs = find_runs(a.stage, a.repeat)
    if not runs:
        print(f"no runs found for stage {a.stage} repeat {a.repeat}", file=sys.stderr)
        return 2
    cal = find_calibration(a.stage)
    if not cal:
        print("WARNING: no C=1 calibration run found. Attribution weights will "
              "be fitted per point, on runs that contain queueing delay, and "
              "the per-stage split will not be comparable across points.",
              file=sys.stderr)

    # Re-cost every point at one rate with one set of weights. Without this the
    # points are not comparable: each report.json may have been generated with
    # a different rate or different weights.
    if not a.no_recost:
        for c, d in runs:
            cmd = [sys.executable, os.path.join(HERE, "report.py"), d,
                   "--usd-per-gpu-hour", str(a.usd_per_gpu_hour),
                   "--rate-source", a.rate_source]
            if cal:
                cmd += ["--weights-from", cal]
            subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)

    rows = []
    for c, d in runs:
        p = os.path.join(d, "report.json")
        if not os.path.exists(p):
            print(f"  (skipping C={c}: no report.json)", file=sys.stderr)
            continue
        r = json.load(open(p, encoding="utf-8"))
        v = os.path.join(d, "validity.json")
        valid = True
        if os.path.exists(v):
            valid = all(ch.get("ok") for ch in json.load(open(v, encoding="utf-8")))
        rel = r.get("reliability") or {}
        gpus = (r.get("gpu") or {}).get("gpus") or {}
        g = next(iter(gpus.values()), {})
        rows.append({
            "concurrency": c,
            "valid": valid,
            "wall_s": r["run"]["wall_s"],
            "queries_per_s": r["throughput"]["queries_per_s"],
            "output_tok_per_s": r["throughput"]["output_tokens_per_s"],
            "usd_per_attempted": r["cost"]["cost_per_attempted_query_usd"],
            "e2e_p50": (r["latency_s"]["end_to_end"] or {}).get("p50"),
            "e2e_p95": (r["latency_s"]["end_to_end"] or {}).get("p95"),
            "e2e_p99": (r["latency_s"]["end_to_end"] or {}).get("p99"),
            "ttft_p95": (r["latency_s"]["ttft"] or {}).get("p95"),
            "tpot_p50": (r["latency_s"]["tpot"] or {}).get("p50"),
            "workflow_failure_rate": rel.get("workflow_failure_rate"),
            "infra_failure_rate": rel.get("infrastructure_failure_rate"),
            "sm_clock_mhz_mean": g.get("sm_clock_mhz_mean"),
            "power_w_mean": g.get("power_w_mean"),
            "run_dir": os.path.basename(d),
        })

    if not rows:
        print("no reports to summarise", file=sys.stderr)
        return 2

    csv_path = os.path.join(ROOT, "results", f"sweep-{a.stage}-rep{a.repeat}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    base = rows[0]
    print(f"\n{'='*82}")
    print(f"CONCURRENCY SWEEP -- {a.stage}, repeat {a.repeat}")
    print(f"rate ${a.usd_per_gpu_hour}/GPU-h  ({a.rate_source})")
    if cal:
        print(f"attribution weights from {os.path.basename(cal)} (C=1) for every point")
    print("="*82)
    print(f"{'C':>4}{'wall s':>9}{'q/s':>8}{'out tok/s':>11}"
          f"{'$/query':>11}{'vs C=1':>8}{'p50 s':>8}{'p95 s':>8}{'p99 s':>8}{'wf%':>6}")
    for r in rows:
        speedup = base["usd_per_attempted"] / r["usd_per_attempted"] \
            if r["usd_per_attempted"] else float("nan")
        flag = "" if r["valid"] else "  <-- FAILED VALIDITY"
        print(f"{r['concurrency']:>4}{r['wall_s']:>9.1f}{r['queries_per_s']:>8.2f}"
              f"{r['output_tok_per_s']:>11.0f}{r['usd_per_attempted']:>11.6f}"
              f"{speedup:>7.1f}x{r['e2e_p50']:>8.2f}{r['e2e_p95']:>8.2f}"
              f"{r['e2e_p99']:>8.2f}"
              f"{100*(r['workflow_failure_rate'] or 0):>5.0f}%{flag}")

    good = [r for r in rows if r["valid"]]

    # A single SLO threshold applied to a p95 that happens to sit on it gives a
    # recommendation that flips between identical experiments. It did: rep1 put
    # the knee at C=16 and rep2 at C=32, because C=32's p95 measured 5.02s and
    # 4.84s against a 5.0s budget. The threshold moved, not the system. So the
    # budget is swept instead of assumed, and any point whose p95 lands within
    # 10% of a budget is named as marginal rather than counted as inside it.
    print("\nSLO SENSITIVITY -- highest concurrency inside each p95 budget")
    print("  (a budget is a choice, not a measurement; here is how the answer moves)")
    budgets = sorted({round(b, 2) for b in
                      [a.slo_p95, 3.0, 3.5, 4.0, 5.0, 6.0, 8.0]})
    for b in budgets:
        inside = [r for r in good if (r["e2e_p95"] or 1e9) <= b]
        if not inside:
            print(f"  p95 <= {b:>4.1f}s   no point qualifies")
            continue
        k = max(inside, key=lambda r: r["concurrency"])
        marginal = ""
        near = [r for r in good if r["e2e_p95"] and b < r["e2e_p95"] <= b * 1.10]
        if near:
            marginal = ("   MARGINAL: C=" +
                        ",".join(str(r["concurrency"]) for r in near) +
                        f" within 10% of this budget")
        print(f"  p95 <= {b:>4.1f}s   C={k['concurrency']:<3} "
              f"${k['usd_per_attempted']:.6f}/query  "
              f"{base['usd_per_attempted']/k['usd_per_attempted']:>5.1f}x cheaper "
              f"than C=1{marginal}")

    best = min(good, key=lambda r: r["usd_per_attempted"])
    print(f"\nCHEAPEST at C={best['concurrency']}: ${best['usd_per_attempted']:.6f}"
          f"/query, but p95 {best['e2e_p95']:.2f}s "
          f"({best['e2e_p95']/base['e2e_p95']:.1f}x the C=1 tail)")
    print("  The cheapest point is the right point only if nobody is waiting on")
    print("  the answer. Report the curve and the budget you chose, not a knee.")

    clocks = [r["sm_clock_mhz_mean"] for r in rows if r["sm_clock_mhz_mean"]]
    if len(clocks) >= 2:
        drift = (max(clocks) - min(clocks)) / max(clocks)
        print(f"\nSM clock across points: {min(clocks):.0f}-{max(clocks):.0f} MHz "
              f"({100*drift:.1f}% spread)")
        if drift > 0.05:
            print("  >5% spread: the device throttled during the sweep. Run "
                  "`./scripts/sweep_concurrency.sh --reverse` and compare -- "
                  "if the curves disagree, thermal drift is confounding the "
                  "treatment and the sweep needs cooldowns, not conclusions.")
        else:
            print("  clocks held: thermal drift is not driving this curve")

    print(f"\nwrote {csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
