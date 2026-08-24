#!/usr/bin/env bash
# The Makefile's targets, for machines without make.
#
# The measurement node has no make(1), so this exists to run the same steps
# there. "Same targets, same flags" is a promise that has to be kept literally
# or it is worse than no promise at all: a shell script that quietly behaves
# differently from the Makefile is a second implementation nobody diffs.
#
# It did not keep it. `report` here carried a fallback the Makefile had already
# removed:
#
#     [ -z "${d:-}" ] && d=$(ls -dt results/*/ | head -1)
#
# so `STAGE=B0 C=8 ./run.sh report` with no matching B0 run reported whichever
# experiment happened to be newest -- an A1 directory under a B0 heading. The
# Makefile exits 1 there, with a comment explaining that exact bug. This file
# was the one actually used for the measurements.
#
#   ./run.sh smoke
#   STAGE=A1 C=8 ./run.sh bench
#   STAGE=A1 C=8 ./run.sh report
#
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PY="${PY:-python3}"
N="${N:-100}"
C="${C:-8}"
STAGE="${STAGE:-B0}"
APC="${APC:-off}"
QSET="data/query_sets/systems_100.json"

target="${1:-help}"

case "$target" in

help)
  cat <<'EOF'
  ./run.sh bootstrap  one-time setup + preflight; run this first on the lab box
  ./run.sh smoke      harness check, no GPU or network needed
  ./run.sh snapshot   build the frozen price snapshot (once, first)
  ./run.sh serve      launch vLLM for B0 (prefix caching OFF)
  ./run.sh calibrate  C=1 run, supplies attribution weights for the report
  ./run.sh bench      run the frozen systems_100 set  [C=8 STAGE=B0 APC=off]
  ./run.sh report     metrics for the newest run of THIS stage and concurrency
  ./run.sh verify     confirm the handout patches still apply cleanly
  ./run.sh audit      what would be committed, and does any of it leak
  ./run.sh recheck    re-judge archived runs under the current validity rules
EOF
  ;;

bootstrap) ./scripts/bootstrap.sh ;;
smoke)     "$PY" scripts/smoke_test.py ;;
snapshot)  "$PY" scripts/build_snapshot.py ;;
serve)     ./scripts/serve_vllm.sh ;;

calibrate)
  "$PY" scripts/run_bench.py --stage "${STAGE}cal" --concurrency 1 \
    --query-set "$QSET" --expect-apc "$APC" --max-failure-rate 0
  ;;

bench)
  "$PY" scripts/run_bench.py --stage "$STAGE" --concurrency "$C" \
    --query-set "$QSET" --expect-apc "$APC" --max-failure-rate 0
  ;;

report)
  # NO FALLBACK. Identical to the Makefile: if no run matches the stage and
  # concurrency being asked for, that is an error, not an invitation to report
  # a different experiment.
  d=$(ls -dt results/"$STAGE"-offline-*-c"$C"-* 2>/dev/null | head -1 || true)
  cal=$(ls -dt results/"$STAGE"cal-offline-*-c1-* 2>/dev/null | head -1 || true)
  if [ -z "${d:-}" ]; then
    echo "no run directory matching results/$STAGE-offline-*-c$C-*" >&2
    echo "available:" >&2
    ls -1dt results/*/ 2>/dev/null | head -8 >&2
    exit 1
  fi
  if [ -n "${cal:-}" ]; then
    echo "reporting $d with weights from $cal"
    "$PY" scripts/report.py "$d" --weights-from "$cal"
  else
    echo "reporting $d (no C=1 calibration for stage $STAGE -- run './run.sh calibrate')"
    "$PY" scripts/report.py "$d"
  fi
  ;;

verify)
  echo "--- upstream does not compile (expected) ---"
  "$PY" -m py_compile docs/upstream/portfolio/agents/metrics_agent.py || true
  echo "--- patched tree compiles ---"
  "$PY" -m py_compile $(find workflow -name '*.py') && echo OK
  echo "--- diff vs upstream ---"
  diff -ru -x '__pycache__' -x '*.pyc' docs/upstream/portfolio workflow/portfolio || true
  ;;

audit)   ./scripts/pre_commit_audit.sh ;;

compare) "$PY" scripts/compare_runs.py --headline ;;

recheck)
  "$PY" scripts/rescore_parser.py --write
  "$PY" scripts/recheck_runs.py --write \
    --advisor-reference results/A1-offline-n1000-c8-rep2
  "$PY" scripts/results_index.py --write
  ;;

clean)
  rm -rf results/*/ logs/*.log
  find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
  find . -name '*.pyc' -delete 2>/dev/null || true
  ;;

*)
  echo "unknown target: $target" >&2
  "$0" help >&2
  exit 2
  ;;
esac
