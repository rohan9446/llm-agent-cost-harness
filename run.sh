#!/usr/bin/env bash
# Stand-in for the Makefile on boxes without `make`. Same targets, same flags.
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PY=${PY:-python}
C=${C:-8}
STAGE=${STAGE:-B0}
APC=${APC:-off}
QSET=data/query_sets/systems_100.json

case "${1:-help}" in
  bootstrap) ./scripts/bootstrap.sh ;;
  smoke)     $PY scripts/smoke_test.py ;;
  snapshot)  $PY scripts/build_snapshot.py ;;
  serve)     ./scripts/serve_vllm.sh ;;

  calibrate) $PY scripts/run_bench.py --stage "${STAGE}cal" --concurrency 1 \
               --query-set "$QSET" --expect-apc "$APC" --max-failure-rate 0 ;;

  bench)     $PY scripts/run_bench.py --stage "$STAGE" --concurrency "$C" \
               --query-set "$QSET" --expect-apc "$APC" --max-failure-rate 0 \
               --repeat "${2:-1}" ;;

  report)
    # Attribution weights are fitted at C=1 and carried into the C=8 report,
    # because above C=1 the measured times contain queueing delay.
    d=$(ls -dt results/${STAGE}-offline-*-c${C}-* 2>/dev/null | head -1 || true)
    cal=$(ls -dt results/${STAGE}cal-offline-*-c1-* 2>/dev/null | head -1 || true)
    [ -z "${d:-}" ] && d=$(ls -dt results/*/ | head -1)
    if [ -n "${cal:-}" ]; then
      echo "reporting $d with weights from $cal"
      $PY scripts/report.py "$d" --weights-from "$cal" "${@:2}"
    else
      echo "reporting $d (no C=1 calibration found -- run ./run.sh calibrate)"
      $PY scripts/report.py "$d" "${@:2}"
    fi ;;

  verify)
    echo "--- upstream should NOT compile ---"
    $PY -m py_compile docs/upstream/portfolio/agents/metrics_agent.py || true
    echo "--- patched tree ---"
    $PY -m py_compile $(find workflow -name '*.py') && echo OK
    diff -ru -x '__pycache__' -x '*.pyc' docs/upstream/portfolio workflow/portfolio || true ;;

  clean)
    rm -rf results/*/ logs/*.log
    find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true ;;

  *) echo "usage: ./run.sh {bootstrap|smoke|snapshot|serve|calibrate|bench [rep]|report|verify|clean}" ;;
esac
