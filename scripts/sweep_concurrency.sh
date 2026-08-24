#!/usr/bin/env bash
# S1 -- cost and latency versus request concurrency.
#
# The one experiment B0 already has direct evidence for: vLLM reported
# `GPU KV cache usage: 0.0%` and `Waiting: 0 reqs` throughout the C=8 run, on
# a server with 451,056 tokens of KV capacity. The batch was never full. This
# sweep finds out what that headroom is worth and what it costs.
#
# What is held constant: the server (same process, same weights, same flags),
# the frozen query set, prefix caching off, the seed, the snapshot.
# What varies: client-side concurrency only.
#
# NOTE ON THE PROTOCOL. METHODS.md says never reuse a server across a
# configuration change. Concurrency is not a server configuration -- it is how
# many requests the client has in flight. The server's argv, weights and KV
# allocation are byte-identical across every point here, which is exactly why
# the comparison is clean. Restarting between points would ADD a confound
# (fresh CUDA graphs, cold allocator) rather than remove one.
#
# ORDER AND THERMAL DRIFT. Points run in ascending order, so load increases
# monotonically over ~15 minutes. That aliases thermal drift onto the
# treatment. Two defences: every run records SM clock and power draw in
# gpu.json, and `--reverse` re-runs the sweep descending. If the two orders
# agree, drift is not driving the result. If they disagree, the drift is the
# finding.
#
#   ./scripts/sweep_concurrency.sh                 # ascending, rep1
#   ./scripts/sweep_concurrency.sh --reverse       # descending, rep2
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

PY="${PY:-python3}"
STAGE="${STAGE:-B0}"
QSET="${QSET:-data/query_sets/systems_100.json}"
POINTS=(${POINTS:-1 2 4 8 16 32 64})
REP=1

if [[ "${1:-}" == "--reverse" ]]; then
  # shellcheck disable=SC2207
  POINTS=($(printf '%s\n' "${POINTS[@]}" | tac))
  REP=2
  echo "reversed order, repeat $REP -- this is the drift control"
fi

echo "sweep: C = ${POINTS[*]}   (repeat $REP)"
echo "server is NOT restarted between points; concurrency is client-side"
echo

FAILED=()
for C in "${POINTS[@]}"; do
  echo "=============================================================="
  echo "  C=$C   ($(date +%H:%M:%S))"
  echo "=============================================================="
  if ! $PY scripts/run_bench.py \
        --stage "$STAGE" \
        --concurrency "$C" \
        --repeat "$REP" \
        --query-set "$QSET" \
        --expect-apc off \
        --max-failure-rate 0 \
        --overwrite; then
    echo "  !! C=$C did not pass its validity checks -- recorded, not skipped"
    FAILED+=("$C")
  fi
  echo
  # Let the device settle so the next point does not inherit the previous
  # point's thermal state. Cheap insurance against the drift this sweep is
  # most vulnerable to.
  sleep 20
done

echo "=============================================================="
if [[ ${#FAILED[@]} -eq 0 ]]; then
  echo "all ${#POINTS[@]} points passed validity"
else
  echo "points that FAILED validity: ${FAILED[*]}"
  echo "these are excluded from the curve by sweep_report.py, and named there."
fi
echo
echo "next:  $PY scripts/sweep_report.py --repeat $REP"
