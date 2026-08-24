#!/usr/bin/env bash
# Launch vLLM for B0 and record what is actually running.
#
# The record is not bookkeeping. vLLM enables automatic prefix caching by
# default, so a B0 run that assumed APC was off -- because nobody passed a flag
# -- would be measuring a cached baseline, and the later S3 comparison would
# find nothing. The runner asserts the setting rather than trusting a default.
#
# But a file written at launch only states *intent*. It says nothing about which
# process currently holds the port: leave a stale server.json next to a server
# restarted with different flags and every check still passes while the real
# configuration is wrong. So the PID is recorded too.
#
# The PID is captured with $$ BEFORE the exec. exec replaces the process image
# without changing the PID, so $$ is exactly the PID vLLM will run under -- but
# only if nothing turns the exec into a pipeline. That is why logging goes
# through a process substitution rather than `| tee`: a pipe would fork, the
# shell would survive as a separate process, and the recorded PID would belong
# to the wrong one.
#
#   ./scripts/serve_vllm.sh              # B0: prefix caching OFF
#   APC=on ./scripts/serve_vllm.sh       # S3a
#
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
mkdir -p "$ROOT/results" "$ROOT/logs"
LOG="$ROOT/logs/vllm.log"

MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
PORT="${PORT:-8000}"
DTYPE="${DTYPE:-bfloat16}"
TP="${TP:-1}"
MAX_LEN="${MAX_LEN:-8192}"
GPU_UTIL="${GPU_UTIL:-0.90}"
SEED="${SEED:-1337}"
APC="${APC:-off}"
GPUS="${CUDA_VISIBLE_DEVICES:-0}"
export CUDA_VISIBLE_DEVICES="$GPUS"

if [[ "$APC" == "on" ]]; then
  APC_FLAG="--enable-prefix-caching"
else
  APC_FLAG="--no-enable-prefix-caching"
fi

if ! python -c "import vllm" 2>/dev/null; then
  echo "vLLM is not importable in this environment." >&2
  exit 1
fi

VLLM_VERSION="$(python -c 'import vllm; print(vllm.__version__)')"
# vLLM 0.2x groups its help; plain --help is a summary and omits most
# engine args. Ask for the full list, and fall back for older builds.
HELP="$(vllm serve --help=all 2>/dev/null)"
[[ -z "$HELP" ]] && HELP="$(vllm serve --help 2>&1 || true)"

# Feature checks are hard failures, not warnings. A warning that scrolls past
# leaves the run measuring a configuration nobody chose -- the whole failure
# mode this script exists to prevent.
if ! grep -q -- "--no-enable-prefix-caching" <<<"$HELP"; then
  echo "FATAL: vLLM $VLLM_VERSION does not accept --no-enable-prefix-caching." >&2
  echo "Prefix caching could not be set explicitly, so B0 and S3 would not be" >&2
  echo "distinguishable. Check the flag name for this version." >&2
  exit 1
fi

# Request-logging spelling changed across versions: older builds take
# --disable-log-requests, current ones --no-enable-log-requests (and default
# the feature off). Either is fine; silently having neither is not, because on
# a build that logs by default that overhead lands inside every measured
# latency.
if grep -q -- "--no-enable-log-requests" <<<"$HELP"; then
  LOGFLAG="--no-enable-log-requests"
elif grep -q -- "--disable-log-requests" <<<"$HELP"; then
  LOGFLAG="--disable-log-requests"
else
  echo "FATAL: vLLM $VLLM_VERSION accepts neither --no-enable-log-requests nor" >&2
  echo "--disable-log-requests, so request logging cannot be turned off" >&2
  echo "explicitly. Pin a version that does before benchmarking." >&2
  exit 1
fi

ARGS=(
  serve "$MODEL"
  --port "$PORT"
  --dtype "$DTYPE"
  --tensor-parallel-size "$TP"
  --max-model-len "$MAX_LEN"
  --gpu-memory-utilization "$GPU_UTIL"
  --seed "$SEED"
  "$LOGFLAG"
  "$APC_FLAG"
)

# Process substitution, not a pipe: keeps this shell's PID for the exec below.
exec > >(tee -a "$LOG") 2>&1

python - "$ROOT/results/server.json" "$VLLM_VERSION" "$MODEL" "$APC" "$GPUS" \
         "$PORT" "$DTYPE" "$TP" "$$" "$LOGFLAG" <<'PY' "${ARGS[@]}"
import json, sys, time
out, version, model, apc, gpus, port, dtype, tp, pid, logflag = sys.argv[1:11]
argv = sys.argv[11:]
json.dump({
    "launched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    "vllm_version": version,
    "model": model,
    "prefix_caching": apc,          # asserted by the runner, not assumed
    "cuda_visible_devices": gpus,   # the runner scopes GPU telemetry to this
    "port": int(port),
    "dtype": dtype,
    "tensor_parallel_size": int(tp),
    "log_requests_flag": logflag,   # spelling varies by vLLM version
    # This PID is what /proc/<pid>/cmdline is read from before every run, so
    # the configuration under test is verified against the live process rather
    # than against this file.
    "pid": int(pid),
    "argv": ["vllm"] + argv,
}, open(out, "w"), indent=2)
print(f"recorded launch (pid {pid}) -> {out}")
PY

echo "vLLM $VLLM_VERSION | $MODEL | dtype=$DTYPE tp=$TP apc=$APC $LOGFLAG gpus=$GPUS port=$PORT pid=$$"
exec vllm "${ARGS[@]}"
