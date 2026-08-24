#!/usr/bin/env bash
# One-time setup and pre-flight for the lab box.
#
# Everything here is cheap and none of it needs a GPU reservation. Run it as
# soon as you land on the machine -- the failures it catches (gated model,
# missing vLLM, no free GPU, wrong Python) all cost far more when discovered
# halfway into a booked window.
#
#   ./scripts/bootstrap.sh
#
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODEL="${LLM_MODEL:-meta-llama/Llama-3.1-8B-Instruct}"
FAIL=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAIL=1; }

echo "== 0. where am I =="
echo "     host: $(hostname)"
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  ok "inside Slurm job $SLURM_JOB_ID on ${SLURM_JOB_NODELIST:-?}"
  [[ -n "${SLURM_JOB_GPUS:-}${SLURM_STEP_GPUS:-}" ]] && \
    echo "     allocated GPUs: ${SLURM_JOB_GPUS:-$SLURM_STEP_GPUS}"
  if command -v squeue >/dev/null 2>&1; then
    LEFT=$(squeue -h -j "$SLURM_JOB_ID" -o "%L" 2>/dev/null)
    [[ -n "$LEFT" ]] && echo "     walltime remaining: $LEFT"
    echo "     NOTE: vLLM dies when the allocation ends. Keep the server and the"
    echo "     benchmark inside the SAME job, and leave margin -- B0 is minutes,"
    echo "     but the first model download is not."
  fi
elif command -v sinfo >/dev/null 2>&1 || command -v sbatch >/dev/null 2>&1; then
  warn "Slurm is present but this is not inside a job -- probably a login node"
  echo "     Login nodes usually have internet but no GPU; compute nodes are"
  echo "     often the reverse. Do the two internet steps here (make snapshot,"
  echo "     model download) and the GPU steps in a job."
elif command -v qsub >/dev/null 2>&1; then
  warn "PBS/Torque detected; same split applies as for Slurm"
else
  ok "no batch scheduler detected -- treating this as a plain machine"
fi

echo
echo "== 1. python =="
PY_OK=$(python3 -c 'import sys; print(1 if sys.version_info[:2] >= (3,10) else 0)' 2>/dev/null || echo 0)
if [[ "$PY_OK" == "1" ]]; then
  ok "python3 $(python3 -V 2>&1 | cut -d' ' -f2)"
else
  bad "python3 >= 3.10 required (the code uses PEP 604 unions)"
fi

echo
echo "== 2. virtualenv + client deps =="
if [[ ! -d .venv ]]; then
  python3 -m venv .venv && ok "created .venv" || bad "could not create .venv"
fi
# shellcheck disable=SC1091
source .venv/bin/activate 2>/dev/null || bad "could not activate .venv"
pip install -q --upgrade pip >/dev/null 2>&1
if pip install -q -r requirements.txt; then
  ok "openai, yfinance, numpy installed"
else
  bad "pip install -r requirements.txt failed"
fi

echo
echo "== 3. vLLM =="
if python -c "import vllm" 2>/dev/null; then
  VLLM_V=$(python -c 'import vllm; print(vllm.__version__)')
  ok "vLLM $VLLM_V"
  echo "     -> record this in METHODS.md under 'vLLM version'"
  # vLLM 0.2x groups its help; plain --help is a summary and omits most
# engine args. Ask for the full list, and fall back for older builds.
HELP="$(vllm serve --help=all 2>/dev/null)"
[[ -z "$HELP" ]] && HELP="$(vllm serve --help 2>&1 || true)"
  grep -q -- "--no-enable-prefix-caching" <<<"$HELP" \
    && ok "--no-enable-prefix-caching supported (B0 needs it)" \
    || bad "--no-enable-prefix-caching NOT supported; B0 and S3 would be indistinguishable"
  if grep -q -- "--no-enable-log-requests" <<<"$HELP"; then
    ok "request logging: --no-enable-log-requests"
  elif grep -q -- "--disable-log-requests" <<<"$HELP"; then
    ok "request logging: --disable-log-requests (older spelling)"
  else
    bad "no way to disable request logging; its overhead would land in every latency"
  fi
else
  bad "vLLM not importable in this venv"
  echo "     vLLM must be installed INTO .venv (it needs the same interpreter):"
  echo "       source .venv/bin/activate && pip install vllm"
  echo "     If the lab already has a vLLM environment, use that interpreter"
  echo "     instead and install requirements.txt into it."
fi

echo
echo "== 4. GPUs =="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader | while IFS= read -r line; do
    echo "     $line"
  done

  # bf16 needs compute capability >= 8.0 (Ampere). On Volta/Turing the B0
  # launch would fail outright, or silently fall back, so the dtype has to
  # change -- and METHODS.md has to say it changed.
  CC=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' ')
  if [[ -n "$CC" ]]; then
    CC_MAJOR=${CC%%.*}
    if [[ "$CC_MAJOR" -ge 8 ]]; then
      ok "compute capability $CC -- bfloat16 supported, keep DTYPE=bfloat16"
    else
      warn "compute capability $CC -- NO bfloat16 (needs >= 8.0)"
      echo "     Launch with DTYPE=float16 make serve, and record the change in"
      echo "     METHODS.md. fp16 vs bf16 is a real difference in the baseline,"
      echo "     not a formality."
    fi
    if [[ "$CC_MAJOR" -ge 10 ]]; then
      ok "SM${CC_MAJOR}x -- MXFP8 available later for the S2 experiment"
    fi
  fi
  VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
  if [[ -n "$VRAM" && "$VRAM" -lt 20000 ]]; then
    warn "${VRAM} MiB VRAM -- an 8B model in 16-bit is ~16 GB of weights alone"
    echo "     Either lower MAX_LEN and GPU_UTIL, or use a smaller model and say"
    echo "     so in METHODS.md."
  fi
  FREE=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' '$2 < 2000 {print $1}' | paste -sd, -)
  if [[ -n "$FREE" ]]; then
    ok "idle GPUs: $FREE   -> use one of these as CUDA_VISIBLE_DEVICES"
  else
    warn "no GPU is idle right now; B0 needs a quiet device or the cost basis is meaningless"
  fi
else
  bad "nvidia-smi not found"
fi

echo
echo "== 5. model access =="
echo "     $MODEL is a GATED repo on Hugging Face."
if [[ -n "${HF_TOKEN:-}${HUGGING_FACE_HUB_TOKEN:-}" ]] || [[ -f "$HOME/.cache/huggingface/token" ]]; then
  ok "a Hugging Face token is present"
  if python - "$MODEL" <<'PY' 2>/dev/null
import sys
from huggingface_hub import HfApi
HfApi().model_info(sys.argv[1])
PY
  then ok "the token can read $MODEL"
  else bad "token present but cannot read $MODEL -- accept the licence at
       https://huggingface.co/$MODEL and wait for approval"
  fi
else
  bad "no Hugging Face token found"
  echo "     1. accept the licence at https://huggingface.co/$MODEL"
  echo "     2. pip install -U huggingface_hub && huggingface-cli login"
  echo "     Do this BEFORE booking GPU time; approval is not always instant."
fi

echo
echo "== 5b. network reachability =="
# Compute nodes on HPC clusters frequently have no egress. Both internet-
# dependent steps happen once and can be done from wherever egress exists,
# but only if you know before you start.
for host in huggingface.co query1.finance.yahoo.com; do
  if timeout 8 bash -c "cat < /dev/null > /dev/tcp/$host/443" 2>/dev/null; then
    ok "$host reachable"
  else
    warn "$host NOT reachable from here"
    case "$host" in
      huggingface.co) echo "     -> download the weights where egress exists, then set" ;
                      echo "        HF_HOME to a shared path and run offline with" ;
                      echo "        HF_HUB_OFFLINE=1" ;;
      *)              echo "     -> run 'make snapshot' where egress exists; the" ;
                      echo "        snapshot is a portable directory, copy it in" ;;
    esac
  fi
done

echo
echo "== 5c. port availability =="
PORT="${PORT:-8000}"
if timeout 3 bash -c "cat < /dev/null > /dev/tcp/127.0.0.1/$PORT" 2>/dev/null; then
  warn "port $PORT is already in use on this node"
  echo "     On a shared node pick your own: PORT=8137 make serve, and"
  echo "     LLM_BASE_URL=http://127.0.0.1:8137/v1 for the benchmark shell."
else
  ok "port $PORT free"
fi

echo
echo "== 6. disk for the weights =="
WEIGHTS_DIR="${HF_HUB_CACHE:-${HF_HOME:-$HOME/.cache/huggingface}}"
mkdir -p "$WEIGHTS_DIR" 2>/dev/null
echo "     weights cache: $WEIGHTS_DIR"
AVAIL=$(df -BG --output=avail "$WEIGHTS_DIR" 2>/dev/null | tail -1 | tr -dc '0-9')
if [[ -n "$AVAIL" && "$AVAIL" -ge 40 ]]; then
  ok "${AVAIL}G free in \$HOME (weights are ~16G, plus HF cache)"
else
  warn "only ${AVAIL:-?}G free in \$HOME; BF16 weights are ~16G"
  echo "     set HF_HOME to a larger volume if needed"
fi

echo
echo "== 7. harness self-test (no GPU, no network) =="
if python scripts/smoke_test.py >/tmp/b0-smoke.out 2>&1; then
  ok "$(grep -c '\[PASS\]' /tmp/b0-smoke.out) checks passed"
else
  bad "smoke test failed -- see /tmp/b0-smoke.out"
  tail -20 /tmp/b0-smoke.out | sed 's/^/     /'
fi

echo
if [[ "$FAIL" -eq 0 ]]; then
  echo "READY. Next:"
  echo "  make snapshot                                # once, needs internet"
  if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "  # inside this allocation, two panes of one tmux session:"
    echo "  tmux new -s b0"
    echo "  CUDA_VISIBLE_DEVICES=<free gpu> make serve   # pane 1"
    echo "  # ctrl-b \" to split, then in pane 2:"
    echo "  make calibrate && make bench && make report"
    echo "  # Do NOT let the allocation end between serve and bench."
  else
    echo "  tmux new -s vllm                             # server must outlive your shell"
    echo "  CUDA_VISIBLE_DEVICES=<free gpu> make serve"
    echo "  # detach: ctrl-b d, then in a second shell:"
    echo "  make calibrate && make bench && make report"
  fi
else
  echo "NOT READY -- fix the [FAIL] lines above before booking GPU time."
  exit 1
fi
