# Environment the vLLM server must be launched with.
# These are NOT cosmetic: VLLM_USE_FLASHINFER_SAMPLER selects which sampling
# kernel runs, so a server started without it is a different configuration
# from the one B0 was measured on. CC/CXX are needed because Triton compiles
# at startup and this image ships no system compiler.
export CC=$(ls "$HOME"/ccenv/bin/*-gcc 2>/dev/null | head -1)
export CXX=$(ls "$HOME"/ccenv/bin/*-g++ 2>/dev/null | head -1)
export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HUB_CACHE=/tmp/hf
[ -x "$CC" ] || echo "WARNING: no C compiler found at \$CC -- vLLM will fail in Triton"
