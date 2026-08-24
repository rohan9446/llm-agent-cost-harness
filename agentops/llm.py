"""
The single door to the model.

Both the parser and the advisor call through here, so there is exactly one
place that knows about vLLM, one place that records tokens and timing, and
one place that can be asserted against afterwards.

Streaming is always on internally, even though callers get a plain string
back. That is how TTFT is captured, and TTFT is what separates prefill cost
from decode cost -- the central question of the A2 experiment.

There is deliberately no fallback. The handout's advisor swallows a failed
model call and returns a templated summary, which during a measured run reads
as a successful free query. Here a failure raises, the runner records it, and
the validity check refuses to report the run.
"""

from __future__ import annotations

import os
import random
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any

from . import trace
from .trace import LLMCall


class LLMError(RuntimeError):
    pass


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cached_prompt_tokens: int
    latency_s: float
    ttft_s: float | None
    tpot_s: float | None
    model: str
    finish_reason: str | None


# --------------------------------------------------------------------------
# call accounting, for the validity assertions
# --------------------------------------------------------------------------

class _Counter:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.calls = 0
        self.failures = 0
        self.by_agent: dict[str, int] = {}
        # What we asked for. NOT proof of what the server ran -- that comes
        # from probing /v1/models before the run. Named accordingly so the
        # distinction cannot be lost downstream.
        self.requested_models: set[str] = set()

    def record(self, agent: str, model: str, ok: bool) -> None:
        with self._lock:
            self.calls += 1
            if not ok:
                self.failures += 1
            self.by_agent[agent] = self.by_agent.get(agent, 0) + 1
            self.requested_models.add(model)

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "llm_calls": self.calls,
                "llm_failures": self.failures,
                "llm_calls_by_agent": dict(self.by_agent),
                "requested_models": sorted(self.requested_models),
            }

    def reset(self) -> None:
        with self._lock:
            self.calls = 0
            self.failures = 0
            self.by_agent.clear()
            self.requested_models.clear()


COUNTER = _Counter()


def max_attempts_setting() -> int:
    """Attempts per model call, first try included.

    LLM_MAX_RETRIES is honoured as a deprecated alias so an existing shell
    profile keeps working; whichever one is set, the effective number goes into
    the manifest so the run records what it actually did rather than what its
    environment was named.
    """
    v = os.environ.get("LLM_MAX_ATTEMPTS") or os.environ.get("LLM_MAX_RETRIES")
    try:
        return max(1, int(v)) if v else 3
    except ValueError:
        return 3


def effective_generation_config() -> dict[str, Any]:
    """Every sampling knob that could move a number, as actually resolved.

    Recorded rather than assumed. The manifest already carried max_tokens and
    temperature for the two agents; this adds the rest -- the seed, the attempt
    count, the base URL, the resolved model name -- so the generation config in
    the artifact is the whole of it and not the part somebody remembered to
    copy in.
    """
    seed = os.environ.get("LLM_SEED")
    return {
        "model": model_name(),
        "base_url": os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        "seed": int(seed) if seed else None,
        "max_attempts": max_attempts_setting(),
        "advisor_max_tokens": int(os.environ.get("ADVISOR_MAX_TOKENS", "400")),
        "advisor_temperature": float(os.environ.get("ADVISOR_TEMPERATURE", "0.2")),
        "advisor_style": os.environ.get("ADVISOR_STYLE", "handout"),
        "parser_max_tokens": int(os.environ.get("PARSER_MAX_TOKENS", "200")),
        "parser_temperature": float(os.environ.get("PARSER_TEMPERATURE", "0.0")),
        "_seed_note": (
            "a fixed seed fixes sampling, not floating-point reduction order: "
            "vLLM batches continuously, so the same prompt at a different "
            "batch size can still produce different logits in the last bits"),
    }


# --------------------------------------------------------------------------
# client
# --------------------------------------------------------------------------

_local = threading.local()


def _client():
    c = getattr(_local, "client", None)
    if c is not None:
        return c
    from openai import OpenAI
    c = OpenAI(
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        api_key=os.environ.get("LLM_API_KEY", "EMPTY"),
        timeout=float(os.environ.get("LLM_TIMEOUT_S", "180")),
        max_retries=0,          # retries are ours, so each attempt is traced
    )
    _local.client = c
    return c


def model_name() -> str:
    return os.environ.get("LLM_MODEL", "meta-llama/Llama-3.1-8B-Instruct")


_RETRYABLE = ("rate limit", "429", "timeout", "timed out", "overloaded",
              "503", "502", "500", "connection", "temporarily")


def _retryable(exc: Exception) -> bool:
    s = f"{type(exc).__name__} {exc}".lower()
    return any(h in s for h in _RETRYABLE)


def chat(
    messages: list[dict],
    *,
    max_tokens: int,
    temperature: float = 0.0,
    seed: int | None = None,
    stop: list[str] | None = None,
    max_attempts: int | None = None,
    tag: str = "",
) -> LLMResult:
    """One model call. Raises LLMError rather than degrading.

    NAMING, because this was wrong and the wrong name was load-bearing.
    The loop below is `for attempt in range(1, N + 1)`, so N is the number of
    ATTEMPTS -- the first try plus at most N-1 retries. The knob was called
    LLM_MAX_RETRIES, which reads as "3 retries after the first try", i.e. 4
    attempts. Every run so far was made with 3 attempts while the manifest
    implied 4. Nothing measured is wrong -- the setting was constant across all
    arms -- but a reproduction driven from the recorded name would have used a
    different value than the one that produced these numbers.

    LLM_MAX_ATTEMPTS is the name now. LLM_MAX_RETRIES is still read, so an old
    shell script does not silently change behaviour, and the effective value is
    recorded in the manifest either way.
    """
    model = model_name()
    attempts = max_attempts if max_attempts is not None else max_attempts_setting()
    if seed is None:
        env_seed = os.environ.get("LLM_SEED")
        seed = int(env_seed) if env_seed else None

    call_id = uuid.uuid4().hex[:16]
    agent = trace.current_agent()
    last: Exception | None = None

    for attempt in range(1, attempts + 1):
        rec = LLMCall(
            run_id=trace.run_id(), query_id=trace.current_query(),
            agent=agent, call_id=call_id, model=model, attempt=attempt,
            ts=time.time(),
            meta={"max_tokens": max_tokens, "temperature": temperature,
                  "seed": seed, "tag": tag},
        )
        t0 = time.perf_counter()
        try:
            result = _stream(messages, model, max_tokens, temperature,
                             seed, stop, rec, t0)
            rec.latency_s = result.latency_s
            trace.emit(rec)
            COUNTER.record(agent, model, ok=True)
            return result
        except Exception as exc:  # noqa: BLE001
            last = exc
            rec.latency_s = time.perf_counter() - t0
            rec.error = f"{type(exc).__name__}: {exc}"
            trace.emit(rec)
            COUNTER.record(agent, model, ok=False)
            if attempt == attempts or not _retryable(exc):
                raise LLMError(f"{agent}: model call failed: {exc}") from exc
            time.sleep(min(2 ** attempt, 20) * (0.5 + random.random()))

    raise LLMError(f"{agent}: model call failed: {last}")


def _stream(messages, model, max_tokens, temperature, seed, stop, rec, t0) -> LLMResult:
    kwargs: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        kwargs["seed"] = seed
    if stop:
        kwargs["stop"] = stop

    parts: list[str] = []
    usage = None
    finish = None
    ttft = None

    for chunk in _client().chat.completions.create(**kwargs):
        if getattr(chunk, "usage", None):
            usage = chunk.usage
        if not getattr(chunk, "choices", None):
            continue
        ch = chunk.choices[0]
        if ch.finish_reason:
            finish = ch.finish_reason
        delta = getattr(ch, "delta", None)
        if delta is not None and getattr(delta, "content", None):
            if ttft is None:
                ttft = time.perf_counter() - t0
            parts.append(delta.content)

    latency = time.perf_counter() - t0
    prompt_tokens = int(getattr(usage, "prompt_tokens", 0) or 0)
    completion_tokens = int(getattr(usage, "completion_tokens", 0) or 0)

    cached = 0
    ptd = getattr(usage, "prompt_tokens_details", None)
    if ptd is not None:
        cached = int(getattr(ptd, "cached_tokens", 0) or 0)

    # Decode rate excludes the first token, which is the tail of prefill.
    tpot = None
    if ttft is not None and completion_tokens > 1:
        tpot = (latency - ttft) / (completion_tokens - 1)

    rec.prompt_tokens = prompt_tokens
    rec.completion_tokens = completion_tokens
    rec.cached_prompt_tokens = cached
    rec.ttft_s = ttft
    rec.tpot_s = tpot
    rec.finish_reason = finish

    if usage is None:
        raise LLMError(
            "server returned no usage block; token accounting would be "
            "invented rather than measured"
        )

    return LLMResult(
        text="".join(parts),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_prompt_tokens=cached,
        latency_s=latency,
        ttft_s=ttft,
        tpot_s=tpot,
        model=model,
        finish_reason=finish,
    )
