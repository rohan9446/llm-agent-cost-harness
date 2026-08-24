"""
Tracing for the colocated portfolio workflow.

Three record kinds land in one JSONL file:

  span   one agent method call -- wall time, thread CPU time, agent name
  llm    one model call -- tokens, TTFT, latency, attempt number
  query  one end-to-end query -- wall time, outcome

Attribution rides on context variables rather than on arguments, so the
workflow and the agents never learn they are being measured. The bench runner
drives queries from a thread pool; contextvars are per-task and thread CPU
time is per-thread, so both stay correct under concurrency.
"""

from __future__ import annotations

import contextlib
import contextvars
import json
import os
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Iterator

_run_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("run_id", default=None)
_query_id: contextvars.ContextVar[str | None] = contextvars.ContextVar("query_id", default=None)
_agents: contextvars.ContextVar[tuple[str, ...]] = contextvars.ContextVar("agents", default=())


# --------------------------------------------------------------------------
# records
# --------------------------------------------------------------------------

@dataclass
class Span:
    kind: str = "span"
    run_id: str | None = None
    query_id: str | None = None
    agent: str = ""
    method: str = ""
    wall_s: float = 0.0      # inclusive: contains nested agent calls
    self_s: float = 0.0      # exclusive: this agent's own work, additive
    cpu_s: float = 0.0
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


@dataclass
class LLMCall:
    kind: str = "llm"
    run_id: str | None = None
    query_id: str | None = None
    agent: str = ""
    call_id: str = ""
    model: str = ""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_prompt_tokens: int = 0

    latency_s: float = 0.0
    ttft_s: float | None = None
    tpot_s: float | None = None      # per-output-token time, decode only

    attempt: int = 1
    finish_reason: str | None = None
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


@dataclass
class QueryRecord:
    kind: str = "query"
    run_id: str | None = None
    query_id: str | None = None
    wall_s: float = 0.0
    ok: bool = True
    error: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    ts: float = 0.0


# --------------------------------------------------------------------------
# writer
# --------------------------------------------------------------------------

class _Writer:
    def __init__(self) -> None:
        self._fh = None
        self._lock = threading.Lock()
        self.path: str | None = None

    def open(self, path: str) -> None:
        self.close()
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        self._fh = open(path, "a", encoding="utf-8")
        self.path = path

    def write(self, obj: Any) -> None:
        if self._fh is None:
            return
        row = asdict(obj) if hasattr(obj, "__dataclass_fields__") else obj
        line = json.dumps(row, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def flush(self) -> None:
        if self._fh:
            with self._lock:
                self._fh.flush()

    def close(self) -> None:
        if self._fh:
            with self._lock:
                self._fh.close()
                self._fh = None


W = _Writer()


def start_run(path: str, run_id: str | None = None) -> str:
    W.open(path)
    rid = run_id or uuid.uuid4().hex[:12]
    _run_id.set(rid)
    return rid


def end_run() -> None:
    W.flush()
    W.close()


def run_id() -> str | None:
    return _run_id.get()


def emit(rec: Any) -> None:
    W.write(rec)


# --------------------------------------------------------------------------
# context
# --------------------------------------------------------------------------

@contextlib.contextmanager
def query(query_id: str, run_id_value: str | None = None, **meta: Any) -> Iterator[None]:
    """Mark everything inside as belonging to one end-user query.

    run_id is re-set here as well as in start_run, because a thread-pool
    worker starts from a copy of the submitting context and would otherwise
    inherit whatever the pool thread last held.
    """
    if run_id_value is not None:
        _run_id.set(run_id_value)
    tok = _query_id.set(query_id)
    t0 = time.perf_counter()
    err = None
    try:
        yield
    except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
        err = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        _query_id.reset(tok)
        emit(QueryRecord(
            run_id=_run_id.get(), query_id=query_id,
            wall_s=time.perf_counter() - t0,
            ok=err is None, error=err, meta=meta, ts=time.time(),
        ))


@contextlib.contextmanager
def agent(name: str) -> Iterator[None]:
    stack = _agents.get()
    tok = _agents.set(stack + (name,))
    try:
        yield
    finally:
        _agents.reset(tok)


def current_agent() -> str:
    s = _agents.get()
    return s[-1] if s else "workflow"


def current_query() -> str | None:
    return _query_id.get()
