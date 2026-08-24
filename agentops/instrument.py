"""
Attach per-agent timing without editing the agents.

The handout's agents are plain Python classes. Rather than sprinkle timing
calls through code we want to keep close to upstream, we wrap their public
methods at import time from the runner. The agents stay byte-identical to the
handout apart from the documented patches, and per-agent wall and CPU time
come out anyway.

CPU time uses time.thread_time(), not time.process_time(): the bench runner
drives concurrent queries from a thread pool, and process CPU time would
count every other in-flight query's work as this agent's.
"""

from __future__ import annotations

import functools
import threading
import time
from typing import Any, Iterable

from . import trace
from .trace import Span

# Agent calls nest -- MetricsAgent.compute() calls PriceAgent.get_history() --
# so a span's wall time already contains its children's. Reporting inclusive
# times side by side would produce percentages that sum past 100%. Each frame
# accumulates the wall time of the children beneath it, and the difference is
# recorded as self_s, which IS additive.
_frames = threading.local()


def _push() -> None:
    stack = getattr(_frames, "stack", None)
    if stack is None:
        stack = _frames.stack = []
    stack.append(0.0)


def _pop(wall_s: float) -> float:
    stack = _frames.stack
    child_s = stack.pop()
    if stack:
        stack[-1] += wall_s          # charge this frame to its parent
    return max(0.0, wall_s - child_s)


def _wrap(agent_name: str, method_name: str, fn):
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any):
        with trace.agent(agent_name):
            t0 = time.perf_counter()
            c0 = time.thread_time()
            _push()
            err = None
            try:
                return fn(*args, **kwargs)
            except BaseException as exc:  # noqa: BLE001 - recorded then re-raised
                err = f"{type(exc).__name__}: {exc}"
                raise
            finally:
                _wall = time.perf_counter() - t0
                _self = _pop(_wall)
                trace.emit(Span(
                    run_id=trace.run_id(),
                    query_id=trace.current_query(),
                    agent=agent_name,
                    method=method_name,
                    wall_s=_wall,
                    self_s=_self,
                    cpu_s=time.thread_time() - c0,
                    error=err,
                    meta={"args": _describe(kwargs)},
                    ts=time.time(),
                ))
    wrapper.__agentops_wrapped__ = True  # type: ignore[attr-defined]
    return wrapper


def _describe(kwargs: dict) -> dict:
    """Small, non-bulky summary of the call arguments.

    Never stores the payload itself -- a returns[] array would balloon the
    trace file and slow the very thing we are measuring.
    """
    out = {}
    for k, v in kwargs.items():
        if isinstance(v, (int, float, str, bool)) or v is None:
            out[k] = v
        elif isinstance(v, dict):
            out[k] = f"dict[{len(v)}]"
        elif isinstance(v, (list, tuple)):
            out[k] = f"{type(v).__name__}[{len(v)}]"
        else:
            out[k] = type(v).__name__
    return out


def wrap_agent(cls: type, name: str | None = None) -> type:
    """Wrap every public callable on the class, in place."""
    agent_name = name or cls.__name__
    for attr in dir(cls):
        if attr.startswith("_"):
            continue
        fn = getattr(cls, attr)
        if not callable(fn) or getattr(fn, "__agentops_wrapped__", False):
            continue
        setattr(cls, attr, _wrap(agent_name, attr, fn))
    return cls


def wrap_agents(classes: Iterable[type]) -> None:
    for c in classes:
        wrap_agent(c)
