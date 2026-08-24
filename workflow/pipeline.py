"""
One callable for one natural-language query.

The handout's entry point takes structured arguments, and the corpus is
natural language, so something has to compose the parser with it. Doing that
here rather than inside portfolio_workflow.py keeps the handout file at three
mechanical fixes and nothing else, which is what makes PATCHES.md short enough
to be worth reading.

    query text
       -> ParserAgent.parse       (LLM at B0, cascade at A1)
       -> normalize               (policy: our default lookback)
       -> portfolio_workflow.main (the handout, unchanged)
       -> result dict
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
_AGENTS = os.path.join(_HERE, "portfolio", "agents")
_WORKFLOWS = os.path.join(_HERE, "portfolio", "workflows")
for _p in (_AGENTS, _WORKFLOWS):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from advisor_agent import AdvisorAgent          # noqa: E402
from metrics_agent import MetricsAgent          # noqa: E402
from parser_agent import ParserAgent, normalize  # noqa: E402
from price_agent import PriceAgent              # noqa: E402
from risk_agent import RiskAgent                # noqa: E402

import portfolio_workflow                        # noqa: E402

from parser_cascade import CascadeParserAgent   # noqa: E402

AGENT_CLASSES = [ParserAgent, PriceAgent, MetricsAgent, RiskAgent, AdvisorAgent]


def instrument() -> None:
    """Attach per-agent timing. Idempotent; call once before any query.

    CascadeParserAgent is wrapped too, and because its Tier 2 path calls the
    wrapped ParserAgent, the nested spans separate cleanly: the cascade's
    self_s is the deterministic work, ParserAgent's is the model call it could
    not avoid. That split is the A1 measurement.
    """
    from agentops.instrument import wrap_agents
    wrap_agents(AGENT_CLASSES + [CascadeParserAgent])


class PipelineError(RuntimeError):
    """A downstream failure that still had a successful parse.

    The parse is attached because a query can fail *because of* the parse --
    a hallucinated ticker that is not in the snapshot, say -- and scoring the
    parser only over successful queries would hide exactly those cases, making
    parser accuracy look best where it is worst.
    """

    def __init__(self, message: str, parsed: dict | None = None) -> None:
        super().__init__(message)
        self.parsed = parsed


class Pipeline:
    """B0 uses the LLM parser; A1 swaps in the cascade and changes nothing else.

    The swap is one object, so every other measured quantity -- snapshot,
    advisor, agents, cost model, validity checks -- is held constant by
    construction rather than by discipline.
    """

    def __init__(self, default_lookback_days: int = 365,
                 parser: str = "llm") -> None:
        if parser == "cascade":
            self.parser = CascadeParserAgent()
        elif parser == "llm":
            self.parser = ParserAgent()
        else:
            raise ValueError(f"unknown parser {parser!r}; use 'llm' or 'cascade'")
        self.parser_kind = parser
        self.default_lookback_days = default_lookback_days

    def reset_parser_stats(self) -> None:
        """Zero the cascade tier counters; no-op for the B0 parser."""
        fn = getattr(self.parser, "reset_stats", None)
        if fn:
            fn()

    def parser_stats(self) -> dict:
        """Tier 1 / Tier 2 counts, empty for the B0 parser."""
        return dict(getattr(self.parser, "stats", {}) or {})

    def run(self, query_text: str) -> dict:
        parsed = self.parser.parse(query=query_text)
        args = normalize(parsed, self.default_lookback_days)
        try:
            result = portfolio_workflow.main(**args)
        except Exception as exc:
            raise PipelineError(f"{type(exc).__name__}: {exc}", parsed=parsed) from exc
        # Carry the parse through so the evaluation can score it later without
        # re-running anything.
        result["_parsed"] = parsed
        result["_lookback_source"] = (
            "stated" if parsed["lookback_days"] is not None else "policy_default"
        )
        return result
