"""
Cost model.

One rule underneath everything here: the total is measured, the split is
estimated, and the two are never confused.

  TOTAL   allocated GPU time x rental-equivalent rate. Wall-clock of the
          serving window times the number of GPUs held. Defensible because it
          is a measurement, and it does not care how the work was distributed.

  SPLIT   the token-work-weighted share of that measured total attributable to
          each LLM stage. Estimated by fitting per-request and per-token weights
          from the run's own timings, then dividing the measured total in
          proportion. It answers "which stage should I optimize". It is NOT a
          literal "% of system cost per agent" -- the deterministic agents
          consume CPU that this split does not see, and the weights are a fit,
          not a measurement. The report labels it accordingly.

The rate is called rental-equivalent throughout, because the lab GPUs are not
rented. Presenting an opportunity cost as an incurred cost would be a lie that
costs nothing to avoid.

What this module deliberately does NOT do is multiply per-request latency by a
GPU-hour rate. Under continuous batching many requests share the GPU at once,
so summing per-request wall time overcounts by roughly the batch factor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence


# --------------------------------------------------------------------------
# the measured total
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class GpuRate:
    """Rental-equivalent price of the hardware under test.

    RTX PRO 6000 Blackwell is not a mainstream cloud SKU, so the default here
    is a placeholder that MUST be replaced with a rate you can cite. Record
    where the number came from; a cost report without a sourced, dated rate is
    not reproducible.
    """
    n_gpus: int = 1
    usd_per_gpu_hour: float = 1.60
    source: str = "PLACEHOLDER -- replace with a citable rate before reporting"
    as_of: str = ""

    @property
    def usd_per_second(self) -> float:
        return self.n_gpus * self.usd_per_gpu_hour / 3600.0


@dataclass
class AllocatedCost:
    wall_s: float
    n_queries: int
    rate: GpuRate

    @property
    def total_usd(self) -> float:
        return self.wall_s * self.rate.usd_per_second

    @property
    def per_query_usd(self) -> float:
        return self.total_usd / self.n_queries if self.n_queries else float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {
            "wall_s": self.wall_s,
            "n_queries": self.n_queries,
            "n_gpus": self.rate.n_gpus,
            "usd_per_gpu_hour": self.rate.usd_per_gpu_hour,
            "rate_source": self.rate.source,
            "total_usd": self.total_usd,
            "cost_per_query_usd": self.per_query_usd,
            "basis": "allocated GPU time x rental-equivalent rate",
        }


# --------------------------------------------------------------------------
# the estimated split
# --------------------------------------------------------------------------

@dataclass
class TokenWeights:
    """Fitted ATTRIBUTION WEIGHTS -- deliberately not called GPU-seconds/token.

    Two separate regressions, because one regression on total latency conflates
    things that behave differently:

        TTFT        ~ alpha + a * prompt_tokens
        latency-TTFT ~ beta  + b * (completion_tokens - 1)

    The intercepts matter. Even at concurrency 1 a request carries HTTP
    overhead, scheduler dispatch, a fixed per-forward-pass cost and stream
    teardown. Folding all of that into a per-token slope, as a single
    `latency ~ a*prefill + b*decode` fit does, inflates both coefficients and
    inflates them unequally -- the shorter the request, the worse.

    So `a` and `b` are the marginal cost of one more token, `alpha` and `beta`
    are what a request costs before any tokens, and the pair is used to divide
    a MEASURED total. They are not a claim about what the GPU spent per token.

    The b/a ratio is still the quantitative form of "does decode dominate",
    which is the hypothesis A2 exists to test -- it is just now a ratio of
    marginal slopes rather than of two contaminated averages.

    Valid only inside the concurrency regime fitted in. Above C=1 the measured
    times also contain queueing delay. Fit on a C=1 calibration run and carry
    the coefficients across regimes with --weights-from.
    """
    a_prefill: float
    b_decode: float
    alpha_request_s: float = 0.0     # fixed per-request cost before prefill
    beta_decode_s: float = 0.0       # fixed cost of entering decode
    r2_prefill: float = float("nan")
    r2_decode: float = float("nan")
    r2: float = float("nan")         # kept for backwards compatibility
    n_calls: int = 0
    concurrency: int = 1
    note: str = ""

    @property
    def decode_ratio(self) -> float:
        return self.b_decode / self.a_prefill if self.a_prefill > 0 else float("inf")

    def weight(self, prompt_tokens: int, completion_tokens: int,
               cached_tokens: int = 0, include_fixed: bool = True) -> float:
        """Attribution weight of ONE request.

        The fixed terms are included. Fitting alpha and beta specifically to
        keep per-request overhead out of the slopes, and then dropping them
        when dividing the bill, would make that overhead vanish from the split
        entirely -- it would be charged to nobody. At B0 the parser and the
        advisor each make one request per query, so the fixed cost divides
        evenly between them, and excluding it would overstate the advisor's
        share by attributing the parser's request overhead to token volume the
        parser does not have.

        A prefix-cache hit skips the prefill matmul but still costs the
        attention read over the cached blocks during decode. Charged at 5% of a
        fresh prefill token: a stated assumption, not a measurement. S3a is the
        experiment that would replace it.
        """
        cached = min(cached_tokens, prompt_tokens)
        fresh = prompt_tokens - cached
        w = (fresh * self.a_prefill
             + cached * self.a_prefill * 0.05
             + max(0, completion_tokens - 1) * self.b_decode)
        if include_fixed:
            w += self.alpha_request_s
            if completion_tokens > 0:
                w += self.beta_decode_s
        return w

    def as_dict(self) -> dict[str, Any]:
        return {
            "_kind": "fitted attribution weights, NOT measured GPU-seconds/token",
            "marginal_s_per_prefill_token": self.a_prefill,
            "marginal_s_per_decode_token": self.b_decode,
            "fixed_request_overhead_s": self.alpha_request_s,
            "fixed_decode_entry_s": self.beta_decode_s,
            "decode_to_prefill_ratio": self.decode_ratio,
            "r2_prefill_fit": self.r2_prefill,
            "r2_decode_fit": self.r2_decode,
            "n_calls": self.n_calls,
            "fitted_at_concurrency": self.concurrency,
            "note": self.note,
        }


def fit_token_weights(calls: Sequence[dict], concurrency: int = 1) -> TokenWeights:
    """Two regressions: prefill against TTFT, decode against the remainder.

    Requires TTFT, which is why every call streams even though callers get a
    plain string back. Without TTFT the two phases cannot be separated and the
    weights collapse to the contaminated single fit this replaced.
    """
    import numpy as np

    rows = [
        c for c in calls
        if not c.get("error")
        and (c.get("prompt_tokens") or 0) > 0
        and (c.get("latency_s") or 0) > 0
        and c.get("ttft_s") is not None
    ]
    if len(rows) < 8:
        raise ValueError(
            f"only {len(rows)} usable calls with TTFT; refusing to fit "
            f"attribution weights on that little data. Run the C=1 "
            f"calibration first."
        )

    def ols(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float]:
        """Slope, intercept, R^2 -- intercept kept, not forced through zero."""
        A = np.column_stack([x, np.ones_like(x)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        slope, intercept = float(coef[0]), float(coef[1])
        pred = A @ coef
        ss_res = float(((y - pred) ** 2).sum())
        ss_tot = float(((y - y.mean()) ** 2).sum())
        r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        return slope, intercept, r2

    fresh = np.array([c["prompt_tokens"] - c.get("cached_prompt_tokens", 0)
                      for c in rows], dtype=float)
    ttft = np.array([c["ttft_s"] for c in rows], dtype=float)
    a, alpha, r2p = ols(fresh, ttft)

    decode_rows = [c for c in rows if (c.get("completion_tokens") or 0) > 1]
    if len(decode_rows) >= 8:
        out_tok = np.array([c["completion_tokens"] - 1 for c in decode_rows], dtype=float)
        dec_s = np.array([c["latency_s"] - c["ttft_s"] for c in decode_rows], dtype=float)
        b, beta, r2d = ols(out_tok, dec_s)
    else:
        b, beta, r2d = 0.0, 0.0, float("nan")

    note = ""
    if concurrency > 1:
        note = (f"fitted at concurrency {concurrency}; TTFT and decode time both "
                f"include queueing delay, so these weights are contaminated -- "
                f"prefer --weights-from a C=1 run")

    return TokenWeights(
        a_prefill=max(a, 1e-12), b_decode=max(b, 1e-12),
        alpha_request_s=max(alpha, 0.0), beta_decode_s=max(beta, 0.0),
        r2_prefill=r2p, r2_decode=r2d, r2=r2p,
        n_calls=len(rows), concurrency=concurrency, note=note,
    )


@dataclass
class StageSplit:
    """One LLM stage's share. Read as token-work-weighted share of the measured
    GPU allocation, not as that stage's total cost to the system."""
    stage: str
    calls: int
    prompt_tokens: int
    completion_tokens: int
    cached_tokens: int
    weight: float
    share: float
    attributed_usd: float
    usd_per_query: float


def attribute(
    calls: Iterable[dict],
    weights: TokenWeights,
    total: AllocatedCost,
) -> list[StageSplit]:
    """Divide the measured total between stages, in proportion to token work."""
    agg: dict[str, dict[str, float]] = {}
    for c in calls:
        if c.get("error"):
            # Failed attempts still consumed GPU time -- the prefill ran before
            # the failure. Attributed, not dropped.
            pass
        s = agg.setdefault(c.get("agent") or "unattributed",
                           {"calls": 0, "p": 0, "d": 0, "cached": 0, "w": 0.0})
        s["calls"] += 1
        p = c.get("prompt_tokens", 0) or 0
        d = c.get("completion_tokens", 0) or 0
        k = c.get("cached_prompt_tokens", 0) or 0
        s["p"] += p
        s["d"] += d
        s["cached"] += k
        s["w"] += weights.weight(p, d, k)

    total_w = sum(v["w"] for v in agg.values()) or 1.0
    out = []
    for stage, v in sorted(agg.items(), key=lambda kv: -kv[1]["w"]):
        share = v["w"] / total_w
        out.append(StageSplit(
            stage=stage,
            calls=int(v["calls"]),
            prompt_tokens=int(v["p"]),
            completion_tokens=int(v["d"]),
            cached_tokens=int(v["cached"]),
            weight=v["w"],
            share=share,
            attributed_usd=share * total.total_usd,
            usd_per_query=share * total.per_query_usd,
        ))
    return out


# --------------------------------------------------------------------------
# the modelled deployment view, kept separate on purpose
# --------------------------------------------------------------------------

@dataclass
class DeploymentModel:
    """What the original global_controller.yaml would bill on AWS.

    Never merged with measured cost. This is arithmetic over their config,
    offered as context for the idle-infrastructure finding, and labelled as a
    model everywhere it appears.
    """
    instance_type: str = "t3.micro"
    usd_per_instance_hour: float = 0.0104
    n_instances: int = 5
    source: str = "AWS on-demand, us-east-1 -- record the date you looked it up"

    def monthly_idle_usd(self) -> float:
        return self.n_instances * self.usd_per_instance_hour * 24 * 30

    def per_query_usd(self, queries_per_month: int) -> float:
        if queries_per_month <= 0:
            return float("nan")
        return self.monthly_idle_usd() / queries_per_month

    def as_dict(self) -> dict[str, Any]:
        return {
            "kind": "modelled, not measured",
            "instance_type": self.instance_type,
            "n_instances": self.n_instances,
            "usd_per_instance_hour": self.usd_per_instance_hour,
            "monthly_always_on_usd": self.monthly_idle_usd(),
            "source": self.source,
        }
