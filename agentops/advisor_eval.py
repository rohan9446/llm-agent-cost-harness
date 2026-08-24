"""
Advisor output quality, scored deterministically.

A2 asks the Advisor to write less. "Shorter is cheaper" is true by
construction, so without a counterweight the experiment cannot fail and
therefore cannot inform anything. This module is the counterweight.

Three gates, in increasing order of how much they would embarrass the system:

  TRUNCATION      finish_reason must be "stop", never "length". A briefing cut
                  mid-sentence is a defect regardless of what it saved. Already
                  captured in the trace, so this costs nothing.

  TOPIC COVERAGE  The handout's prompt asks for return, risk, diversification
                  and concentration. Brevity that silently drops a topic has
                  changed the deliverable, not compressed it.

  NUMERIC         Every figure in the briefing should be one the workflow
  GROUNDING       actually computed. This is the gate that matters most and the
                  one nobody usually builds: a model under pressure to be brief
                  can drop specifics, but it can also invent them, and an
                  invented Sharpe ratio in a confident two-sentence summary is
                  worse than a verbose accurate one.

All three are computed from artifacts already written to results.jsonl -- the
briefing text, the holdings, the per-ticker metrics and the portfolio risk
figures -- so any completed run can be scored after the fact, with no GPU and
no model in the loop.

Deliberately NOT here: any judgement of whether the prose is good. That would
need a model, which would make the quality gate itself a cost centre and put a
second nondeterministic component inside the measurement. These three gates are
mechanical, cheap and falsifiable, which is the whole point.
"""

from __future__ import annotations

import hashlib
import os
import re
from typing import Any


def scorer_sha256() -> str:
    """Checksum of THIS file, stamped into every score it produces.

    Learned the hard way. This module's number tokeniser had a bug -- "3-4"
    read as 3 and MINUS 4 -- which inflated the fabrication rate in any
    briefing containing a hyphenated range. It was fixed partway through the A2
    experiment, so some runs were scored before the fix and some after, and the
    resulting advisor_eval.json files were then compared to each other as
    though they meant the same thing. One of those comparisons failed a
    non-inferiority gate: A2-short rep1 reported 23.6% of briefings carrying an
    ungrounded figure against rep2's 2.8%, for the same prompt on the same
    workload.

    A quality score is a measurement, and a measurement without its instrument
    recorded is not comparable to another one. The manifest pins the model, the
    snapshot, the query set and the source tree. This pins the ruler.
    """
    try:
        with open(os.path.abspath(__file__), "rb") as fh:
            return hashlib.sha256(fh.read()).hexdigest()
    except OSError:
        return ""

# Topic -> the vocabulary that counts as covering it. Word families rather
# than exact strings, because the model paraphrases and a keyword miss would
# be scored as a content failure it did not commit.
_TOPICS: dict[str, tuple[str, ...]] = {
    "return": ("return", "annualized", "annualised", "gain", "performance",
               "appreciat", "yield"),
    "risk": ("volatil", "risk", "drawdown", "sharpe", "downside", "swing"),
    "diversification": ("diversif", "correlat", "spread across", "spread over",
                        "uncorrelated", "breadth"),
    "concentration": ("concentrat", "hhi", "largest position", "largest holding",
                      "dominat", "weighted toward", "weighted towards",
                      "top holding"),
}

# A number written in the text: 12.3%, 12%, -4.5%, 0.72, 1.12
#
# The negative lookbehind is load-bearing. Without it, "a 3-4 sentence view"
# tokenises as 3 and MINUS 4, and the minus-four gets scored as a fabricated
# figure -- twice per briefing, in every briefing containing a hyphenated
# range. That artifact put the measured grounding rate at 93.4% when the model
# had invented nothing at all, and it varied across the A2 arms purely because
# each arm's prompt contains a different number ("3-4 sentence", "2 sentence",
# "35 words"). A quality gate that moves with the instruction rather than with
# the output is measuring itself.
_NUM = re.compile(r"(?<![\d.])-?\d+(?:\.\d+)?%?")


def _truthy_numbers(holdings: dict, metrics: dict, risk: dict) -> list[float]:
    """Every figure the workflow actually computed, as a plain float.

    Percentages are stored twice -- as the fraction and as the percentage --
    because the prompt formats them with :.1% and the model may echo either
    form. Being generous here is deliberate: a false 'ungrounded' reading
    would manufacture a quality problem that does not exist.
    """
    vals: list[float] = []

    def add(x: Any, pct: bool = False) -> None:
        try:
            f = float(x)
        except (TypeError, ValueError):
            return
        vals.append(f)
        if pct:
            vals.append(f * 100.0)

    for w in (holdings or {}).values():
        add(w, pct=True)
    for m in (metrics or {}).values():
        if not isinstance(m, dict):
            continue
        add(m.get("annualized_return"), pct=True)
        add(m.get("annualized_volatility"), pct=True)
        add(m.get("sharpe"))
        add(m.get("max_drawdown"), pct=True)
    r = risk or {}
    add(r.get("portfolio_annualized_return"), pct=True)
    add(r.get("portfolio_annualized_volatility"), pct=True)
    add(r.get("portfolio_sharpe"))
    add(r.get("concentration_hhi"), pct=True)
    add(r.get("diversification_ratio"))
    top = r.get("top_holding") or {}
    add(top.get("weight"), pct=True)
    add(len(holdings or {}))          # "five holdings" is a true statement
    return vals


def _grounded(value: float, truth: list[float], tol: float = 0.15) -> bool:
    """Within rounding distance of something the workflow computed.

    The prompt formats to one decimal place, so an exact match is not the
    right test -- 18.0% in the prompt can legitimately appear as 18% in the
    prose. The tolerance is absolute rather than relative so that small
    values (Sharpe 0.72) are not held to an unreasonably tight bound.
    """
    return any(abs(value - t) <= tol for t in truth)


def score_briefing(text: str, holdings: dict, metrics: dict,
                   risk: dict) -> dict[str, Any]:
    """Deterministic quality score for one Advisor briefing."""
    text = text or ""
    low = text.lower()

    topics = {name: any(k in low for k in keys) for name, keys in _TOPICS.items()}

    truth = _truthy_numbers(holdings, metrics, risk)

    # Statistics and bare counts are scored separately. "3 of the five
    # holdings" is a true sentence whose 3 appears in no metric; holding it to
    # the same standard as a Sharpe ratio manufactures a fidelity problem out
    # of ordinary English. The claim worth making is about FIGURES -- returns,
    # volatilities, ratios, weights -- so that is what the headline rate covers.
    nums, ungrounded = [], []
    counts, counts_ungrounded = [], []
    for m in _NUM.finditer(text):
        raw = m.group(0)
        try:
            v = float(raw.rstrip("%"))
        except ValueError:
            continue
        is_stat = raw.endswith("%") or "." in raw
        if is_stat:
            nums.append(v)
            if not _grounded(v, truth):
                ungrounded.append(raw)
        else:
            counts.append(v)
            if not _grounded(v, truth):
                counts_ungrounded.append(raw)

    # Sentence count, for checking that a brevity instruction was obeyed at all
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s]

    named = [t for t in (holdings or {}) if re.search(
        rf"(?<![A-Za-z]){re.escape(t)}(?![A-Za-z])", text)]

    return {
        "chars": len(text),
        "words": len(text.split()),
        "sentences": len(sentences),
        "topics": topics,
        "topics_covered": sum(topics.values()),
        "all_topics": all(topics.values()),
        "n_numbers": len(nums),
        "n_ungrounded": len(ungrounded),
        "ungrounded": ungrounded[:6],
        "grounded_fraction": (
            (len(nums) - len(ungrounded)) / len(nums) if nums else None),
        "n_bare_integers": len(counts),
        "n_bare_integers_ungrounded": len(counts_ungrounded),
        "bare_integers_ungrounded": counts_ungrounded[:6],
        "n_holdings": len(holdings or {}),
        "n_holdings_named": len(named),
        "all_holdings_named": len(named) == len(holdings or {}),
    }


def score_run(results: list[dict], trace: list[dict] | None = None
              ) -> dict[str, Any]:
    """Aggregate advisor quality over a completed run."""
    rows = []
    for r in results:
        rows.append(score_briefing(
            r.get("summary") or "", r.get("holdings") or {},
            r.get("metrics") or {}, r.get("risk") or {}))
    if not rows:
        return {"n": 0}

    n = len(rows)

    def mean(key: str) -> float:
        vals = [x[key] for x in rows if x.get(key) is not None]
        return sum(vals) / len(vals) if vals else 0.0

    truncated = 0
    if trace is not None:
        truncated = sum(
            1 for t in trace
            if t.get("kind") == "llm" and t.get("agent") == "AdvisorAgent"
            and t.get("finish_reason") == "length"
            and not str(t.get("query_id", "")).startswith("warmup-"))

    with_nums = [x for x in rows if x["n_numbers"]]
    tot_nums = sum(x["n_numbers"] for x in rows)
    tot_ungrounded = sum(x["n_ungrounded"] for x in rows)
    return {
        "n": n,
        "_scorer_sha256": scorer_sha256(),
        "_gates": "truncation, topic coverage and numeric grounding -- all "
                  "mechanical, so a brevity gain can be priced against them",
        "truncated_briefings": truncated,
        "truncation_rate": truncated / n,
        "mean_words": mean("words"),
        "mean_sentences": mean("sentences"),
        "all_topics_rate": sum(1 for x in rows if x["all_topics"]) / n,
        "topic_rate": {
            t: sum(1 for x in rows if x["topics"][t]) / n for t in _TOPICS},
        "mean_numbers_per_briefing": mean("n_numbers"),
        "_numbers_note": "figures only (percentages and decimals). Bare "
                         "integers are counted separately: they are usually "
                         "holding counts and ordinals, not claims about the "
                         "portfolio's statistics.",
        "mean_bare_integers": mean("n_bare_integers"),
        "bare_integers_not_matching_a_metric": mean("n_bare_integers_ungrounded"),
        "briefings_with_an_ungrounded_number": (
            sum(1 for x in rows if x["n_ungrounded"]) / n),
        "grounded_fraction_mean": (
            sum(x["grounded_fraction"] for x in with_nums) / len(with_nums)
            if with_nums else None),
        # Pooled over FIGURES rather than averaged over briefings.
        #
        # The mean-of-fractions is length-sensitive by construction: in a
        # briefing with five figures one bad number costs 20 points, in one
        # with ten it costs 10. A2 changes briefing length on purpose, so the
        # headline gate runs on a statistic the treatment moves directly.
        # Reported alongside, not instead of -- the mean is what the gate was
        # registered against, and swapping the metric after seeing a failure is
        # how a pre-registered margin becomes decoration.
        "grounded_fraction_pooled": (
            (tot_nums - tot_ungrounded) / tot_nums if tot_nums else None),
        "_pooled_note": ("pooled over all figures in the run; the mean above "
                         "is over briefings and is sensitive to how many "
                         "figures each contains"),
        "n_figures_total": tot_nums,
        "n_figures_ungrounded": tot_ungrounded,
        "all_holdings_named_rate": (
            sum(1 for x in rows if x["all_holdings_named"]) / n),
        "worst_ungrounded": [
            {"ungrounded": x["ungrounded"], "n_numbers": x["n_numbers"]}
            for x in sorted(rows, key=lambda y: -y["n_ungrounded"])[:5]
            if x["n_ungrounded"]],
    }
