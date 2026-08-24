# Measuring, then reducing, the cost of a multi-agent LLM workflow

A financial-advisory workflow — natural-language query in, portfolio briefing
out — served by Llama-3.1-8B on vLLM. This repository is the harness that
measured it and the two optimizations that came out of the measurement.

```
                 cost/query    q/s    workflow    ticker    briefing   invented
                              (C=8)   failures    accuracy   topics     figures
  B0  baseline    $0.000113   3.30      3.4%        95.3%     99.8%      6.6%
  A1  cascade     $0.000081   4.77      0.0%       100.0%     99.9%      6.8%   -28.4%
  A2  + brevity   $0.000069   5.57      0.0%       100.0%     99.3%      2.6%   -38.7%
```

1,000 queries per run, single A100 80GB PCIe, rental-equivalent rate cited
below. **Cheaper and more correct at the same time** — which is not the usual
shape of a cost optimization, and is the reason the quality gates matter more
than the cost numbers.

Two notes on how to read that table:

**Cost is per ATTEMPTED query**, and that is the conservative choice. Only B0
has failures, so only B0's denominator is in question — A1 and A2 answer
everything and their two figures are identical. Dividing B0's GPU time by its
966 successes instead of its 1,000 attempts raises its cost to $0.000117 and
*enlarges* every saving in the table (A1 becomes 30.8% rather than 28.4%).

I published that larger figure for about an hour. It is defensible — you did
pay for answers you did not get — but I reached it by picking the denominator
that made my own result look better, having just written three paragraphs
elsewhere in this repository about not doing that. The failure rate already
has its own column; putting it in the cost as well counts it twice and turns
"cheaper **and** more correct" into one fact reported as two. Per-attempted
makes the claim harder to support, so per-attempted is the headline.
`report.json` carries both, with `_headline_denominator` naming which is which.

**"Invented figures" is the share of briefings containing at least one number
the workflow never computed.** A2 asks the model to write less and it invents
*less* — 2.6% against the baseline's 6.6%, with barely fewer figures per
briefing (5.4 vs 5.9). That was not the expected result and it is not a
throwaway: brevity was supposed to cost quality, and the gate existed to price
that cost. It found the opposite sign.

---

## Reproducibility note — read this first

**This repository is not standalone, deliberately.** It excludes the workflow
source and query corpus supplied with the exercise, because publishing an
employer's take-home would leak it for every future candidate and it is not
mine to republish.

Missing, and required to run anything:

```
workflow/portfolio/agents/{price,metrics,risk,advisor}_agent.py
workflow/portfolio/workflows/portfolio_workflow.py
data/queries.json
```

`PATCHES.md` documents every change made to those files without reproducing
them. `parser_agent.py` and `parser_cascade.py` **are** here — both are new
stages written for this project, not part of the handout.

Also excluded: the frozen price snapshot (provider data — `MANIFEST.json` with
its per-ticker checksums is kept, so a rebuilt snapshot can be *proved*
identical to the one these results came from), the vLLM logs, and the raw
per-query traces.

---

## What the baseline actually did

The supplied workflow makes two model calls per query: a parser that turns the
sentence into `{holdings, lookback_days}`, and an advisor that writes the
briefing. Three deterministic stages sit between them.

Three things about B0 were only visible because the harness looked for them.

**It can complete a 1,000-query sweep without ever contacting a model.**
`import boto3` sits inside a `try` that falls back to a hardcoded summary. Every
query returns, every query looks successful. That single fact is why this
repository has 15 validity gates instead of a timer — see `METHODS.md`.

**Its real error rate is roughly double its failure rate.**

```
operational success   966/1000   96.6%   the workflow returned something
semantic success      930/1000   93.0%   ...and it was the right portfolio
```

34 queries failed loudly. **36 more succeeded while analysing a portfolio
nobody asked for** — the parser resolved a company to a plausible but wrong
ticker that happened to exist in the snapshot. Those are worse than failures:
a crash says "I don't know", a wrong portfolio says something false with
correct-looking numbers attached. They are visible only because parser accuracy
is scored independently of whether the workflow returned.

**Whether an error is loud or silent is an accident.** Three companies account
for most parser errors, at ~16% each. Two resolve to tickers outside the
snapshot and crash; one resolves to a ticker that *is* in the snapshot and
silently analyses the wrong company. Same mistake, opposite visibility,
decided by which symbols happened to be frozen.

---

## The cost model, and why it is believable

**Cost per query is not latency times a GPU-hour rate.** Under continuous
batching many requests share the GPU, so summing per-request wall time
overcounts by roughly the batch factor.

What is *measured* is the window: `GPUs × wall-clock × $/GPU-hour`.
What is *estimated* is each query's share, from two regressions fitted on a
separate C=1 calibration run:

```
TTFT           ~ alpha + a * prompt_tokens
latency - TTFT ~ beta  + b * (completion_tokens - 1)
```

Two fits rather than one, so HTTP overhead and scheduler dispatch land in the
intercepts instead of contaminating the per-token slopes. They are called
**attribution weights**, never GPU-seconds per token: they divide a measured
total, they are not a claim about what the GPU spent.

**The model made a falsifiable prediction twice and was right twice.** It
assigned the parser 28.3% of the LLM allocation. A1 removes the parser
entirely, so it predicted A1's cost before A1 existed:

| | predicted | measured | error |
|---|---|---|---|
| n=100 | $0.0000842 | $0.000083 | 0.7% |
| n=1000 | $0.0000810 | $0.000081 | 0.1% |

That moves the per-agent split from "a defensible way to divide a total" to a
validated predictor.

---

## A1 — replace the parser with a cascade

```
Tier 1   deterministic scan    no model call, cannot emit an unknown ticker
Tier 2   the B0 LLM parser     unchanged, for anything Tier 1 declines
```

**Company names come from the market-data provider, never from the corpus.**
A lookup table transcribed from the test queries would report ~100% coverage by
construction and prove nothing. Tier 1 normalises the provider's official name
and matches against that; the ticker *universe* is legitimately known (it is the
frozen snapshot — the assets the system can price).

Result on the supplied corpus: **1000/1000 resolved without a model call,
zero disagreements** with the derived labels. LLM calls per 1,000 queries fall
from 1,966 to 1,000; prompt tokens fall 53%.

**Declining is the feature.** Tier 1 refuses on an unrecognised name, an
unconvertible time expression, a percentage count that doesn't match the
holdings, or weights that don't sum to a portfolio. So the worst case of a
corpus-shaped fast path is *cost* reverting to baseline, not correctness
degrading.

### A1 broke under distribution shift, and that is the interesting part

100% on the corpus means little when the corpus is what you debugged against.
A 91-query robustness set — companies outside the universe, word-boundary
traps, ambiguous weights, unreadable time windows — measured **routing**
rather than coverage, weighting the two error types by consequence:

```
false accept   resolved something it should have handed off   unbounded
false decline  handed off something it could have resolved    one LLM call
```

It immediately found a defect the corpus could never show: **8.8% false
accepts.** Tier 1 has no concept of an entity it cannot see, so a query naming
an unknown company alongside known ones resolved the known ones and silently
dropped the rest — the same silent-wrong-answer shape as the baseline's, one
layer up. A1 had traded hallucination for omission.

Two conservative rules (unknown-entity evidence forces a decline; stated
weights must sum to a portfolio) took it to **1.1% false accepts, 0% false
declines, corpus coverage unchanged at 1000/1000**. Safety was free at this
operating point.

The one remaining false accept is a genuinely ambiguous lexical case, reported
as a known limitation rather than patched with a special case.

---

## A2 — make the advisor write less

The obvious experiment was a `max_tokens` sweep. **The traces said it would
have been four identical rows:** the advisor generates ~127 tokens against a
400 budget, nothing is ever truncated, and every cap down to 192 is a no-op.
The caps that *do* bind bind by cutting briefings mid-sentence.

So the treatment is the instruction, not the ceiling, and `max_tokens` stays at
400 in every arm — any length change is the model choosing to write less rather
than the harness amputating it.

Three deterministic quality gates, because "shorter is cheaper" is true by
construction and cannot fail:

- **truncation** — `finish_reason` must stay `stop`
- **topic coverage** — the four topics the prompt asks for
- **numeric provenance** — every figure traceable to something computed

| arm | cost/query | words | all four topics | untraceable figures |
|---|---|---|---|---|
| handout | $0.000081 | 79 | 99.9% | 1.32% |
| **short** | **$0.000069** | 65 | **99.3%** | **0.54%** |
| ~~terse~~ | $0.000035 | 28 | **76.6%** ✗ | 0.46% |

**`terse` is rejected by the gate**, and that rejection is the most useful row
in the table. It's 69% cheaper with zero failures and *better* numeric
provenance — and it silently stops answering part of the question, dropping
diversification in 20% of briefings. Without the coverage gate it would have
looked like the best result here.

`short` is the recommendation: cheaper *and* more numerically faithful than the
baseline.

---

## Optimizations rejected on measurement

Two were planned and not attempted, because profiling said not to:

- **Price memoisation** — the deterministic stages total under 0.5% of workflow
  self-time. Eliminating them entirely could not compete with either shipped
  optimization.
- **Parser distillation** — the cascade's measured Tier-2 fallback is 0/1000 on
  the supplied corpus, so there is no parser inference left for a cheaper model
  to remove. Reopens if realistic traffic drives meaningful fallback.

---

## Concurrency

Cost per query falls **19.8×** from C=1 to C=64, replicated in both time orders
to within 2% — so thermal drift explains none of the curve. The 8.8% SM-clock
spread is load-dependent, not thermal: power sits at the 300 W cap throughout
and clock is the variable the controller drops to hold it.

Two findings from the sweep worth more than the curve:

**Reproducibility degrades with concurrency.** C=1 reproduces exactly. Four
identical C=64 runs produced 2 or 3 failures depending on the run — continuous
batching changes which requests share a decode step, which changes matmul
reduction order, which occasionally flips a token. A fixed seed fixes sampling,
not arithmetic. Failure rates above C=16 are quoted as ranges, never point
estimates.

**`utilization.gpu` at 100% does not mean saturated.** It is an occupancy-blind
duty cycle. During the C=8 runs vLLM reported `GPU KV cache usage: 0.0%` and
`Waiting: 0 reqs` — the batch was never full, and the device was starved rather
than busy.

---

## Layout

```
agentops/       measurement, independent of the workflow
  trace.py      spans, LLM calls, query records -> JSONL
  llm.py        the single door to the model; no silent fallback
  instrument.py per-agent timing without editing the agents
  cost.py       allocated-GPU total, fitted attribution weights
  validity.py   run manifest and post-run assertions
  preflight.py  snapshot integrity and live-server checks, before the run
  parser_eval.py    parser correctness, scored over ATTEMPTED queries
  advisor_eval.py   briefing quality gates
  gpu.py        nvidia-smi sampler, scoped to the server's devices
scripts/        bootstrap, serve, bench, report, analysis, robustness, smoke
workflow/       pipeline + the two parser stages written for this project
results/        manifests, validity verdicts and reports for every run
```

`METHODS.md` — pinned environment, run protocol, and every deviation the
cluster forced. `PATCHES.md` — every change to the supplied workflow.

**`python scripts/smoke_test.py`** runs the whole harness against a mock server
and a synthetic price fixture in seconds, with no GPU and no network. 76 checks,
most of them regressions for bugs found during this project.

---

## Known limitations

Stated because a benchmark that only reports its wins is not a benchmark.

- **Tier 1 was tuned against the evaluation corpus.** Two of its rules came from
  watching it fail on those queries. 100% coverage is an upper bound measured on
  the debugging set; the robustness set exists because of this, and is the
  number to trust.
- **Numeric provenance is value-level, not semantic.** A figure is checked
  against everything the workflow computed, so a correct number attached to the
  wrong concept would pass.
- **Repeat counts vary by experiment.** Headline 1,000-query comparisons have
  1–2 repeats; the concurrency sweep has 2 full orders; C=64 has 4. The
  protocol in METHODS asks for three and a median with an interval; the
  headline table does not meet its own protocol and is quoted as point
  estimates for that reason.

### Fixed after an external audit — and what that means for the numbers above

An external review found that several claims in this README rested on checks
weaker than the claims. They are now enforced in code, and
`scripts/recheck_runs.py` re-judges every archived run under the current rules:

| was | now |
|---|---|
| query set hashed **IDs**, not text | `query_content_sha256` + `corpus_sha256` in every manifest |
| `git_sha` null on a non-git working copy | `source_tree_sha256` over every `.py`/`.sh`, which needs no clone |
| a retried model call left a run "clean" | `llm_calls_all_succeeded_first_time`, limit 0 |
| A2's quality gates were **printed**, and weighed by hand | non-inferiority checks inside `validity.check_run` |
| `--stage` was a directory prefix | the stage selects parser and advisor; a contradicting flag is refused |
| `LLM_MAX_RETRIES` was really an attempt count | `LLM_MAX_ATTEMPTS`, old name still read |
| the pre-commit audit checked **filenames** | it now reads the corpus and looks for its words in the bytes being committed |
| quality scores did not record **what scored them** | `_scorer_sha256` in every `advisor_eval.json`, and a gate that refuses to compare two arms scored by different versions |
| Tier 1 **dropped an unknown company in first position** | holdings position is decided by punctuation, not by where the capital letter falls |
| `run.sh report` fell back to the newest run of **any** stage | no fallback: a stage/concurrency with no matching run is an error |
| `eval_cascade.py` exited 0 on **lookback** disagreements | both kinds of disagreement fail the gate |
| `--weights-from` silently ignored a missing calibration | fails closed, and checks model, snapshot, APC, GPUs and C=1 |
| the robustness scorer always exited 0 | pinned budget: 1 false accept, 0 false declines |
| the live-argv check named five flags, claimed "every setting" | full recorded argv compared flag by flag |
| parser scoring sat in `except: print(...)` | `parser_quality_scored` — the accuracy is reported, its absence is gated |
| `data/vocab.json` set derived accuracy but was pinned nowhere | `_vocab_sha256` and `_scorer_sha256` in `parser_eval.json` |

The first row there is the one worth reading twice. `Rivian, AAPL and MSFT`
returned `{AAPL: 0.5, MSFT: 0.5}` — the unrecognised company dropped and the
remainder silently re-weighted into a portfolio nobody asked for. That is the
exact failure A1 was built to eliminate, sitting in the one position the guard
did not look, because the sentence-initial exemption was written for
`Assess the risk.` and then applied to every shape. Tier-1 coverage is
unchanged at 1000/1000, so no published number moves; the routing is simply
correct now where it was confidently wrong before.

### The scorer was the last thing measuring itself

`advisor_eval.py` had a number-tokeniser bug — `"3-4"` read as `3` and
*minus 4*, scoring an invented figure in every briefing with a hyphenated
range. It was fixed partway through the A2 experiment, so some runs were
scored before the fix and some after, and the resulting files were then
compared to each other as though they meant the same thing.

The cost of that was a wrong conclusion, not just a wrong number. A2-short's
two repeats reported 23.6% and 2.8% of briefings carrying an invented figure —
same prompt, same workload, an 8× spread — and the high one failed the
grounding gate. Re-scored on one ruler they read 2.6% and 2.8%, and both arms
pass. **A2-short spent a day looking like a quality regression because of the
tool measuring it.**

A run's manifest pins the model, the snapshot, the query set and the source
tree. It said nothing about the code that produced its quality scores. It does
now, and `validity.check_run` refuses to report a topic or grounding verdict
across mismatched stamps rather than reporting a confident wrong one —
`scripts/rescore_advisor.py` re-scores every run from briefings already frozen
on disk, with no GPU and no model in the loop.

This is the fourth time in this project a quality metric turned out to be
measuring itself before it measured the system: parser accuracy computed over
successes only, `utilization.gpu` read as saturation, the grounding tokeniser,
and now two *versions* of a working scorer compared as one. The generalisation
is not "check your metrics" — it is that an instrument deserves the same
provenance discipline as a model weight.

### Two correct fixes that composed into a false number

The fifth instance is the one worth keeping, because neither half was a
mistake.

**Fix one:** parser accuracy was computed over successful queries only, which
made it most flattering exactly where the parser was worst — a hallucinated
ticker is what crashes the pipeline, so the parser's own errors were the rows
being excluded. Failure rows were added to the scoring.

**Fix two:** `failures.jsonl` carried the supplied query text, so the
redaction policy stripped it.

Each is right. Together they meant that re-scoring a redacted run fed empty
strings to the expected-parse builder, which returned `None`, which put those
rows back into the excluded bucket — the precise state fix one existed to end,
restored by fix two through a door nobody was watching:

```
947 correct / 994 scored = 95.3%    as run, before redaction
947 correct / 966 scored = 98.0%    rescored, 28 hallucinations dropped
```

The numerator never moved. B0's reported ticker accuracy rose 2.7 points
because the parser's worst 28 cases were deleted from its own exam.

It was caught because a published number changed with no measurement behind
it, and that was treated as a reason for suspicion rather than as good news.
Canyon Code's shipped `n_holdings` label settled it in minutes: it never
touches our alias map, it was byte-identical across both scorings, and it
confirmed the parser really had read *Mastercard* as **MCD**, *Salesforce* as
**SFM** and *Alphabet* as **GOOG**.

`parser_eval.json` now reports `n_rows_without_query_text`, `check_run` fails
any run where it is non-zero, and `scripts/rescore_parser.py` rehydrates the
text by `query_id` and refuses to start without the corpus. The general lesson
is narrower and sharper than "check your metrics": **exclusion that correlates
with the thing being measured is not a rounding detail.** Randomly missing
rows cost precision; selectively missing rows manufacture a result.

The last row is the one worth dwelling on. That audit passed four times while
supplied query text was being published — in `failures.jsonl`, in
`report.json`, as a transcribed list of the corpus's sentence openings, and
inside `ParseError` messages. Each round fixed the instance and left the class.
Content scanning found two more the moment it existed, in a file already
"fixed" and in the regression test for an unrelated bug.

**The published numbers predate these gates.** They pass every check that
existed when they were made, and `recheck_runs.py` reports two provenance
fields as gaps on the archived runs rather than backfilling them — a hash of
today's tree stamped onto yesterday's numbers would be exactly the move this
project exists to refuse.

Re-judging all 27 archived runs under the current rules produced three
outcomes, and [`results/INDEX.md`](results/INDEX.md) lists every run under the
one it got:

- **The headline runs hold.** B0, A1 and A2-short at n=1000 fail nothing on
  their own terms.
- **A2-terse is now rejected by the harness**, on topic coverage, for the same
  reason I rejected it by hand. That is the one that mattered: the judgement is
  no longer mine to make or to forget.
- **Three early A1 runs at n=100 fail**, and the tool says why — their Tier-1
  counts exceed attempted queries by exactly the warm-up count, which is the
  warm-up-leak bug `pipe.reset_parser_stats()` fixed. The check works and the
  runs are superseded. They are kept rather than deleted; a results directory
  with the failures removed is a highlight reel.

---

## Rate

All dollar figures use **$1.39/GPU-hour**, RunPod A100 PCIe 80GB on-demand,
retrieved 2026-08-22. Published rates for the same silicon span roughly 8×
across providers, so every figure here is one multiplication from any other
provider's — which is why the rate is always cited with its source and date, and
why stage comparisons are quoted as ratios.

The rate is called **rental-equivalent** throughout. The GPU was not rented;
presenting an opportunity cost as an incurred cost would be a lie that costs
nothing to avoid.
