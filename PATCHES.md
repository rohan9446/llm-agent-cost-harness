# Changes to the supplied workflow

Two categories, kept separate on purpose.

**Section 1** is the three mechanical fixes required to make `portfolio_clean.zip`
run at all. No redesign, no reformatting, no improvements. The pristine files are
kept at `docs/upstream/` so `diff -u` shows exactly this and nothing else.

**Section 2** is deliberate adaptation the assignment asks for — an open-weight
model on an inference server of our choice, plus the natural-language stage the
corpus needs and the data control the benchmark needs. Each one is justified and
each one is reversible.

Reported upstream on 21 August 2026.

---

## 1. Mechanical fixes

### 1.1 `agents/metrics_agent.py:26` — unmatched `)`

The file does not compile as shipped.

```
$ python -m py_compile agents/metrics_agent.py
  File "agents/metrics_agent.py", line 26
    )
    ^
SyntaxError: unmatched ')'
```

### 1.2 Three `json.loads()` calls now wrap dicts

Fixing the paren is not enough. `PriceAgent.get_history()`, `MetricsAgent.compute()`
and `RiskAgent.assess()` all return dicts directly in the clean version — the
Futures are gone — but the `json.loads()` wrappers that used to decode the
Redis strings are still around them.

```
$ python -c "import portfolio_workflow as w; w.main(holdings={'AAPL':1.0}, lookback_days=30)"
TypeError: the JSON object must be str, bytes or bytearray, not dict
  raised at metrics_agent.py:24
```

Three sites, all the same leftover:

| File | Line | Was | Now |
|---|---|---|---|
| `agents/metrics_agent.py` | 24 | `json.loads(self.price.get_history(...))` | `self.price.get_history(...)` |
| `workflows/portfolio_workflow.py` | 41 | `json.loads(f)` | `f` |
| `workflows/portfolio_workflow.py` | 46 | `json.loads(risk_agent.assess(...))` | `risk_agent.assess(...)` |

### 1.3 Result

```
$ python -m py_compile $(find workflow -name '*.py')      # clean
$ python -c "...w.main(holdings={'AAPL':0.5,'MSFT':0.5}, lookback_days=90)"
keys: ['holdings', 'lookback_days', 'metrics', 'risk', 'summary']
```

`agents/risk_agent.py` is byte-identical to the handout. `agents/price_agent.py`
needed no fix; it is changed only by the adaptation in 2.2.

---

## 2. Deliberate adaptations

### 2.1 Bedrock replaced by vLLM — `agents/advisor_agent.py`

The assignment asks for an open-weight model on an inference server of our
choice, so `boto3` + Bedrock Converse is replaced by a call into
`agentops.llm`, which serves Llama-3.1-8B-Instruct from vLLM. **The prompt,
the 400-token budget and the 0.2 temperature are unchanged from the handout** —
changing them would make B0 a different baseline and every later comparison
would inherit the confound.

The same change removes a silent failure mode. As shipped:

```python
try:
    import boto3
    ...
except Exception as e:
    print(f"AdvisorAgent: Bedrock call failed ({e}); using templated summary.")
    return self._fallback_summary(metrics, risk)
```

`import boto3` sits *inside* the try. With boto3 absent or credentials unset,
every query completes, returns a well-formed result, and never contacts a
model. Verified:

```
AdvisorAgent: Bedrock call failed (No module named 'boto3'); using templated summary.
=> workflow COMPLETED, returned keys: ['holdings','lookback_days','metrics','risk','summary']
=> summary: "The portfolio has an estimated annualized return of 35.7% against 14.6%..."
```

A 1,000-query sweep would report 1,000 successes at zero cost. So the failure
now raises, and `_fallback_summary` is reachable only under
`ADVISOR_ALLOW_TEMPLATE=1` — which the validity check refuses to accept in a
measured run. The method is kept, because routing simple portfolios to it
deliberately is a candidate optimization; reaching it by accident is not.

### 2.2 Frozen price snapshot — `agents/price_agent.py`

`PriceAgent` tries yfinance and falls back to a deterministic synthetic series
when the fetch fails. A rate limit part-way through a sweep would hand some
queries real prices and others fabricated ones, corrupting cost and quality
comparisons at once, with nothing in the output to show for it.

So `PRICE_SNAPSHOT_DIR` adds a branch that reads from a snapshot built once by
`scripts/build_snapshot.py`. Unset the variable and the original behaviour
returns untouched.

Two details that matter:

- **Sliced by calendar date, not row count.** The upstream call is
  `history(period=f"{lookback_days}d")` — calendar days, of which roughly 5/7
  are trading days. Slicing by rows would hand the workflow ~40% more data than
  the original ever saw and silently change every volatility figure.
- **A miss raises** (`SnapshotMiss`). The synthetic generator is unreachable in
  snapshot mode. A benchmark that quietly substitutes invented prices is worse
  than one that stops.
- **The miss message names the snapshot directory by basename, not by absolute
  path.** A one-word change with a publication reason: this string travels into
  `failures.jsonl` and from there into the committed `report.json`, so the
  absolute form put a home directory into eight public artifacts. It was found
  by the pre-commit audit *after* the same path had already been scrubbed out
  of the manifests — fixed at the source here rather than by another pass of
  the redactor, since the redactor kept finding it one step too late.

This is an experimental **control, not an optimization**. B0 deliberately
re-reads and re-slices the snapshot on every single call. Caching that work is
the A3 experiment, and doing it here would steal A3's result.

### 2.3 New stage: `agents/parser_agent.py`

`main()` takes `holdings: dict, lookback_days: int`; `queries.json` is natural
language. The bridge has to exist, and it is part of what gets measured — at B0
it is a second LLM stage on the same vLLM endpoint, which is why it appears in
the per-agent cost breakdown.

Two rules exist for the evaluation rather than for the code:

- The parser emits `lookback_days = None` when the query states no window.
  `queries.json` records `null` in exactly that case, so null is ground truth
  for the parser. What the application defaults to afterwards is *policy*, and
  policy lives in `normalize()`, downstream.
- Unparseable output raises. Defaulting a bad parse to equal weights over an
  empty portfolio would turn a parser failure into a plausible answer — the same
  class of bug as 2.1.

### 2.4 New file: `workflow/pipeline.py`

Composes parser → normalize → `portfolio_workflow.main`. Kept out of
`portfolio_workflow.py` so that file stays at three mechanical fixes and
nothing else, which is what makes section 1 short enough to be worth reading.

### 2.5 Instrumentation is external

Per-agent timing is attached by `agentops.instrument.wrap_agents()`, which
wraps the classes' public methods at runtime. No timing calls were added to any
agent. The agents are the handout plus the changes above, and nothing else.

---

## Reverting

| To undo | Do this |
|---|---|
| Frozen prices | `unset PRICE_SNAPSHOT_DIR` |
| Raise-on-LLM-failure | `export ADVISOR_ALLOW_TEMPLATE=1` (run becomes unmeasurable) |
| Instrumentation | don't call `pipeline.instrument()` |
| Everything | `diff -u docs/upstream/ workflow/portfolio/` |
