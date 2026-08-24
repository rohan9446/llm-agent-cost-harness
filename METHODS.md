# Methods

Everything that would change a number, and how it is held still.

## Configuration under test at B0

| | |
|---|---|
| Workflow | `portfolio_clean.zip`, patched per `PATCHES.md` §1 |
| Model | `meta-llama/Llama-3.1-8B-Instruct`, BF16 |
| Server | vLLM, tensor-parallel size 1, one GPU |
| GPU | NVIDIA A100 80GB PCIe, compute capability 8.0 (SM80), driver 595.71.05 |
| vLLM version | 0.27.1 (torch 2.13.0+cu130, CUDA 13.0, Python 3.11.9) — `capture_env()` records it per run |
| Request logging | off, explicitly — flag spelling resolved by probing `vllm serve --help` |
| Prefix caching | **off**, explicitly — asserted from `results/server.json` |
| Precision | bfloat16 (needs compute capability >= 8.0; `bootstrap.sh` checks) |
| Market data | frozen snapshot, re-read and re-sliced on every call |
| Memoization | none — that is A3 |
| Advisor budget | 400 max tokens, temperature 0.2 (both as shipped) |
| Parser budget | 200 max tokens, temperature 0.0 |
| Seed | 1337, passed to vLLM and to every completion request |
| Query set | `data/query_sets/systems_100.json`, frozen with a sha256 before measurement |
| Failure tolerance | infrastructure 0%; workflow reported and capped at 10% (see *Failure classification*) |
| Attribution weights | fitted on a separate C=1 calibration run (`make calibrate`) |

**On temperature.** The plan originally said temperature 0 everywhere for
reproducibility. The handout ships 0.2, and changing it would make B0 a
different baseline than the one Canyon Code described — every later comparison
would inherit that confound. So the handout's 0.2 is kept and reproducibility
rests on the fixed seed instead, which vLLM honours per request. The parser is a
structured-extraction task with no reason to sample, so it runs at 0.

**But a fixed seed does not buy determinism above C=1 — measured, not assumed.**
See *Reproducibility is concurrency-dependent* below. Every claim in this
document about reproducibility is scoped to a concurrency level.

## Reproducibility is concurrency-dependent

The seed fixes *sampling*. It does not fix *arithmetic*. Under continuous
batching, which requests share a decode step depends on arrival timing, and
batch composition selects different matmul tilings and reduction orders.
Floating-point addition is not associative, so the logits differ in the last
bits, and a token sitting near a decision boundary can flip.

This was measured on the frozen `systems_100` set, same seed, same server
process, temperature 0 at the parser:

| concurrency | runs | outcome |
|---|---|---|
| C=1 | 2 (4 h apart) | 202.4 s and 202.7 s wall, **identical** failing query ids, identical parser scores |
| C=64 | 4 | **2 or 3 failures depending on the run** — query 75 emits valid JSON in some runs and not others |

The four C=64 runs were interleaved in wall-clock time with each other, so this
is run-to-run nondeterminism, not drift. C=1 has a batch size of one at every
step, which is why it reproduces exactly.

Two consequences, both binding on how results are quoted:

1. **A workflow failure rate measured at high concurrency is a sample, not a
   constant.** 2/100 versus 3/100 is a 50% relative swing. Above C=16 the rate
   is reported with replicates and a range, never as a point estimate from a
   single run.
2. **Reproducibility is itself part of the cost/throughput trade.** The 20×
   cost reduction at C=64 is bought partly with determinism. A benchmark run
   once at high concurrency cannot detect this, which is why the sweep is run
   twice in opposite order and C=64 is replicated four times.

## Concurrency sweep protocol, and its drift control

`scripts/sweep_concurrency.sh` varies **only** client-side concurrency. The
server process, weights, KV allocation, flags and argv are byte-identical at
every point, so the server is deliberately *not* restarted between points —
restarting would add a confound (cold allocator, freshly captured CUDA graphs)
rather than remove one. The "never reuse a server across a configuration
change" rule above is about *server* configuration and does not apply.

Ascending order aliases any time-dependent drift onto the treatment. The
control is to run the whole sweep again in descending order:

- cost per query agreed to **under 2% at all seven points** between the two
  orders
- C=64 ran last in one sweep and first in the other and produced the same
  figure

Time order therefore explains nothing in the curve.

**The 8.8% SM clock spread is load-dependent, not thermal.** Clock falls
monotonically with concurrency in both orders (1343 → 1225 MHz), while power
sits at 295–304 W against the A100 PCIe's 300 W cap at every point. The device
is power-limited throughout, and clock is the free variable the controller
lowers to hold the cap; C=64 draws the *most* power at the *lowest* clock,
which is the signature of more work per cycle rather than of throttling. It is
second-order regardless: throughput rose 20× while clock fell 8.8%.

## Pinned and recorded

Captured into every `manifest.json` by `agentops.validity.capture_env()`:

- Python, OS, and a **hashed** host id (never the machine name)
- vLLM, openai, torch, yfinance versions
- git sha, and whether the tree was dirty
- per-GPU: name, driver, VRAM, **max SM clock, power limit, persistence mode**
- snapshot id and snapshot source
- the full vLLM launch argv, via `results/server.json`

The launcher probes `vllm serve --help` and **fails hard** if it cannot set
prefix caching explicitly, or if it can set request logging by neither
`--no-enable-log-requests` (current) nor `--disable-log-requests` (older). Both
spellings exist across versions; having neither would leave request logging at
its default, and on a build that logs by default that overhead lands inside
every measured latency. The resolved flag is recorded in `server.json`.

GPU clock and power state are recorded because a long sweep can thermally
drift. Without them, a drift-induced change is indistinguishable from a
treatment effect.

## Deviations forced by the serving environment

None of these change the workload; all of them change what "the same setup"
means on another machine, so they are recorded rather than tidied away.

| what | why | effect on the measurement |
|---|---|---|
| `VLLM_USE_FLASHINFER_SAMPLER=0` | the FlashInfer sampler JIT-compiles at startup and needs `nvcc`, which is not in the cluster's CUDA install | vLLM's native sampler is used instead. Sampling is a fixed per-token cost; it is inside every measured latency, identically for every stage. |
| FlashInfer patched with `from __future__ import annotations` | FlashInfer 0.5.x annotates with `array.array[int]`, which is not subscriptable on Python 3.11, so `import vllm` raised `TypeError` | none — the patch defers annotation evaluation. Uninstalling FlashInfer was not an option: vLLM imports it unguarded. |
| `gcc`/`g++` from conda-forge, `CC`/`CXX` exported | the node has no system C compiler; vLLM's inductor path needs one | none |
| `HF_HUB_CACHE=/tmp/hf` | `$HOME` has 31 GB, the weights plus cache need more; `/tmp` has 1.2 TB | none. `HF_HOME` was deliberately *not* moved, so the auth token stays on persistent storage. |
| DeepGEMM unavailable | not built for this CUDA/driver combination | rules out one FP8 kernel path. Moot here: SM80 has no native FP8 anyway, which is why the S2 quantization experiment is BF16→INT8 W8A8 rather than MXFP8. |
| Tensor-parallel size 1 | **TP=3 is invalid for this model** — Llama-3.1-8B has 32 attention heads and 8 KV heads, and 3 divides neither. TP=2 is the only other valid setting on this box. | B0 is deliberately single-GPU regardless; recorded here because the scaling study can only step 1 → 2. |
| `run.sh` instead of `make` | no `make` on the node | none — same commands, same order. |

KV cache capacity was checked against the arithmetic rather than trusted:
128 KiB/token × 55.06 GiB of allocatable cache predicted 451,056 tokens, and
vLLM reported exactly that. The prediction matching is the evidence that the
memory configuration is understood, not merely accepted.

## Run protocol

For every measured configuration:

1. Restart or reconfigure the server. Never reuse a server across a
   configuration change.
2. Warm up **through the same thread pool that will measure**, at least once
   per worker. The OpenAI client is `threading.local`, so each worker builds
   its own client and HTTP connection on first use; warming on the main thread
   and then measuring through a fresh pool would warm the model but leave
   client construction, TCP setup and TLS negotiation inside the first measured
   request on every worker. The first `n_workers` warm-up tasks rendezvous at a
   `threading.Barrier`, which forces every worker thread to exist and be busy
   simultaneously — without it one fast thread could serve them all while the
   rest stayed cold. Warm-up is **traced** under `warmup-*` ids and **excluded
   by name** from every metric.
3. Measure.
4. Repeat at least three times (`--repeat 1|2|3`).
5. Report the median with a confidence interval, never a single run.

> **This protocol is not what the headline table met, and saying otherwise
> would be the easiest lie in the document.** The 1,000-query B0/A1/A2
> comparisons have one to two repeats each; the concurrency sweep has two full
> orders, and C=64 has four. Steps 4 and 5 describe how the numbers *should* be
> reported, and the headline figures are therefore quoted as point estimates
> rather than medians with intervals. Where a result was replicated, it says so
> and gives both runs: the concurrency reduction reproduced within 2% in both
> directions, which is the strongest replication claim this project can make.
> Reporting a "median of three" over one run is not a rounding error in
> presentation, it is a fabricated statistic.

**Run order is interleaved, not sequential.** Configurations are cycled
A,B,C,A,B,C rather than AAA,BBB,CCC, so thermal drift spreads across treatments
instead of loading onto whichever ran last.

## Corpus handling

- Order is the file's own order by default. `--shuffle` uses `--seed`; either
  way the choice is recorded in the manifest.
- The full 1,000 for offline cost runs.
- Online rate sweeps may use a stratified subsample if GPU windows are tight;
  the subsample is validated against the full set at B0 before anything is
  built on it.

## Cost

**Measured:** `GPUs x wall-clock x rental-equivalent $/GPU-second`. A
measurement, not a fit, and indifferent to how work was scheduled.

**Estimated:** each stage's and each query's share of that total, using
attribution weights fitted in two regressions —
`TTFT ~ alpha + a*prompt` and `latency-TTFT ~ beta + b*(completion-1)` — so the
fixed per-request overhead stays out of the per-token slopes.

Three constraints on the fit, all of which the code enforces or annotates:

- It is only valid inside the concurrency regime it was fitted in. Above C=1,
  measured latency also contains queueing delay, which inflates both
  coefficients and inflates them unequally. Fit on a C=1 calibration run and
  carry those coefficients across regimes; `--weights-from` exists for this,
  and a fit performed at C>1 is labelled as such in `report.json`.
- The fitted intercepts `alpha` and `beta` are included when dividing the
  total. Excluding them would charge per-request overhead to nobody, and would
  overstate the advisor's share — at B0 both stages make exactly one request
  per query, so the fixed cost divides evenly between them.
- Cached prompt tokens are charged at 5% of a fresh prefill token. That is a
  **stated assumption, not a measurement** — S3a is the experiment that would
  replace it.

**Never done:** multiplying per-request latency by a GPU-hour rate. Continuous
batching means many requests share the GPU simultaneously; summing per-request
wall time overcounts by roughly the batch factor.

**Separately modelled:** what the original `global_controller.yaml` would bill
on AWS — five `t3.micro` instances of uptime — kept in `cost.DeploymentModel`,
labelled "modelled, not measured", and never merged with measured cost.

## Validity

A run is not a result until it proves what it did. The checks are listed in
`README.md` and implemented in `agentops/validity.py`. A run failing any of
them is discarded rather than annotated.

The check most worth understanding is `snapshot_is_real_data`. The smoke test
builds a synthetic price fixture so the harness can be exercised without a GPU
or a network — and that fixture satisfies every other check perfectly. Without
snapshot provenance, a smoke run would be indistinguishable from a measurement.

## Pre-run checks

Two things are verified before a benchmark starts, because a validity check on
the trace can only tell you a run was wrong once the GPU time is spent.

- **Snapshot integrity.** Every ticker file is re-hashed and compared to
  `MANIFEST.json`. Writing checksums and never verifying them would let a file
  edited after the snapshot was built keep the same `snapshot_id` forever.
- **Live server identity and configuration.** `results/server.json` is written
  *before* vLLM starts and describes intent, not the process currently holding
  the port. So `/v1/models` is fetched from the running server, and
  `/proc/<pid>/cmdline` is read back and checked for the model, the
  prefix-cache flag, dtype, tensor-parallel size and port. The launcher
  captures its PID with `$$` before `exec`, which preserves it — logging goes
  through a process substitution rather than a pipe precisely so the exec is
  not turned into a pipeline and the PID stays correct. The LLM counter
  reports `requested_models` -- what we asked for -- and is named that way so
  it is never mistaken for proof of what ran.

## GPU telemetry scope

The sampler is scoped to the devices recorded in the *server's* launch record,
not to the benchmark shell's `CUDA_VISIBLE_DEVICES`. `make serve` and
`make bench` run in different shells, and only the first is required to set it;
on a shared three-GPU box an unscoped sampler would fold other users' work into
`gpu.json` while the cost model charged for one device. A validity check asserts
the two device sets match.

## Interpreting the agent table

Agent spans nest: `MetricsAgent.compute()` calls `PriceAgent.get_history()`, so
`wall_s` is **inclusive** and those numbers do not sum to the total. `self_s` is
exclusive and does. Percentages are computed from `self_s`.

## Parser correctness

A query can complete perfectly while the parser misread it, and the benchmark
would then be measuring the cost of answering the wrong question accurately.
Every run scores the parse and writes `parser_eval.json`: holding count,
lookback value and stated-vs-unstated against the shipped labels; ticker set
against the derived alias map, and **portfolio weights** against the same map --
all clearly marked as derived.

Weights are scored because 620 of the 1,000 queries are percentage-weighted, and
a parser that returns 10/90 for a 90/10 portfolio passes holding count, ticker
set and lookback while the workflow analyses something nobody asked for.
Reported as exact-match within 1e-3, plus p95 max-absolute error and mean L1
error.

The derived labels resolve 1000/1000 queries, and the derived holding counts
agree with the shipped `n_holdings` label on all 1000 -- an independent check
that the derivation is correct rather than merely permissive.

Reported, not gated: a real error rate is a useful number, and A1 exists to
move it.

**Scored over attempted queries, not successful ones.** A hallucinated ticker
raises `SnapshotMiss` and takes the workflow down with it, so a parser metric
computed only over completed queries drops exactly the cases where the parser
was most wrong — it would be most optimistic precisely where the parser is
worst. `PipelineError` carries the parse that caused the failure through to the
failure record, and `parser_eval.score()` takes a `failures` argument so those
rows are graded alongside the successes. `parser_eval.json` reports
`n_failed_scored` so the difference is visible rather than assumed.

This is not hypothetical. In the first C=1 calibration run on the frozen
`systems_100` set, 2 of 100 queries failed this way — "Mastercard" parsed to
**MCD** (McDonald's; correct is MA) and "Salesforce" to **SFM** (Sprouts;
correct is CRM). Scored over successes only, derived ticker-set accuracy reads
99% of 98. Scored over attempted, it is 97% of 100. The 2-point difference is
entirely composed of the parser's worst errors.

## Failure classification

Failures are split by whether they invalidate the measurement or *are* the
measurement:

| class | examples | treatment |
|---|---|---|
| **infrastructure** | `ConnectionError`, `TimeoutError`, `APIError`, `LLMError`, anything unclassified | gate at 0%. The run did not measure what it claims; it is discarded. |
| **workflow** | `SnapshotMiss`, `ParseError`, `PipelineError` | reported as a rate, capped at 10%. This is a property of the baseline being measured. |

An error nobody has classified counts as infrastructure deliberately — failing
closed means a novel failure stops a run instead of passing quietly as a finding.

Workflow failures stay in the cost denominator. They burned GPU time, so
**cost per attempted query** is the headline figure; dividing only by successes
would let a system look cheaper for having failed. Both figures are reported.

## Requesting resources

Sized for one GPU serving Llama-3.1-8B in bf16, with the benchmark driver on
the same node.

| | ask for | why |
|---|---|---|
| GPU | 1 whole device for B0; 2 for S4 | a MIG slice or a shared device makes allocated-GPU-time costing meaningless |
| CPU | 8 cores (4 minimum) | vLLM's API server and scheduler, plus the deterministic agents — `RiskAgent` computes an O(n²T) covariance in pure Python |
| RAM | 64 GB (32 GB minimum) | weights are staged in host memory before transfer; transient peak is ~22–30 GiB |
| Disk | 25 GB free | ~15 GiB of weights plus the HF cache; set `HF_HOME` if `$HOME` is small |
| Walltime | 2 h for the first session | the model download dominates; later sessions need ~30 min |

The first session is longer than the work because of the download. Once the
weights are cached, B0 end to end — calibrate, bench, report, three repeats —
is well under half an hour.

## S4 on two GPUs

Two devices makes the scaling question sharper than three would have. With
three, the comparison was "how far does data parallelism scale" — with two, it
is the question a deployment actually faces:

**one model split across both GPUs, or two independent replicas?**

| config | GPUs | what it tests |
|---|---|---|
| DP=1 | 1 | the B0 baseline |
| DP=2 | 2 | two replicas, requests spread across both |
| TP=2 | 2 | one model sharded across both |

TP=2 is legal for this model, unlike TP=3: 32 attention heads, 8 KV heads and
an intermediate size of 14336 all divide evenly by 2. For an agentic workload —
many concurrent, mostly-small requests — data parallelism should win, because
tensor parallelism pays cross-GPU communication on every layer to buy
single-request latency this workload does not need. Measuring that rather than
asserting it is the point.

Two operational constraints on this node, both of which would otherwise be
discovered the hard way:

- **Launch DP=2 replicas one at a time.** Each stages ~15 GiB of weights
  through host memory, peaking around 30 GiB. Started together that is ~60 GiB
  against 64 GiB of RAM, which OOM-kills during load. Start the second only
  after the first logs `Application startup complete`; once resident, steady-
  state host RAM is far lower.
- **8 CPU cores across two servers plus the driver is tight.** Record CPU
  utilization for the DP=2 runs. If the cores saturate, DP=2 is being measured
  against a CPU ceiling rather than a GPU one, and the result has to say so.

**Not yet built:** DP=2 needs the client to spread requests across two
endpoints. `agentops/llm.py` currently reads a single `LLM_BASE_URL`. Making it
accept a comma-separated list and round-robin per worker thread is a small
change, but it is deliberately not in the B0 harness — B0 is frozen for
measurement, and untested code added now would risk the runs that matter.

## Known limitations

- **GIL contention.** The deterministic agents are pure-Python and CPU-bound;
  `RiskAgent` computes an O(n²T) covariance without numpy. Under a concurrent
  thread pool they contend for the GIL, so per-agent wall time at high
  concurrency includes contention that would not exist in a process-parallel
  deployment. CPU time is per-thread (`time.thread_time`) and is not affected.
- **Attribution weights are fitted, not measured.** Two regressions —
  `TTFT ~ alpha + a*prompt` and `latency-TTFT ~ beta + b*(completion-1)` — keep
  the fixed per-request overhead out of the per-token slopes. They still divide
  a measured total; they are not a claim about GPU-seconds per token, and the
  report labels them accordingly. Both R² values are printed; treat a low one
  as a reason to distrust the split, never the total.
- **Cost per query has two denominators.** Failed queries consume GPU time, so
  dividing only by successes understates cost. Both `cost_per_query_usd` (over
  successes) and `cost_per_attempted_query_usd` are reported. At the official
  B0 setting they are equal, because 100/100 is required.
- **Rental-equivalent rate.** Must be replaced with a citable, dated rate for
  the *actual* device before any number is reported. An A100 80GB has an easy
  public market rate, which makes this simpler than an RTX PRO 6000 would have
  been.
- **S2 cannot be MXFP8 on an A100.** MXFP8 W8A8 needs SM100 (Blackwell); the
  A100 is SM80 and has no native FP8 tensor cores at all — FP8 arrived with Ada
  (SM89) and Hopper (SM90). The A100 *does* have INT8 tensor cores, so the
  quantization experiment becomes **bf16 → INT8 W8A8** (via
  compressed-tensors / llm-compressor). Same question — does lower precision cut
  GPU cost without losing quality — on the hardware actually available, and the
  report says which scheme was used and why.
- **KV cache is not the constraint.** On an 80 GB device, bf16 weights take
  ~15 GiB and leave ~55 GiB for KV cache -- about 451k tokens, or roughly 500
  concurrent requests at this workload's ~900 tokens each. C=8 uses under 2% of
  it. So nothing in B0 is memory-bound, and the S1 concurrency sweep can run to
  saturation without hitting a KV wall. It also means low GPU utilization at
  100 queries is a scheduling result, not a capacity one.
- **Parser vocabulary.** At B0 the parser is asked to emit tickers directly and
  is given no alias map. A ticker outside the corpus vocabulary raises
  `SnapshotMiss` and is recorded as a failure — which is a real parse error, and
  measuring it is the point.
