"""
Run validity assertions.

The handout can complete a 1,000-query sweep without ever contacting a model
and report every query as a success. Verified before building any of this:
with boto3 missing, the workflow returns a well-formed result whose briefing
is a hardcoded template, and the only trace is one print to stdout.

So a run is not a result until it proves what it did. Every measured run
records a manifest of what it claims, then checks the trace against it. A run
that fails any assertion is discarded -- not annotated, not caveated,
discarded. Cheap to build, and the difference between measuring the system
and measuring an accident.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import time
from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RunManifest:
    run_id: str
    stage: str                       # B0, A1, ...
    n_queries: int
    concurrency: int

    model: str = ""
    served_model: str = ""
    base_url: str = ""
    snapshot_id: str = ""
    name_table_sha256: str = ""   # A1: binds data/names.json into the run
    snapshot_source: str = "unknown"      # "yfinance" for real data
    prefix_caching: str = "unknown"       # observed, from the launch record
    expected_prefix_caching: str = "off"  # what this stage REQUIRES
    parser_kind: str = "llm"              # "llm" at B0, "cascade" at A1
    max_failure_rate: float = 0.0         # INFRASTRUCTURE failures; 0 for official runs
    max_workflow_failure_rate: float = 0.10  # parser/model errors: reported, capped
    query_set_id: str = ""                # sha256 of the frozen id list
    query_set_name: str = ""
    # An id list pins WHICH queries ran. It does not pin WHAT THEY SAID.
    # data/queries.json is not in this repository, so a reader comparing two
    # runs by query_set_id alone is comparing two lists of integers -- edit a
    # sentence in the corpus and every id-based checksum stays identical while
    # the workload changes underneath. These hash the text.
    query_content_sha256: str = ""        # sha256 of the query TEXT that ran
    corpus_sha256: str = ""               # sha256 of the whole corpus file
    # What the source tree looked like. git_sha is recorded in env, but it is
    # null on a machine where the code was rsynced rather than cloned -- which
    # is exactly how these runs were produced. This is computed from the files
    # themselves, so it exists either way.
    source_tree_sha256: str = ""
    max_llm_call_failures: int = 0        # failed model calls tolerated
    llm_max_attempts: int = 0             # attempts per call, incl. the first
    generation: dict[str, Any] = field(default_factory=dict)
    # A2 non-inferiority gating
    advisor_reference: str = ""           # run dir whose quality this must match
    advisor_gate_margins: dict[str, float] = field(default_factory=dict)
    telemetry_devices: str = ""           # GPUs nvidia-smi was scoped to
    server_devices: str = ""              # GPUs the server was launched on
    allow_unverified_server: bool = False
    n_workers: int = 0                    # pool size the run measured through
    warmed_workers: int = 0               # workers warmed before the clock started

    advisor_style: str = "handout"   # A2 arm; "handout" is as shipped
    advisor_max_tokens: int = 0
    advisor_temperature: float = 0.0
    parser_max_tokens: int = 0
    seed: int | None = None

    expected_llm_calls_per_query: int = 2   # parser + advisor at B0

    started_at: float = 0.0
    finished_at: float = 0.0
    env: dict[str, Any] = field(default_factory=dict)
    server: dict[str, Any] = field(default_factory=dict)
    server_probe: dict[str, Any] = field(default_factory=dict)
    snapshot_verified: bool = False
    notes: str = ""

    def to_json(self, path: str) -> None:
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, default=str)


def capture_env() -> dict[str, Any]:
    """Everything that would change a number, recorded so it can be pinned."""
    env: dict[str, Any] = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        # Hashed, not recorded verbatim. Manifests ship with the submission,
        # and an externally reachable machine name is not something an
        # evaluator needs -- the hardware configuration is. The hash still
        # proves two runs came from the same host.
        "host_id": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
        "captured_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    for mod in ("vllm", "openai", "torch", "yfinance"):
        try:
            env[mod] = __import__(mod).__version__
        except Exception:  # noqa: BLE001
            env[mod] = None

    env["nvidia_smi"] = _nvidia_smi()
    env["git_sha"] = _cmd(["git", "rev-parse", "--short", "HEAD"])
    env["git_dirty"] = bool(_cmd(["git", "status", "--porcelain"]))
    return env


SOURCE_DIRS = ("agentops", "scripts", "workflow")
SOURCE_SUFFIXES = (".py", ".sh")


def source_tree_sha256(root: str) -> str:
    """One checksum over every source file that could change a number.

    WHY NOT git rev-parse HEAD.
    It is recorded too, in env.git_sha, and it is the better identifier when it
    exists. It did not exist for these runs: the code reached the lab box by
    rsync, so `git rev-parse` returned None and `git status --porcelain` was
    empty for the same reason -- not because the tree was clean, but because
    there was no tree to be dirty. A gate written as `git_sha is not None`
    would then have failed every real run while passing any run made inside a
    clone with uncommitted edits, which is the wrong way round.

    Hashing the files answers the question git was standing in for: were these
    numbers produced by this code? Sorted relative paths go into the hash
    alongside the contents, so a renamed or deleted file changes it too.
    """
    h = hashlib.sha256()
    paths = []
    for d in SOURCE_DIRS:
        base = os.path.join(root, d)
        for dirpath, dirnames, filenames in os.walk(base):
            dirnames[:] = [x for x in dirnames
                           if x not in ("__pycache__", ".ipynb_checkpoints")]
            for fn in filenames:
                if fn.endswith(SOURCE_SUFFIXES):
                    paths.append(os.path.join(dirpath, fn))
    for p in sorted(paths):
        rel = os.path.relpath(p, root).replace(os.sep, "/")
        h.update(rel.encode() + b"\0")
        try:
            with open(p, "rb") as fh:
                h.update(fh.read())
        except OSError:
            h.update(b"<unreadable>")
        h.update(b"\0")
    return h.hexdigest() if paths else ""


def _cmd(argv: list[str]) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None
    except Exception:  # noqa: BLE001
        return None


def _nvidia_smi() -> list[dict[str, Any]]:
    """Per-GPU identity, clocks and power caps.

    Clock and power state are recorded because a long sweep can thermally
    drift; without them a drift-induced change is indistinguishable from a
    treatment effect.
    """
    q = ("index,name,driver_version,memory.total,clocks.max.sm,"
         "power.limit,persistence_mode")
    out = _cmd(["nvidia-smi", f"--query-gpu={q}", "--format=csv,noheader"])
    if not out:
        return []
    rows = []
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 7:
            rows.append(dict(zip(
                ["index", "name", "driver", "memory_total",
                 "sm_clock_max", "power_limit", "persistence"], parts)))
    return rows


# --------------------------------------------------------------------------
# assertions
# --------------------------------------------------------------------------

# A failed query is not automatically a broken run. Two very different things
# were being counted together:
#
#   WORKFLOW      the system worked and the answer was wrong -- the parser
#                 hallucinated a ticker, the model emitted unparseable JSON.
#                 That is a measured property of the baseline, and it is the
#                 number A1 exists to move. Suppressing it would hide the
#                 finding; letting it void the run would mean never having one.
#
#   INFRASTRUCTURE the measurement apparatus failed -- server gone, request
#                 timed out, harness bug. Nothing measured here is meaningful.
#
# Unknown errors count as infrastructure on purpose: a failure nobody has
# classified should stop a run, not quietly pass as an interesting result.
WORKFLOW_FAILURES = ("SnapshotMiss", "ParseError", "PipelineError")
INFRA_FAILURES = ("LLMError", "TimeoutError", "ConnectionError", "APIError")


def classify_failure(error: str) -> str:
    """Which kind of failure is this?

    INFRASTRUCTURE MARKERS WIN, WHEREVER THEY APPEAR IN THE CHAIN.
    Pipeline.run wraps every downstream exception in PipelineError, so a dead
    server arrives as "PipelineError: LLMError: Connection refused". Matching
    on the outermost name alone would file that as a workflow failure -- a
    property of the baseline, reported and tolerated -- when it actually means
    the run measured nothing. The gate that must never be relaxed would have
    been silently bypassed by exception nesting.

    So the infrastructure markers are searched across the whole message first,
    and only then is the outer type consulted.
    """
    text = error or ""
    if any(marker in text for marker in INFRA_FAILURES):
        return "infrastructure"
    head = text.split(":", 1)[0].strip()
    if head in WORKFLOW_FAILURES:
        return "workflow"
    if any(h in text for h in ("SnapshotMiss", "ParseError")):
        return "workflow"
    # Unknown errors count as infrastructure on purpose: a failure nobody has
    # classified should stop a run, not quietly pass as an interesting result.
    return "infrastructure"


def split_failures(failures: list[dict]) -> tuple[list[dict], list[dict]]:
    wf = [f for f in failures if classify_failure(f.get("error", "")) == "workflow"]
    infra = [f for f in failures if classify_failure(f.get("error", "")) != "workflow"]
    return wf, infra


class ValidityError(AssertionError):
    pass


@dataclass
class Check:
    name: str
    ok: bool
    detail: str

    def line(self) -> str:
        return f"  [{'PASS' if self.ok else 'FAIL'}] {self.name}: {self.detail}"


def check_run(
    manifest: RunManifest,
    counter_snapshot: dict[str, Any],
    results: list[dict],
    failures: list[dict],
    advisor: dict[str, Any] | None = None,
    advisor_reference: dict[str, Any] | None = None,
    parser: dict[str, Any] | None = None,
) -> list[Check]:
    """Every check that decides whether this run is reportable."""
    checks: list[Check] = []
    n_ok = len(results)

    # A1 changes how many model calls a correct run should make, so the count
    # checks have to know which parser ran. Loosening them for every stage
    # would retire a guard that has already caught real problems at B0.
    cascade = getattr(manifest, "parser_kind", "llm") == "cascade"
    tiers = counter_snapshot.get("parser_tiers") or {}
    tier1 = int(tiers.get("tier1", 0))
    tier2 = int(tiers.get("tier2", 0))
    by_agent = counter_snapshot.get("llm_calls_by_agent", {})

    # 1. the model was actually called, the expected number of times
    if cascade:
        # one Advisor call per successful query, plus one Parser call for each
        # query Tier 1 declined
        expected = n_ok + tier2
    else:
        expected = n_ok * manifest.expected_llm_calls_per_query
    actual_ok_calls = counter_snapshot["llm_calls"] - counter_snapshot["llm_failures"]
    checks.append(Check(
        "llm_call_count",
        actual_ok_calls >= expected,
        f"{actual_ok_calls} successful calls for {n_ok} queries "
        f"(expected >= {expected}"
        + (f" = {n_ok} advisor + {tier2} tier-2 parser" if cascade else "")
        + "; retries and failed queries add more)",
    ))

    # 2. every stage that should have called the model did
    checks.append(Check(
        "stage_called::AdvisorAgent",
        by_agent.get("AdvisorAgent", 0) >= n_ok,
        f"{by_agent.get('AdvisorAgent', 0)} calls (need >= {n_ok})",
    ))
    if cascade:
        # THE A1 CLAIM, ASSERTED RATHER THAN REPORTED.
        # A1's entire cost argument is that Tier 1 queries make no model call.
        # If the cascade silently fell through on every query, the run would
        # still look healthy -- same answers, same accuracy, just no saving --
        # and the stage would be B0 wearing a different name. So the parser's
        # model-call count must equal the Tier 2 count exactly: not >=, not <=.
        checks.append(Check(
            "cascade_saved_the_calls_it_claims",
            by_agent.get("ParserAgent", 0) == tier2,
            f"ParserAgent made {by_agent.get('ParserAgent', 0)} model calls and "
            f"Tier 1 claims to have handled {tier1} of {tier1 + tier2} queries "
            f"without one; these must match exactly or the saving is not real",
        ))
        attempted = n_ok + len(failures)
        checks.append(Check(
            "cascade_saw_every_query",
            tier1 + tier2 == attempted,
            f"tier1={tier1} + tier2={tier2} must equal {attempted} attempted "
            f"queries exactly; a mismatch means either the cascade was bypassed "
            f"or the counters cover a different window than the measurement "
            f"(warm-up leaking in was a real bug here)",
        ))
        checks.append(Check(
            "name_table_recorded",
            bool(manifest.name_table_sha256),
            f"names.json sha256={manifest.name_table_sha256[:16] or 'MISSING'} "
            f"-- A1 coverage is a property of this name table, so a run that "
            f"does not record which table it used is not reproducible",
        ))
    else:
        checks.append(Check(
            "stage_called::ParserAgent",
            by_agent.get("ParserAgent", 0) >= n_ok,
            f"{by_agent.get('ParserAgent', 0)} calls (need >= {n_ok})",
        ))

    # 3. no result came from the template fallback
    template_allowed = os.environ.get("ADVISOR_ALLOW_TEMPLATE") == "1"
    checks.append(Check(
        "template_fallback_disabled",
        not template_allowed,
        "ADVISOR_ALLOW_TEMPLATE is unset" if not template_allowed
        else "ADVISOR_ALLOW_TEMPLATE=1 -- summaries may be templated, run is not measurable",
    ))

    # 4. exactly one model requested, and the live server advertises it
    requested = set(counter_snapshot.get("requested_models", []))
    checks.append(Check(
        "single_model_requested",
        requested == {manifest.model} if requested else False,
        f"requested={sorted(requested)} declared={manifest.model!r}",
    ))
    probe = manifest.server_probe or {}
    checks.append(Check(
        "live_server_advertises_model",
        bool(probe.get("reachable")) and bool(probe.get("model_present")),
        f"reachable={probe.get('reachable')} "
        f"advertised={probe.get('models_advertised')} "
        f"(asked the running server, not the launch record)",
    ))

    # 5. every price read came from the frozen snapshot
    bad_source = []
    for r in results:
        for t, m in (r.get("metrics") or {}).items():
            if m.get("source") != "snapshot":
                bad_source.append((r.get("query_id"), t, m.get("source")))
    checks.append(Check(
        "prices_from_snapshot",
        not bad_source,
        "all reads from snapshot" if not bad_source
        else f"{len(bad_source)} reads not from snapshot, e.g. {bad_source[:3]}",
    ))

    # 6. the snapshot holds real market data, not a test fixture
    #    The smoke test builds a synthetic snapshot so the harness can be
    #    exercised without a GPU or a network. Without this check that fixture
    #    would satisfy check 5 perfectly and a smoke run would look reportable.
    checks.append(Check(
        "snapshot_is_real_data",
        manifest.snapshot_source == "yfinance",
        f"snapshot_source={manifest.snapshot_source!r} id={manifest.snapshot_id!r}",
    ))

    # 7. prefix caching is what this stage REQUIRES, not merely recorded
    #    A presence-only check ("some value was written") would let a B0 run
    #    pass with APC on -- exactly the contamination this is meant to catch,
    #    and it would silently gut the later S3 comparison.
    checks.append(Check(
        "prefix_caching_matches_stage",
        manifest.prefix_caching == manifest.expected_prefix_caching,
        f"expected {manifest.expected_prefix_caching!r}, "
        f"observed {manifest.prefix_caching!r} (from the server launch record)",
    ))

    # 7a. telemetry watched exactly the GPUs the server was given
    checks.append(Check(
        "telemetry_scoped_to_server_gpus",
        bool(manifest.server_devices)
        and manifest.telemetry_devices == manifest.server_devices,
        f"telemetry={manifest.telemetry_devices!r} "
        f"server={manifest.server_devices!r} -- on a shared box an unscoped "
        f"sampler would fold other users' GPUs into gpu.json",
    ))

    # 7b. the live process was verified, not just the launch record
    probe2 = manifest.server_probe or {}
    checks.append(Check(
        "live_process_config_verified",
        probe2.get("process_matches_record") is True
        or manifest.allow_unverified_server,
        (probe2.get("process_detail") or "not checked")
        + (" [ALLOWED UNVERIFIED]" if manifest.allow_unverified_server else ""),
    ))

    # 7c. the snapshot was byte-verified, not just named
    checks.append(Check(
        "snapshot_checksums_verified",
        manifest.snapshot_verified,
        "every ticker file re-hashed against MANIFEST.json before the run"
        if manifest.snapshot_verified else "checksums were NOT verified",
    ))

    # 8. failures, split by kind.
    attempted = n_ok + len(failures)
    wf, infra = split_failures(failures)

    checks.append(Check(
        "infrastructure_failure_rate",
        (len(infra) / max(1, attempted)) <= manifest.max_failure_rate,
        f"{len(infra)}/{attempted} = {len(infra)/max(1,attempted):.2%} "
        f"(limit {manifest.max_failure_rate:.2%}) -- server, timeout or harness faults",
    ))
    checks.append(Check(
        "workflow_failure_rate",
        (len(wf) / max(1, attempted)) <= manifest.max_workflow_failure_rate,
        f"{len(wf)}/{attempted} = {len(wf)/max(1,attempted):.2%} "
        f"(cap {manifest.max_workflow_failure_rate:.2%}) -- parser/model errors, "
        f"REPORTED as a baseline property, not a fault",
    ))

    # 8a. model calls that failed outright.
    #
    # A retried call that eventually succeeds leaves no trace in any other
    # check here: llm_call_count counts successes, the query completes, the
    # failure lists stay empty, and the run reports as clean. But a retry means
    # the server stumbled DURING the measurement window, and the retry's own
    # latency is inside the wall clock the cost is computed from. That is an
    # infrastructure fault wearing a success's clothes.
    #
    # Default 0 for official runs. Raise it deliberately with
    # --max-llm-call-failures and the manifest records that you did.
    llm_failed = int(counter_snapshot.get("llm_failures", 0))
    checks.append(Check(
        "llm_calls_all_succeeded_first_time",
        llm_failed <= manifest.max_llm_call_failures,
        f"{llm_failed} failed model call(s) (limit {manifest.max_llm_call_failures}) "
        f"-- a retried call still spent measured wall-clock, so it is an "
        f"infrastructure fault even when the query it belonged to succeeded",
    ))

    # 9. the workload was fixed before measurement
    checks.append(Check(
        "query_set_frozen",
        bool(manifest.query_set_id),
        f"{manifest.query_set_name or 'ad-hoc slice'} "
        f"sha256={manifest.query_set_id[:16] or 'NOT FROZEN'}",
    ))

    # 9a. ...and the workload's TEXT is pinned, not just its ids.
    checks.append(Check(
        "query_content_pinned",
        bool(manifest.query_content_sha256) and bool(manifest.corpus_sha256),
        f"queries sha256={manifest.query_content_sha256[:16] or 'MISSING'} "
        f"corpus sha256={manifest.corpus_sha256[:16] or 'MISSING'} -- an id "
        f"list pins which queries ran, not what they said, and the corpus is "
        f"not in this repository for a reader to diff",
    ))

    # 9b. the code that produced these numbers is identified.
    checks.append(Check(
        "source_tree_pinned",
        bool(manifest.source_tree_sha256),
        f"source sha256={manifest.source_tree_sha256[:16] or 'MISSING'} "
        f"git_sha={(manifest.env or {}).get('git_sha')} "
        f"git_dirty={(manifest.env or {}).get('git_dirty')}",
    ))

    # 9c. parser correctness was MEASURED. Not that it was high -- that it
    #     exists, and that it covers the workload that ran.
    #
    #     The accuracy itself is deliberately not a threshold: B0's error rate
    #     is the finding this project is built to report, and gating on it
    #     would mean never having a baseline. But the scoring used to sit
    #     inside `except Exception: print(...)`, so a run could report cost
    #     down, failures zero, validity PASS, and no parser_eval.json at all --
    #     and the stage where that matters most is A1, whose entire claim is
    #     that a deterministic parser did not trade accuracy for speed.
    attempted_all = n_ok + len(failures)
    p_scored = int((parser or {}).get("n_scored", 0))
    checks.append(Check(
        "parser_quality_scored",
        bool(parser) and p_scored >= n_ok,
        f"parser_eval covers {p_scored} of {attempted_all} attempted queries "
        f"(need >= {n_ok} successful) -- the accuracy is reported, not gated; "
        f"its ABSENCE is what is gated here",
    ))
    if parser:
        checks.append(Check(
            "parser_scorer_pinned",
            bool(parser.get("_vocab_sha256")) and bool(parser.get("_scorer_sha256")),
            f"vocab {(parser.get('_vocab_sha256') or 'UNPINNED')[:12]} "
            f"scorer {(parser.get('_scorer_sha256') or 'UNPINNED')[:12]} -- "
            f"derived accuracy is a property of the alias map and the scorer, "
            f"and neither was recorded anywhere before",
        ))

    # 10. A2's quality gates, as gates.
    #
    # These existed as measurements first, printed next to the cost saving and
    # weighed by hand. That is not a gate: A2-terse was 69% cheaper and dropped
    # a required topic from a quarter of its briefings, and nothing in the
    # harness would have stopped it being reported as a win. The judgement was
    # mine, made correctly, and made outside the code -- which means the next
    # person to run an arm inherits the saving and not the discipline.
    #
    # So a stage that changes the Advisor must declare a reference run and beat
    # it, within a stated margin. Cheaper is only better if quality holds.
    if manifest.advisor_style != "handout" or advisor_reference is not None:
        checks.extend(_advisor_gates(manifest, advisor, advisor_reference))

    return checks


# Non-inferiority margins. Chosen before the arms ran and stated here rather
# than in a flag, so they cannot be widened after seeing a result.
#
#   truncation      absolute. A briefing cut mid-sentence is a defect, not a
#                   trade-off, so there is no margin at all.
#   topics          2 percentage points below the reference. The reference
#                   itself is not 100%, and a gate set at parity would fail on
#                   sampling noise alone.
#   grounding       2 points. Same reasoning; the figure is a rate over
#                   thousands of numbers, so 2pp is well outside its noise.
DEFAULT_ADVISOR_MARGINS = {
    "truncation_rate_max": 0.0,
    "all_topics_rate_margin": 0.02,
    "grounded_fraction_margin": 0.02,
}


def _advisor_gates(manifest: RunManifest, advisor: dict[str, Any] | None,
                   ref: dict[str, Any] | None) -> list[Check]:
    m = {**DEFAULT_ADVISOR_MARGINS, **(manifest.advisor_gate_margins or {})}
    out: list[Check] = []

    if not advisor or not advisor.get("n"):
        # Scoring used to be wrapped in try/except so that a scoring bug could
        # not void a good run. That protection is correct for a metric and
        # wrong for a gate: it meant the arm with the worst quality was also
        # the arm most likely to skip its own check silently.
        out.append(Check(
            "advisor_quality_scored",
            False,
            "no advisor_eval produced -- a stage that changes the Advisor "
            "cannot be reported without its quality score",
        ))
        return out

    out.append(Check(
        "advisor_no_truncation",
        advisor["truncation_rate"] <= m["truncation_rate_max"],
        f"{advisor.get('truncated_briefings', '?')} of {advisor['n']} briefings hit the "
        f"token ceiling ({100*advisor['truncation_rate']:.2f}%, limit "
        f"{100*m['truncation_rate_max']:.2f}%) -- brevity must be the model "
        f"choosing to write less, never the sampler cutting it off",
    ))

    if ref is None or not ref.get("n"):
        out.append(Check(
            "advisor_reference_available",
            False,
            f"advisor_reference={manifest.advisor_reference!r} could not be "
            f"read -- a brevity arm is only meaningful against the arm it "
            f"claims to be as good as",
        ))
        return out

    # THE CHECK THAT WOULD HAVE SAVED A DAY.
    #
    # A non-inferiority test compares two measurements. If they came from
    # different versions of the measuring code, the comparison is meaningless
    # no matter how carefully the margin was chosen -- and it fails in the most
    # confusing possible way, by looking like a real quality regression in the
    # treatment. advisor_eval.py's number tokeniser was fixed partway through
    # the A2 experiment; the arm scored before the fix reported 23.6% of
    # briefings carrying an ungrounded figure against 2.8% for the identical
    # arm scored after it, and that difference was read as A2 fabricating more.
    #
    # Both files now stamp the scorer. Same stamp or no comparison.
    sc, sc_ref = advisor.get("_scorer_sha256"), ref.get("_scorer_sha256")
    out.append(Check(
        "advisor_scored_by_the_same_ruler",
        bool(sc) and sc == sc_ref,
        f"scorer {(sc or 'UNSTAMPED')[:12]} vs reference "
        f"{(sc_ref or 'UNSTAMPED')[:12]} -- two quality scores are comparable "
        f"only when the code that produced them is identical; re-score both "
        f"with scripts/rescore_advisor.py",
    ))
    if not sc or sc != sc_ref:
        # Do not go on to report a topic or grounding verdict computed across
        # two different scorers. A wrong answer stated confidently is the thing
        # this whole module exists to prevent.
        return out

    topics, topics_ref = advisor["all_topics_rate"], ref["all_topics_rate"]
    out.append(Check(
        "advisor_topic_coverage_non_inferior",
        topics >= topics_ref - m["all_topics_rate_margin"],
        f"{100*topics:.1f}% of briefings cover all four required topics vs "
        f"{100*topics_ref:.1f}% in {manifest.advisor_reference} "
        f"(margin {100*m['all_topics_rate_margin']:.0f}pp) -- dropping a topic "
        f"is a changed deliverable, not a compressed one",
    ))

    g, g_ref = advisor.get("grounded_fraction_mean"), ref.get("grounded_fraction_mean")
    if g is None or g_ref is None:
        out.append(Check("advisor_grounding_non_inferior", False,
                         "grounding rate unavailable in this run or its reference"))
    else:
        out.append(Check(
            "advisor_grounding_non_inferior",
            g >= g_ref - m["grounded_fraction_margin"],
            f"{100*g:.1f}% of figures match something the workflow computed vs "
            f"{100*g_ref:.1f}% in {manifest.advisor_reference} "
            f"(margin {100*m['grounded_fraction_margin']:.0f}pp) -- a model "
            f"under pressure to be brief can drop specifics, but it can also "
            f"invent them",
        ))

    return out


def enforce(checks: list[Check], strict: bool = True) -> None:
    failed = [c for c in checks if not c.ok]
    print("run validity")
    for c in checks:
        print(c.line())
    if failed and strict:
        raise ValidityError(
            f"{len(failed)} validity check(s) failed; this run is not reportable. "
            f"Fix the cause and re-run rather than reporting it with a caveat."
        )
