#!/usr/bin/env python3
"""
The benchmark driver.

Two modes, because cost efficiency and serving quality are different
questions and one harness cannot answer both honestly:

  offline   offer work as fast as the pool accepts it. Answers throughput and
            cost per query. This is the mode the Systems metrics come from.

  online    hold a fixed arrival rate with Poisson inter-arrivals. Answers
            TTFT, TPOT, end-to-end percentiles and where the SLO breaks.
            Poisson rather than fixed-interval because burstiness changes
            queueing behaviour materially, and the default is not
            self-documenting.

Nothing is reported until the run passes its validity checks.

    python scripts/run_bench.py --stage B0 --n 100 --concurrency 8
    python scripts/run_bench.py --stage B0 --n 200 --mode online --rate 4
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import random
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import gpu, llm, preflight, trace, validity   # noqa: E402
from agentops.validity import RunManifest                   # noqa: E402


def _content_sha(queries: list[dict]) -> str:
    """sha256 over the query TEXT, in the order it ran.

    query_set_id hashes an id list. That pins which rows of the corpus were
    used and nothing about what they contain -- and the corpus is not in this
    repository, so a reader cannot diff it. Edit one sentence in queries.json
    and every id-based checksum in every manifest stays byte-identical while
    the workload underneath has changed.

    Hashing the text closes that. It also makes the two hashes independently
    useful: matching content hashes across two runs prove the same sentences
    ran, whatever the ids were called.
    """
    payload = json.dumps([q.get("query", "") for q in queries],
                         separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _corpus_sha(path: str) -> str:
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


def load_queries(path: str, n: int, seed: int, shuffle: bool,
                 query_set: str | None) -> tuple[list[dict], str, str]:
    """Returns (queries, set_name, set_sha256).

    A frozen set is an explicit id list with a checksum, fixed before any
    measurement. `--n 100` on the raw corpus is reproducible but implicit;
    "the workload was fixed before measurement and reused across comparisons"
    is only defensible if there is a file saying so.
    """
    with open(path, encoding="utf-8") as fh:
        qs = json.load(fh)

    if query_set:
        with open(query_set, encoding="utf-8") as fh:
            spec = json.load(fh)
        by_id = {q["id"]: q for q in qs}
        missing = [i for i in spec["ids"] if i not in by_id]
        if missing:
            raise SystemExit(f"query set {spec['name']} references unknown ids {missing[:5]}")
        payload = json.dumps({"ids": spec["ids"]}, separators=(",", ":"),
                             sort_keys=True).encode()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != spec["sha256"]:
            raise SystemExit(
                f"query set {spec['name']} has been edited since it was frozen "
                f"(sha256 {actual[:16]} != {spec['sha256'][:16]})")
        return [by_id[i] for i in spec["ids"]], spec["name"], actual

    if shuffle:
        rng = random.Random(seed)
        qs = qs[:]
        rng.shuffle(qs)
    return (qs[:n] if n > 0 else qs), "", ""


# What each stage IS, rather than what it is called.
#
# --stage was a directory-name prefix. Nothing checked it: `--stage A1 --parser
# llm` produced a directory called A1-offline-... containing a B0 run, and the
# only thing standing between that and a published table was remembering to
# type the second flag. Two of the three findings this project reports are
# comparisons between stages, so the label being decorative is not a cosmetic
# problem.
#
# Now the stage SELECTS the configuration, and an explicit flag that disagrees
# with it is refused rather than silently winning.
#
#   parser         which parser this stage is defined by
#   advisor_style  which Advisor prompt
#   reference      the run whose advisor quality a brevity arm must match
#
# The names here are the names the existing runs already used -- A2-short, not
# A2. Inventing tidier ones would have made every future run of a published
# stage fail this check, which is the failure mode of adding enforcement to a
# system that already has history.
STAGES: dict[str, dict] = {
    "B0":       {"parser": "llm",     "advisor_style": "handout"},
    "A1":       {"parser": "cascade", "advisor_style": "handout"},
    "A2-short": {"parser": "cascade", "advisor_style": "short",
                 "needs_reference": True},
    "A2-terse": {"parser": "cascade", "advisor_style": "terse",
                 "needs_reference": True},
    # The harness fixture. Registered rather than special-cased, so the smoke
    # test exercises the same code path as a real run instead of a bypass --
    # a stage-enforcement bug that only appears outside the smoke test is a
    # bug the smoke test cannot find.
    "SMOKE": {"parser": "llm", "advisor_style": "handout"},
}


def stage_spec(stage: str) -> dict:
    """Configuration for a stage name, tolerating the 'cal' suffix.

    Calibration runs are the same stage at C=1 -- B0cal must be B0 or the
    attribution weights come from a different system than the run they are
    applied to.
    """
    base = stage[:-3] if stage.endswith("cal") else stage
    return STAGES.get(base, {})


def apply_stage(a) -> None:
    """Make --stage mean something, or refuse to run."""
    spec = stage_spec(a.stage)
    if not spec:
        if not a.stage_freeform:
            raise SystemExit(
                f"--stage {a.stage!r} is not a defined stage "
                f"({', '.join(sorted(STAGES))}).\n"
                f"A stage name reaches the run directory, the report and the "
                f"README table, so an undefined one is an unlabelled result.\n"
                f"Pass --stage-freeform to run it anyway; the manifest will "
                f"record that the stage enforced nothing.")
        return

    for flag, key in (("parser", "parser"), ("advisor_style", "advisor_style")):
        want = spec[key]
        given = getattr(a, flag)
        explicit = f"--{flag.replace('_', '-')}" in sys.argv
        if explicit and given != want:
            raise SystemExit(
                f"stage {a.stage} is defined as {key}={want!r}, but "
                f"--{flag.replace('_', '-')} {given!r} was passed.\n"
                f"One of the two is wrong, and guessing which would put a "
                f"mislabelled directory into the results table. Fix the "
                f"command, or add a new stage to STAGES in this file.")
        setattr(a, flag, want)

    if spec.get("needs_reference") and not a.advisor_reference:
        raise SystemExit(
            f"stage {a.stage} changes the Advisor prompt, so it must declare "
            f"the run whose quality it claims to match:\n"
            f"  --advisor-reference results/A1-offline-n1000-c8-rep1\n"
            f"Cheaper is only better if quality holds, and 'holds' needs "
            f"something to hold against.")


def read_server_record(path: str) -> dict:
    """What the serve script recorded when it launched vLLM.

    Read rather than assumed: prefix caching is enabled by default upstream,
    so a run that assumes 'off' because it never asked would be measuring
    something other than what it claims.
    """
    if not os.path.exists(path):
        return {}
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def snapshot_info(snapshot_dir: str) -> tuple[str, str]:
    """Returns (snapshot_id, source). Source distinguishes real market data
    from the synthetic fixture the smoke test builds."""
    m = os.path.join(snapshot_dir, "MANIFEST.json")
    if not os.path.exists(m):
        return "", "missing"
    with open(m, encoding="utf-8") as fh:
        d = json.load(fh)
    return d.get("snapshot_id", ""), d.get("source", "unknown")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", default="B0")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--mode", choices=["offline", "online"], default="offline")
    ap.add_argument("--concurrency", type=int, default=8, help="offline mode")
    ap.add_argument("--rate", type=float, default=4.0, help="online mode, queries/sec")
    ap.add_argument("--online-workers", type=int, default=0,
                    help="pool size for online mode; 0 auto-sizes. The pool must "
                         "never be the bottleneck -- if submissions queue "
                         "client-side the arrival process itself is distorted, "
                         "which is a measurement error, not just slowness. So it "
                         "is sized generously and capped, since every worker is "
                         "warmed before the clock starts.")
    ap.add_argument("--burstiness", type=float, default=1.0,
                    help="1.0 = Poisson arrivals; >1 more uniform, <1 burstier")
    ap.add_argument("--repeat", type=int, default=1, help="repeat index, for labelling")
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shuffle", action="store_true",
                    help="shuffle the corpus with --seed; order is recorded either way")
    ap.add_argument("--queries", default=os.path.join(ROOT, "data", "queries.json"))
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "data", "snapshot"))
    ap.add_argument("--server-record", default=os.path.join(ROOT, "results", "server.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results"))
    ap.add_argument("--query-set", default=None,
                    help="path to a frozen query-set file, e.g. "
                         "data/query_sets/systems_100.json")
    ap.add_argument("--parser", choices=["llm", "cascade"], default="llm",
                    help="llm = B0 baseline; cascade = A1 rules-first. The only "
                         "thing that changes between the two stages, so the "
                         "comparison holds everything else constant by "
                         "construction rather than by discipline.")
    ap.add_argument("--advisor-style", choices=["handout", "short", "terse"],
                    default="handout",
                    help="A2 brevity arm. 'handout' is the shipped prompt and "
                         "is what B0 and A1 measured. max_tokens stays at 400 "
                         "in every arm, so a shorter briefing means the model "
                         "chose to write less rather than being cut off.")
    ap.add_argument("--expect-apc", choices=["on", "off"], default="off",
                    help="prefix-caching setting this stage REQUIRES; the run "
                         "fails if the server was launched differently")
    ap.add_argument("--max-failure-rate", type=float, default=0.0,
                    help="INFRASTRUCTURE failures (server, timeout, harness). "
                         "0.0 for official runs -- these mean the measurement "
                         "itself is unsound")
    ap.add_argument("--max-workflow-failure-rate", type=float, default=0.10,
                    help="WORKFLOW failures (parser hallucination, unparseable "
                         "output). These are a property of the baseline being "
                         "measured, not a fault in the measurement, so they are "
                         "reported and costed against attempted queries rather "
                         "than voiding the run. The cap exists to catch a "
                         "genuinely broken parser, not to hide a real error rate.")
    ap.add_argument("--advisor-reference", default=None,
                    help="run directory whose advisor_eval.json this run's "
                         "briefing quality must be non-inferior to. Required "
                         "for any stage that changes the Advisor prompt: a "
                         "brevity saving is only a saving if the briefing is "
                         "still as good, and that comparison belongs in the "
                         "validity checks rather than in someone's judgement "
                         "after the fact.")
    ap.add_argument("--stage-freeform", action="store_true",
                    help="permit a --stage name that is not in STAGES; the "
                         "manifest records that the stage enforced nothing")
    ap.add_argument("--max-llm-call-failures", type=int, default=0,
                    help="failed model CALLS tolerated (not failed queries). "
                         "A call that failed and was retried still spent "
                         "measured wall-clock, so 0 for official runs.")
    ap.add_argument("--allow-unverified-server", action="store_true",
                    help="proceed when the live vLLM process cannot be verified "
                         "against results/server.json; recorded in the manifest")
    ap.add_argument("--overwrite", action="store_true",
                    help="delete an existing run directory before starting")
    ap.add_argument("--no-strict", action="store_true",
                    help="report validity failures without aborting -- for debugging only")
    a = ap.parse_args()

    # Before anything else: make the stage label binding.
    apply_stage(a)

    if not os.path.exists(os.path.join(a.snapshot, "MANIFEST.json")):
        print(f"no frozen snapshot at {a.snapshot}\n"
              f"run: python scripts/build_snapshot.py", file=sys.stderr)
        return 2
    os.environ["PRICE_SNAPSHOT_DIR"] = a.snapshot
    # ASSIGNED, not setdefault. setdefault let an LLM_SEED already exported in
    # the shell win over --seed while the manifest recorded --seed, so the
    # artifact would name a seed the run did not use. The flag is the record;
    # the environment is how it gets there.
    os.environ["LLM_SEED"] = str(a.seed)
    # Set before the agents are constructed and before any prompt is
    # built, and recorded in the manifest below so the arm is part of
    # what the run claims rather than something the shell remembered.
    os.environ["ADVISOR_STYLE"] = a.advisor_style

    # Loaded before the tag is built: a frozen query set overrides --n, so a
    # tag derived from --n would name a directory that misdescribes its own
    # contents.
    queries, qset_name, qset_sha = load_queries(
        a.queries, a.n, a.seed, a.shuffle, a.query_set)

    tag = f"{a.stage}-{a.mode}-n{len(queries)}-" + (
        f"c{a.concurrency}" if a.mode == "offline" else f"r{a.rate:g}"
    ) + f"-rep{a.repeat}"
    out_dir = os.path.join(a.out, tag)

    # trace.jsonl is opened in append mode, so re-running the same tag would
    # concatenate the old run's spans onto the new one's while results.jsonl
    # and manifest.json were overwritten -- silently contaminating latency,
    # tokens, agent timings and the whole cost attribution. Refuse instead.
    if os.path.isdir(out_dir) and os.listdir(out_dir):
        if not a.overwrite:
            print(f"run directory already exists and is not empty:\n  {out_dir}\n"
                  f"Re-running would append to trace.jsonl while overwriting "
                  f"results.jsonl, mixing two runs into one analysis.\n"
                  f"Pass --overwrite to discard it, or --repeat N for a new one.",
                  file=sys.stderr)
            return 2
        shutil.rmtree(out_dir)
    os.makedirs(out_dir, exist_ok=True)

    run_id = trace.start_run(os.path.join(out_dir, "trace.jsonl"), run_id=tag)

    # import after PRICE_SNAPSHOT_DIR is set, so agents see it at construction
    sys.path.insert(0, os.path.join(ROOT, "workflow"))
    import pipeline                                    # noqa: E402
    pipeline.instrument()

    server = read_server_record(a.server_record)
    pipe = pipeline.Pipeline(parser=a.parser)

    # ---- preflight: refuse to start rather than discover it afterwards -----
    base_url = os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1")
    try:
        snap = preflight.verify_snapshot(a.snapshot)
    except preflight.PreflightError as exc:
        print(exc, file=sys.stderr)
        return 2
    facts = preflight.probe_server(base_url, llm.model_name(), server)
    print(f"preflight: snapshot {snap['snapshot_id']} ({snap['source']}) "
          f"{snap['n_tickers']} tickers verified")
    print(f"preflight: server {facts.detail}")
    print(f"preflight: live process {facts.process_detail}")
    if not a.no_strict:
        try:
            preflight.enforce_preflight(
                facts, snap, require_real_data=True,
                require_process_match=not a.allow_unverified_server)
        except preflight.PreflightError as exc:
            print(f"\n{exc}\n\nNo GPU time spent.", file=sys.stderr)
            return 2

    telemetry_devices = server.get("cuda_visible_devices") or \
        os.environ.get("CUDA_VISIBLE_DEVICES") or None

    manifest = RunManifest(
        run_id=run_id,
        stage=a.stage,
        n_queries=len(queries),
        concurrency=a.concurrency if a.mode == "offline" else 0,
        model=llm.model_name(),
        base_url=os.environ.get("LLM_BASE_URL", "http://127.0.0.1:8000/v1"),
        snapshot_id=snap["snapshot_id"],
        name_table_sha256=_name_table_sha(a),
        snapshot_source=snap["source"],
        snapshot_verified=snap["verified"],
        prefix_caching=server.get("prefix_caching", "unknown"),
        expected_prefix_caching=a.expect_apc,
        max_failure_rate=a.max_failure_rate,
        max_workflow_failure_rate=a.max_workflow_failure_rate,
        query_set_id=qset_sha,
        query_set_name=qset_name,
        query_content_sha256=_content_sha(queries),
        corpus_sha256=_corpus_sha(a.queries),
        source_tree_sha256=validity.source_tree_sha256(ROOT),
        max_llm_call_failures=a.max_llm_call_failures,
        llm_max_attempts=llm.max_attempts_setting(),
        generation=llm.effective_generation_config(),
        advisor_reference=a.advisor_reference or "",
        server_probe=facts.as_dict(),
        telemetry_devices=telemetry_devices or "",
        n_workers=_pool_size(a),
        warmed_workers=max(a.warmup, _pool_size(a)),
        server_devices=server.get("cuda_visible_devices", ""),
        allow_unverified_server=a.allow_unverified_server,
        advisor_style=a.advisor_style,
        advisor_max_tokens=int(os.environ.get("ADVISOR_MAX_TOKENS", "400")),
        advisor_temperature=float(os.environ.get("ADVISOR_TEMPERATURE", "0.2")),
        parser_max_tokens=int(os.environ.get("PARSER_MAX_TOKENS", "200")),
        seed=a.seed,
        env=validity.capture_env(),
        server=server,
        parser_kind=a.parser,
        notes=f"mode={a.mode} rate={a.rate} burstiness={a.burstiness} "
              f"shuffle={a.shuffle} repeat={a.repeat} parser={a.parser}",
    )

    results: list[dict] = []
    failures: list[dict] = []

    def one(q: dict) -> None:
        qid = str(q["id"])
        t0 = time.perf_counter()
        try:
            with trace.query(qid, run_id_value=run_id, corpus_id=q["id"]):
                r = pipe.run(q["query"])
            results.append({
                "query_id": qid,
                "query": q["query"],
                "wall_s": time.perf_counter() - t0,
                "holdings": r.get("holdings"),
                "lookback_days": r.get("lookback_days"),
                "metrics": r.get("metrics"),
                "risk": r.get("risk"),
                "summary": r.get("summary"),
                "parsed": r.get("_parsed"),
                "lookback_source": r.get("_lookback_source"),
                "label": {k: q.get(k) for k in
                          ("n_holdings", "phrasing", "expected_lookback_days")},
            })
        except Exception as exc:  # noqa: BLE001
            # A downstream crash can be the parser's fault -- a hallucinated
            # ticker that is not in the frozen snapshot, say. PipelineError
            # carries the parse that caused it, so the parser gets scored on
            # the queries it broke rather than only on the ones it survived.
            failures.append({
                "query_id": qid, "query": q["query"],
                "error": f"{type(exc).__name__}: {exc}",
                "failure_class": validity.classify_failure(
                    f"{type(exc).__name__}: {exc}"),
                "parsed": getattr(exc, "parsed", None),
                "label": {k: q.get(k) for k in
                          ("n_holdings", "phrasing", "expected_lookback_days")},
                "wall_s": time.perf_counter() - t0,
            })

    print(f"{tag}: {len(queries)} queries", flush=True)

    # ---- one pool, created BEFORE warm-up ---------------------------------
    # The OpenAI client is threading.local, so every worker thread builds its
    # own client and its own HTTP connection pool on first use. Warming up on
    # the main thread and then measuring through a fresh pool would warm the
    # model but leave client construction, TCP setup and TLS negotiation
    # inside the first measured request on each worker -- which is exactly
    # what the warm-up protocol claims to have removed.
    n_workers = _pool_size(a)
    pool = ThreadPoolExecutor(max_workers=n_workers)

    try:
        # ---- warm-up: through the same workers, never measured -------------
        warm_n = max(a.warmup, n_workers)
        if warm_n > 0:
            print(f"warm-up: {warm_n} queries across {n_workers} workers "
                  f"(not measured)", flush=True)
            # The first n_workers tasks rendezvous at a barrier, which forces
            # every worker thread to be alive and busy simultaneously. Without
            # it a fast task could be served by one thread repeatedly and the
            # rest would still be cold when measurement starts.
            barrier = threading.Barrier(n_workers, timeout=180)

            def warm(i: int) -> None:
                if i < n_workers:
                    try:
                        barrier.wait()
                    except threading.BrokenBarrierError:
                        pass
                q = queries[i % len(queries)]
                try:
                    # Traced under a warmup- id rather than untraced, so the
                    # cold-start cost stays visible in the file but is excluded
                    # from every metric by name.
                    with trace.query(f"warmup-{i}", run_id_value=run_id,
                                     warmup=True, worker_warm=i < n_workers):
                        pipe.run(q["query"])
                except Exception as exc:  # noqa: BLE001
                    print(f"  warm-up {i} failed: {exc}", flush=True)

            for f in [pool.submit(warm, i) for i in range(warm_n)]:
                f.result()
            llm.COUNTER.reset()
            # The cascade's tier counters must cover the same window as the
            # LLM counters, or coverage is reported over warm-up plus
            # measurement while calls are reported over measurement alone.
            pipe.reset_parser_stats()

        # ---- measurement ---------------------------------------------------
        # Devices come from the SERVER's launch record, not from this shell's
        # environment. `make serve` runs with CUDA_VISIBLE_DEVICES=0 in one
        # shell; `make bench` runs in another that may have none set, and the
        # sampler would then collect all three GPUs on a shared box -- charging
        # for one while measuring someone else's work on the other two.
        sampler = gpu.GpuSampler(devices=telemetry_devices).start()
        manifest.started_at = time.time()
        wall_t0 = time.perf_counter()

        if a.mode == "offline":
            futures = [pool.submit(one, q) for q in queries]
            for i, _ in enumerate(as_completed(futures), 1):
                if i % 25 == 0:
                    print(f"  {i}/{len(queries)}", flush=True)
        else:
            rng = random.Random(a.seed)
            # Poisson arrivals: exponential gaps at mean 1/rate. burstiness
            # scales the shape -- 1.0 is exponential, higher is more regular.
            futures = []
            start = time.perf_counter()
            for i, q in enumerate(queries):
                due = start + _arrival_offset(rng, a.rate, a.burstiness, i)
                delay = due - time.perf_counter()
                if delay > 0:
                    time.sleep(delay)
                futures.append(pool.submit(one, q))
            for f in futures:
                f.result()

        wall_s = time.perf_counter() - wall_t0
    finally:
        pool.shutdown(wait=True)
    manifest.finished_at = time.time()
    sampler.stop()
    trace.end_run()

    counter = llm.COUNTER.snapshot()
    manifest.served_model = ", ".join(facts.models_advertised) or "unknown"

    # ---- write everything before judging it -------------------------------
    with open(os.path.join(out_dir, "results.jsonl"), "w", encoding="utf-8") as fh:
        for r in results:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(out_dir, "failures.jsonl"), "w", encoding="utf-8") as fh:
        for r in failures:
            fh.write(json.dumps(r, default=str) + "\n")
    with open(os.path.join(out_dir, "gpu.json"), "w", encoding="utf-8") as fh:
        json.dump(sampler.summary(), fh, indent=2)
    pstats = pipe.parser_stats()
    with open(os.path.join(out_dir, "counters.json"), "w", encoding="utf-8") as fh:
        json.dump({**counter, "wall_s": wall_s, "parser_kind": a.parser,
                   "parser_tiers": pstats}, fh, indent=2)
    manifest.to_json(os.path.join(out_dir, "manifest.json"))

    wf_fail, infra_fail = validity.split_failures(failures)
    print(f"\nwall {wall_s:.1f}s  ok {len(results)}  "
          f"failed {len(failures)} "
          f"({len(wf_fail)} workflow, {len(infra_fail)} infrastructure)")
    for f in wf_fail:
        print(f"  workflow  [{f['query_id']}] {f['error'][:110]}")
    for f in infra_fail:
        print(f"  INFRA     [{f['query_id']}] {f['error'][:110]}")
    print(f"llm calls {counter['llm_calls']} "
          f"({counter['llm_failures']} failed) {counter['llm_calls_by_agent']}")
    if pstats:
        t1, t2 = pstats.get("tier1", 0), pstats.get("tier2", 0)
        tot = t1 + t2
        print(f"parser   tier1 {t1}/{tot} ({100*t1/max(1,tot):.1f}% no model call)"
              f"  tier2 {t2}  {pstats.get('tier2_reasons') or ''}")

    # ---- FIX 8: parser correctness, reported not gated ---------------------
    try:
        from agentops import parser_eval
        vocab = parser_eval.load_vocab(os.path.join(ROOT, "data", "vocab.json"))
        pscore = parser_eval.score(results, vocab, failures=failures)
        with open(os.path.join(out_dir, "parser_eval.json"), "w", encoding="utf-8") as fh:
            json.dump(pscore, fh, indent=2, default=str)
        sh, dv = pscore["shipped"], pscore["derived"]
        print(f"parser   holding-count {_p(sh['holding_count'])}  "
              f"lookback {_p(sh['lookback_value'])}  "
              f"ticker-set {_p(dv['ticker_set'])} (derived)")
        if pscore["n_failed_scored"]:
            print(f"         (includes {pscore['n_failed_scored']} failed "
                  f"quer{'y' if pscore['n_failed_scored'] == 1 else 'ies'} "
                  f"-- scored over attempted, not successful)")
    except Exception as exc:  # noqa: BLE001 - scoring must not void a good run
        print(f"parser scoring failed: {exc}")

    # ---- advisor quality: SCORED, and for A2 arms, GATED ------------------
    #
    # This block used to be wrapped in `except Exception: print(...)`, on the
    # same reasoning as the parser block above it: a scoring bug should not
    # throw away GPU time that produced a good run. That reasoning holds for a
    # metric and fails for a gate. A2 exists to trade briefing length against
    # cost; if the thing measuring the briefing can vanish silently, the trade
    # is unpriced and the arm reports only its saving. So the score is required
    # here, and check_run decides what to do with it.
    from agentops import advisor_eval
    aq = advisor_eval.score_run(results, trace=_read_trace(out_dir))
    with open(os.path.join(out_dir, "advisor_eval.json"), "w",
              encoding="utf-8") as fh:
        json.dump(aq, fh, indent=2, default=str)
    if aq.get("n"):
        print(f"advisor  {aq['mean_words']:.0f} words  "
              f"{aq['mean_sentences']:.1f} sentences  "
              f"truncated {100*aq['truncation_rate']:.1f}%  "
              f"all-topics {100*aq['all_topics_rate']:.1f}%  "
              f"numbers grounded "
              f"{100*(aq['grounded_fraction_mean'] or 0):.1f}%")

    aref = _read_advisor_reference(a.advisor_reference)
    if a.advisor_reference and aref is None:
        print(f"advisor reference {a.advisor_reference} has no readable "
              f"advisor_eval.json", file=sys.stderr)

    checks = validity.check_run(
        manifest, {**counter, "parser_tiers": pstats}, results, failures,
        advisor=aq, advisor_reference=aref)
    with open(os.path.join(out_dir, "validity.json"), "w", encoding="utf-8") as fh:
        json.dump([c.__dict__ for c in checks], fh, indent=2)

    try:
        validity.enforce(checks, strict=not a.no_strict)
    except validity.ValidityError as exc:
        print(f"\n{exc}", file=sys.stderr)
        return 1

    print(f"\nwrote {out_dir}")
    return 0


def _read_trace(out_dir: str) -> list[dict]:
    rows = []
    p = os.path.join(out_dir, "trace.jsonl")
    if not os.path.exists(p):
        return rows
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _read_advisor_reference(path: str | None) -> dict | None:
    """The reference arm's advisor_eval.json, if one was declared."""
    if not path:
        return None
    p = path if path.endswith(".json") else os.path.join(path, "advisor_eval.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _name_table_sha(a) -> str:
    """A1's name table is an experimental input, so bind it into the manifest.

    Same standard as the price snapshot: without this, someone could swap
    data/names.json, re-run A1 and produce a manifest that looks identical
    while the coverage number came from a different table.
    """
    if a.parser != "cascade":
        return ""
    path = os.path.join(ROOT, "data", "names.json")
    if not os.path.exists(path):
        return ""
    with open(path, encoding="utf-8") as fh:
        return json.load(fh).get("names_sha256", "")


def _pool_size(a) -> int:
    """Worker count for the run.

    Offline: exactly the concurrency under test -- that IS the treatment.

    Online: generous but bounded. Undersizing distorts the arrival process
    (submissions queue in the client instead of at the server), which corrupts
    the very thing an arrival-rate sweep measures. Oversizing only costs
    warm-up time, since every worker is warmed before the clock starts -- so
    the default errs high and caps at 64 rather than growing without limit.
    """
    if a.mode == "offline":
        return a.concurrency
    if a.online_workers > 0:
        return a.online_workers
    return max(8, min(int(a.rate * 8), 64))


def _p(d: dict) -> str:
    acc = d.get("accuracy")
    return "n/a" if acc is None else f"{100*acc:.1f}% (n={d['n']})"


def _arrival_offset(rng: random.Random, rate: float, burstiness: float, i: int) -> float:
    """Cumulative arrival time for request i.

    Recomputed as a running sum by the caller's loop index would drift, so we
    keep a module-level accumulator per call instead.
    """
    global _ACC
    if i == 0:
        _ACC = 0.0
    gap = rng.gammavariate(burstiness, 1.0 / (rate * burstiness))
    _ACC += gap
    return _ACC


_ACC = 0.0


if __name__ == "__main__":
    raise SystemExit(main())
