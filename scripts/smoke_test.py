#!/usr/bin/env python3
"""
End-to-end harness check with no GPU and no network.

Builds a synthetic snapshot, starts the mock model server, runs the real
benchmark driver against both, and then checks two things:

  1. the pipeline works -- queries complete, traces contain spans and LLM
     calls, per-agent attribution is populated, results carry parses
  2. the guard works -- validity FAILS on the synthetic snapshot

The second is the point. A smoke run satisfies every other check perfectly,
so without snapshot provenance it would look exactly like a real measurement.
Run this before every GPU session; it takes seconds and it is the difference
between debugging the harness on a laptop and debugging it on booked hardware.

    python scripts/smoke_test.py
"""

from __future__ import annotations

import hashlib
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

PORT = int(os.environ.get("SMOKE_PORT", "8765"))


def build_fixture_snapshot(out_dir: str) -> str:
    """Deterministic fake price history, labelled so it can never pass as real."""
    with open(os.path.join(ROOT, "data", "vocab.json"), encoding="utf-8") as fh:
        tickers = sorted(set(json.load(fh)["aliases"].values()))

    os.makedirs(out_dir, exist_ok=True)
    entries = {}
    import datetime
    end = datetime.date(2026, 8, 1)

    for t in tickers:
        rng = random.Random(sum(ord(c) for c in t))
        n = 2600
        price = 50 + rng.random() * 200
        closes, dates = [], []
        for i in range(n):
            price = max(1.0, price * (1.0 + rng.gauss(0.0003, 0.015)))
            closes.append(round(price, 2))
            dates.append((end - datetime.timedelta(days=n - i)).isoformat())
        data = {"ticker": t, "dates": dates, "closes": closes, "source": "snapshot"}
        path = os.path.join(out_dir, f"{t.replace('.', '-')}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        # Same digest preflight.verify_snapshot recomputes, so the fixture
        # exercises the integrity check rather than tripping it.
        from agentops.preflight import _digest
        entries[t] = {"file": os.path.basename(path), "rows": n,
                      "provider_symbol": t.replace(".", "-"),
                      "first": dates[0], "last": dates[-1],
                      "sha256": _digest(data)}

    manifest = {
        "built_at": "SMOKE-TEST",
        "source": "SYNTHETIC-TEST-FIXTURE",   # <- what the guard keys on
        "n_tickers": len(entries),
        "tickers": entries,
        "snapshot_id": "smoketest",
        "_warning": "Fabricated prices. Any number derived from this is meaningless.",
    }
    with open(os.path.join(out_dir, "MANIFEST.json"), "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    return manifest["snapshot_id"]


def main() -> int:
    tmp = tempfile.mkdtemp(prefix="b0-smoke-")
    snap = os.path.join(tmp, "snapshot")
    out = os.path.join(tmp, "results")
    os.makedirs(out, exist_ok=True)

    print("building synthetic snapshot fixture")
    build_fixture_snapshot(snap)

    print(f"starting mock model server on :{PORT}")
    import mock_llm_server
    srv = mock_llm_server.serve(PORT)
    time.sleep(0.4)

    # a server record that mimics what serve_vllm.sh writes
    with open(os.path.join(out, "server.json"), "w", encoding="utf-8") as fh:
        json.dump({"vllm_version": "MOCK", "model": mock_llm_server.MODEL,
                   "prefix_caching": "off", "port": PORT,
                   "cuda_visible_devices": "0", "dtype": "bfloat16",
                   "tensor_parallel_size": 1,
                   # No pid: the mock runs in a thread of this process, not as
                   # a vLLM process, so /proc verification cannot apply. The
                   # run therefore passes --allow-unverified-server, and the
                   # manifest records that it did. _argv_agrees is covered
                   # directly below instead.
                   "argv": ["MOCK"]}, fh, indent=2)

    env = dict(os.environ)
    env.update({
        "LLM_BASE_URL": f"http://127.0.0.1:{PORT}/v1",
        "LLM_MODEL": mock_llm_server.MODEL,
        "LLM_API_KEY": "EMPTY",
        "PYTHONPATH": ROOT,
    })
    env.pop("ADVISOR_ALLOW_TEMPLATE", None)

    cmd = [sys.executable, os.path.join(HERE, "run_bench.py"),
           "--stage", "SMOKE", "--n", "12", "--concurrency", "4",
           "--warmup", "2", "--snapshot", snap, "--out", out,
           "--server-record", os.path.join(out, "server.json"),
           "--expect-apc", "off", "--max-failure-rate", "0",
           "--allow-unverified-server",
           "--query-set", os.path.join(ROOT, "data", "query_sets", "smoke_12.json"),
           "--no-strict"]
    print("running the real benchmark driver against the mock\n")
    proc = subprocess.run(cmd, env=env, capture_output=True, text=True)
    print(proc.stdout)
    if proc.stderr.strip():
        print("stderr:", proc.stderr.strip()[:2000])

    srv.shutdown()

    run_dirs = [d for d in os.listdir(out)
                if d.startswith("SMOKE-") and os.path.isdir(os.path.join(out, d))]
    if not run_dirs:
        print("FAIL: no run directory produced")
        return 1
    run_dir = os.path.join(out, run_dirs[0])

    failures = []

    def check(name, cond, detail=""):
        print(f"  [{'PASS' if cond else 'FAIL'}] {name}{(' -- ' + detail) if detail else ''}")
        if not cond:
            failures.append(name)

    print("\nharness checks")

    results = _jsonl(os.path.join(run_dir, "results.jsonl"))
    check("queries completed", len(results) >= 10, f"{len(results)}")

    trace_rows = _jsonl(os.path.join(run_dir, "trace.jsonl"))
    kinds = {}
    for r in trace_rows:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    check("trace has spans", kinds.get("span", 0) > 0, str(kinds))
    check("trace has llm calls", kinds.get("llm", 0) > 0, str(kinds))
    check("trace has query records", kinds.get("query", 0) > 0, str(kinds))

    agents = {r["agent"] for r in trace_rows if r.get("kind") == "span"}
    expected_agents = {"ParserAgent", "PriceAgent", "MetricsAgent",
                       "RiskAgent", "AdvisorAgent"}
    check("all five agents attributed", expected_agents <= agents,
          f"missing {sorted(expected_agents - agents)}" if not expected_agents <= agents
          else f"{len(agents)} agents")

    llm_rows = [r for r in trace_rows if r.get("kind") == "llm" and not r.get("error")]
    check("llm calls carry token counts",
          all(r["prompt_tokens"] > 0 and r["completion_tokens"] > 0 for r in llm_rows),
          f"{len(llm_rows)} calls")
    check("TTFT captured",
          all(r.get("ttft_s") is not None for r in llm_rows),
          f"{sum(1 for r in llm_rows if r.get('ttft_s') is not None)}/{len(llm_rows)}")
    check("TPOT captured",
          sum(1 for r in llm_rows if r.get("tpot_s")) > 0)
    check("both LLM stages present",
          {"ParserAgent", "AdvisorAgent"} <= {r["agent"] for r in llm_rows})

    check("results carry the parse",
          all(r.get("parsed", {}).get("holdings") for r in results))
    check("prices came from the snapshot",
          all(m.get("source") == "snapshot"
              for r in results for m in (r.get("metrics") or {}).values()))
    check("lookback policy recorded",
          all(r.get("lookback_source") in ("stated", "policy_default") for r in results))

    print("\nguard check -- this one must FAIL for the guard to be working")
    checks = json.load(open(os.path.join(run_dir, "validity.json"), encoding="utf-8"))
    by_name = {c["name"]: c for c in checks}
    snap_check = by_name.get("snapshot_is_real_data", {})
    check("synthetic snapshot rejected", snap_check.get("ok") is False,
          snap_check.get("detail", "check missing"))

    others = [c for c in checks if c["name"] != "snapshot_is_real_data" and not c["ok"]]
    check("every other validity check passed",
          not others, "; ".join(f"{c['name']}: {c['detail']}" for c in others))

    print("\nregression checks for the review findings")
    spans = [r for r in trace_rows if r.get("kind") == "span"]
    check("spans carry exclusive self_s",
          all("self_s" in r for r in spans) and
          any(r["self_s"] < r["wall_s"] - 1e-9 for r in spans
              if r["agent"] == "MetricsAgent"),
          "MetricsAgent self_s < wall_s (PriceAgent time subtracted)")

    m = json.load(open(os.path.join(run_dir, "manifest.json"), encoding="utf-8"))
    check("snapshot checksums verified", m.get("snapshot_verified") is True)
    check("expected APC recorded", m.get("expected_prefix_caching") == "off")
    check("query set frozen", bool(m.get("query_set_id")), m.get("query_set_name", ""))
    check("infrastructure failure tolerance is zero",
          m.get("max_failure_rate") == 0.0)
    check("workflow failure cap recorded separately",
          isinstance(m.get("max_workflow_failure_rate"), float),
          f"cap={m.get('max_workflow_failure_rate')}")
    check("live server was probed",
          (m.get("server_probe") or {}).get("model_present") is True)
    check("counter reports requested_models, not served",
          "requested_models" in json.load(
              open(os.path.join(run_dir, "counters.json"), encoding="utf-8")))

    check("telemetry scoped to the server's GPUs",
          m.get("telemetry_devices") == m.get("server_devices") == "0",
          f"telemetry={m.get('telemetry_devices')!r} server={m.get('server_devices')!r}")
    check("unverified server is recorded, not hidden",
          m.get("allow_unverified_server") is True)

    # ---- provenance the audit found missing ------------------------------
    check("query CONTENT hashed, not just the id list",
          bool(m.get("query_content_sha256")) and bool(m.get("corpus_sha256"))
          and m.get("query_content_sha256") != m.get("query_set_id"),
          f"content={str(m.get('query_content_sha256'))[:12]} "
          f"ids={str(m.get('query_set_id'))[:12]}")
    check("source tree hashed",
          len(m.get("source_tree_sha256") or "") == 64,
          str(m.get("source_tree_sha256"))[:16])
    check("effective generation config captured",
          isinstance(m.get("generation"), dict)
          and m["generation"].get("max_attempts", 0) >= 1
          and "seed" in m["generation"],
          str(m.get("generation", {}).get("max_attempts")))
    check("seed in the manifest is the seed the run used",
          m.get("seed") == (m.get("generation") or {}).get("seed"),
          f"manifest={m.get('seed')} effective="
          f"{(m.get('generation') or {}).get('seed')}")

    # ---- the stage label has to bind ---------------------------------------
    import importlib.util as _ilu
    _spec = _ilu.spec_from_file_location("_rb", os.path.join(HERE, "run_bench.py"))
    _rb = _ilu.module_from_spec(_spec)
    _spec.loader.exec_module(_rb)

    class _A:
        pass

    def _stage(stage, parser="llm", style="handout", ref=None,
               freeform=False, argv=()):
        a = _A()
        a.stage, a.parser, a.advisor_style = stage, parser, style
        a.advisor_reference, a.stage_freeform = ref, freeform
        old, sys.argv = sys.argv, ["run_bench.py", *argv]
        try:
            _rb.apply_stage(a)
            return a, None
        except SystemExit as exc:
            return None, str(exc)
        finally:
            sys.argv = old

    a1, _ = _stage("A1")
    check("stage A1 selects the cascade without being told",
          a1 is not None and a1.parser == "cascade",
          "a stage name that does not choose the parser is a directory prefix")
    cal, _ = _stage("A1cal")
    check("calibration inherits its stage's configuration",
          cal is not None and cal.parser == "cascade",
          "B0cal weights applied to an A1 run would describe a different system")
    bad, why = _stage("A1", argv=["--parser", "llm"])
    check("a flag contradicting the stage is refused",
          bad is None and "cascade" in (why or ""), (why or "").splitlines()[0])
    noref, why2 = _stage("A2-short")
    check("an Advisor arm must name the run it claims to match",
          noref is None and "reference" in (why2 or ""),
          (why2 or "").splitlines()[0])
    undef, why3 = _stage("S3")
    check("an undefined stage is refused rather than labelled",
          undef is None, (why3 or "").splitlines()[0])

    # ---- A2's quality gates must actually reject --------------------------
    from agentops.validity import RunManifest as _RM, _advisor_gates
    from agentops import advisor_eval as _ae0
    # The binding fields have to be present on both sides now: a bare
    # advisor_eval dict can no longer establish that two arms are the same
    # experiment, which is the point of the checks further down.
    _EXPFIELDS = dict(model="M", served_model="M", snapshot_id="S",
                      query_content_sha256="Q" * 64, corpus_sha256="C" * 64,
                      query_set_id="K" * 64, prefix_caching="off",
                      server_devices="0")
    _m = _RM(run_id="t", stage="A2-terse", n_queries=1000, concurrency=8,
             advisor_style="terse", advisor_reference="results/A1",
             **_EXPFIELDS)
    # Every fixture carries the scorer stamp, because a comparison without one
    # is now refused -- which is the point of the checks further down.
    _ref = {"n": 1000, "truncated_briefings": 0, "truncation_rate": 0.0,
            "all_topics_rate": 0.997, "grounded_fraction_mean": 0.966,
            "_scorer_sha256": _ae0.scorer_sha256()}
    _good = dict(_ref, all_topics_rate=0.993, grounded_fraction_mean=0.964)
    _refman = dict(_EXPFIELDS, n_queries=1000)

    def _bundle(adv):
        return {"advisor": adv, "manifest": _refman, "path": "results/A1"}
    _bad = dict(_ref, all_topics_rate=0.766)
    _cut = dict(_ref, truncated_briefings=7, truncation_rate=0.007)
    check("a non-inferior brevity arm passes its gates",
          all(c.ok for c in _advisor_gates(_m, _good, _bundle(_ref))))
    check("an arm that drops a required topic is REJECTED",
          not all(c.ok for c in _advisor_gates(_m, _bad, _bundle(_ref))),
          "A2-terse was 69% cheaper and lost a topic in a quarter of briefings")
    check("an arm whose briefings are cut off is REJECTED",
          not all(c.ok for c in _advisor_gates(_m, _cut, _bundle(_ref))))
    check("a missing advisor score fails instead of passing quietly",
          not all(c.ok for c in _advisor_gates(_m, None, _bundle(_ref))),
          "scoring wrapped in try/except made the worst arm the likeliest "
          "to skip its own check")
    check("a missing reference fails instead of passing quietly",
          not all(c.ok for c in _advisor_gates(_m, _good, None)))

    # ---- same scorer is not the same experiment ---------------------------
    _sc0 = _ae0.scorer_sha256()
    _exp = dict(model="M", served_model="M", snapshot_id="S",
                query_content_sha256="Q" * 64, corpus_sha256="C" * 64,
                query_set_id="K" * 64, n_queries=1000, prefix_caching="off",
                server_devices="0")
    _mb = _RM(run_id="a2", stage="A2-short", n_queries=1000, concurrency=8,
              advisor_style="short", advisor_reference="results/A1",
              **{k: v for k, v in _exp.items() if k != "n_queries"})
    _arm = dict(_ref, all_topics_rate=0.993, grounded_fraction_mean=0.994,
                _scorer_sha256=_sc0)
    _refa = dict(_ref, _scorer_sha256=_sc0)

    def _bound(refman):
        return _advisor_gates(_mb, _arm, {"advisor": _refa, "manifest": refman})

    check("an arm bound to the same experiment passes",
          all(c.ok for c in _bound(dict(_exp))))
    for _lbl, _bad_ref in (("a different corpus", dict(_exp, query_content_sha256="Z" * 64)),
                           ("a different model", dict(_exp, model="OTHER")),
                           ("a different snapshot", dict(_exp, snapshot_id="OTHER")),
                           ("a smaller workload", dict(_exp, n_queries=100))):
        check(f"a reference on {_lbl} is REFUSED",
              not all(c.ok for c in _bound(_bad_ref)),
              "same scorer version does not make two populations comparable")
    check("scores without a manifest cannot establish the binding",
          not all(c.ok for c in _advisor_gates(_mb, _arm, _refa)),
          "with the numbers alone there is no way to know what produced them")

    # ---- warm-up outcome is recorded and gated ----------------------------
    from agentops.validity import check_run as _crv2
    _cv2 = {"llm_calls": 20, "llm_failures": 0, "requested_models": ["M"],
            "llm_calls_by_agent": {"AdvisorAgent": 10, "ParserAgent": 10}}
    _mw = _RM(run_id="w", stage="B0", n_queries=10, concurrency=4,
              n_workers=4, warmed_workers=4, warmup_attempted=4, warmup_failed=0)
    _wc = {c.name: c for c in _crv2(_mw, _cv2, None, [])}["every_worker_warmed"]
    check("a fully warmed pool passes", _wc.ok and _wc.verifiable)
    _mw.warmup_failed, _mw.warmed_workers = 1, 3
    _wc2 = {c.name: c for c in _crv2(_mw, _cv2, None, [])}["every_worker_warmed"]
    check("a cold worker FAILS the run",
          _wc2.verifiable and not _wc2.ok,
          "warm-up failures were printed, swallowed, and then erased from the "
          "counters by COUNTER.reset()")
    _mw.warmup_attempted = 0
    _wc3 = {c.name: c for c in _crv2(_mw, _cv2, None, [])}["every_worker_warmed"]
    check("a run that never recorded warm-up is a GAP", not _wc3.verifiable)

    # ---- attribution: fits that the data contradicts, and failed calls ----
    from agentops import cost as _cost
    def _synth(n=40, slope=0.001):
        return [{"agent": "A", "prompt_tokens": 200 + i * 20,
                 "completion_tokens": 40 + i * 4,
                 "ttft_s": 0.05 + slope * (200 + i * 20),
                 "latency_s": 0.05 + slope * (200 + i * 20) + 0.2
                              + 0.004 * (39 + i * 4)} for i in range(n)]
    check("a healthy attribution fit reports no problems",
          not _cost.fit_token_weights(_synth(), concurrency=1).problems)
    _degen = _cost.fit_token_weights(_synth(slope=-0.0004), concurrency=1)
    check("a negative token slope is FATAL, not clamped to 'tokens are free'",
          bool(_degen.fatal_problems) and _degen.raw_a_prefill < 0,
          f"clamped {_degen.a_prefill:.0e} hides raw {_degen.raw_a_prefill:.1e}")
    _w0 = _cost.fit_token_weights(_synth(), concurrency=1)
    _tot = _cost.AllocatedCost(wall_s=10, n_queries=10, rate=_cost.GpuRate())
    _sp = _cost.attribute(_synth(10) + [{"agent": "A", "error": "LLMError: x"}],
                          _w0, _tot)
    check("a failed call is bucketed, not redistributed",
          any(s.stage == "unattributed_failed_attempt" for s in _sp),
          "attribute() had a branch for failed calls that report.py made "
          "unreachable by passing only ok_llm")

    # ---- two arms must be scored by the same ruler ------------------------
    from agentops import advisor_eval as _ae
    _sc = _ae.scorer_sha256()
    _stamped = _ae.score_run([{"summary": "Volatility is 12.0% and the return "
                                          "is 8.0%; risk is concentrated.",
                               "holdings": {"AAPL": 1.0}, "metrics": {},
                               "risk": {}}])
    check("the scorer stamps its own checksum into every score",
          len(_sc) == 64 and _stamped.get("_scorer_sha256") == _sc,
          "a quality score without its instrument recorded is not comparable "
          "to another one")
    check("grounding is reported pooled as well as averaged",
          "grounded_fraction_pooled" in _stamped,
          "the mean over briefings moves with briefing LENGTH, which is the "
          "variable A2 changes on purpose")
    _refs = dict(_ref, _scorer_sha256=_sc)
    _stale = dict(_good, _scorer_sha256="0" * 64)
    _unstamped = {k: v for k, v in _good.items() if k != "_scorer_sha256"}
    check("an arm scored by a DIFFERENT scorer version is refused",
          not all(c.ok for c in _advisor_gates(_m, _stale, _bundle(_refs))),
          "A2-short rep1 read 23.6% ungrounded and rep2 read 2.8% for the same "
          "prompt, because one was scored before the tokeniser fix")
    check("an unstamped score is refused rather than assumed compatible",
          not all(c.ok for c in _advisor_gates(_m, _unstamped, _bundle(_refs))))
    _mismatched = [c for c in _advisor_gates(_m, _stale, _bundle(_refs))
                   if c.name in ("advisor_topic_coverage_non_inferior",
                                 "advisor_grounding_non_inferior")]
    check("no verdict is reported across mismatched scorers",
          not _mismatched,
          "reporting a topic or grounding verdict computed from two different "
          "rulers is worse than reporting none")

    # ---- a check with no evidence is a GAP, never a pass -------------------
    #
    # recheck_runs used to fabricate result rows with empty `metrics` so the
    # counts worked out. check_run iterated over zero price reads, found zero
    # bad ones, and wrote "all reads from snapshot" into published artifacts
    # for runs where nothing had been re-read. An empty set satisfies any
    # universal claim.
    from agentops.validity import RunManifest as _RMv, check_run as _crv
    _mv = _RMv(run_id="t", stage="B0", n_queries=10, concurrency=1,
               model="M", snapshot_source="yfinance", snapshot_verified=True,
               prefix_caching="off", expected_prefix_caching="off",
               telemetry_devices="0", server_devices="0", query_set_id="k" * 64,
               advisor_template_fallback_allowed=False,
               server_probe={"reachable": True, "model_present": True,
                             "process_matches_record": True})
    _cv = {"llm_calls": 20, "llm_failures": 0, "requested_models": ["M"],
           "llm_calls_by_agent": {"AdvisorAgent": 10, "ParserAgent": 10}}

    def _price_check(res):
        return {c.name: c for c in _crv(_mv, _cv, res, [])}["prices_from_snapshot"]

    _none = _price_check(None)
    check("no result rows -> price check is a GAP, not a pass",
          not _none.verifiable and not _none.ok,
          "an empty set satisfies any universal claim")
    _empty = _price_check([{"query_id": f"u{i}", "metrics": {}} for i in range(10)])
    check("rows with zero price reads -> also a GAP",
          not _empty.verifiable,
          "nothing to verify is not the same as nothing wrong")
    _real = _price_check([{"query_id": "q", "metrics": {"AAPL": {"source": "snapshot"}}}])
    check("real rows still verify normally", _real.verifiable and _real.ok)
    _bad = _price_check([{"query_id": "q", "metrics": {"AAPL": {"source": "yfinance"}}}])
    check("a non-snapshot read still FAILS", _bad.verifiable and not _bad.ok)

    # ---- provenance, not the auditor's shell ------------------------------
    _mv.advisor_template_fallback_allowed = None
    _t = {c.name: c for c in _crv(_mv, _cv, None, [])}["template_fallback_disabled"]
    check("unrecorded template fallback is a GAP",
          not _t.verifiable,
          "os.environ at check time describes the terminal, not the run")
    _mv.advisor_template_fallback_allowed = True
    _t2 = {c.name: c for c in _crv(_mv, _cv, None, [])}["template_fallback_disabled"]
    check("a run made with the template fallback FAILS, whatever this shell says",
          _t2.verifiable and not _t2.ok)
    _mv.advisor_template_fallback_allowed = False

    # ---- two runs are comparable, or the difference means nothing ---------
    from agentops.validity import comparability as _cmp
    _b = dict(model="M", served_model="M", snapshot_id="S",
              query_content_sha256="Q" * 64, corpus_sha256="C" * 64,
              query_set_id="K" * 64, n_queries=1000, prefix_caching="off",
              server_devices="0")
    check("runs differing only in the treatment are comparable",
          all(c.ok for c in _cmp([("a", _b), ("b", dict(_b))],
                                 varying=("parser_kind",))))
    check("a different corpus is NOT comparable",
          not all(c.ok for c in _cmp(
              [("a", _b), ("b", dict(_b, query_content_sha256="Z" * 64))])),
          "subtraction does not care whether the operands are the same "
          "experiment; this is what the README's -28.4% rests on")
    _blank = {k: ("" if k != "n_queries" else 0) for k in _b}
    _gapc = _cmp([("a", _blank), ("b", dict(_blank))])
    check("absence on BOTH sides is not agreement",
          all(not c.verifiable for c in _gapc),
          "two runs that both fail to record the corpus hash are not thereby "
          "shown to have used the same corpus")

    # ---- the argv check must be exhaustive, not a handpicked list ---------
    from agentops.preflight import _argv_agrees as _aa
    _rec = {"model": "M", "prefix_caching": "off", "dtype": "bfloat16",
            "tensor_parallel_size": 1, "port": 8000,
            "argv": ["vllm", "serve", "M", "--dtype", "bfloat16",
                     "--tensor-parallel-size", "1", "--port", "8000",
                     "--no-enable-prefix-caching", "--max-model-len", "4096",
                     "--gpu-memory-utilization", "0.90", "--seed", "1337"]}
    check("argv check accepts an identical live process",
          _aa(list(_rec["argv"]), _rec)[0])
    for _flag, _old, _new in (("--max-model-len", "4096", "8192"),
                              ("--gpu-memory-utilization", "0.90", "0.95"),
                              ("--seed", "1337", "7")):
        _live = [x if x != _old else _new for x in _rec["argv"]]
        check(f"argv check catches drifted {_flag}",
              not _aa(_live, _rec)[0],
              "the named-flag list checked five settings while the docstring "
              "claimed it checked every one")
    check("argv check catches a flag added to the live process",
          not _aa(_rec["argv"] + ["--enforce-eager"], _rec)[0])

    pe_path = os.path.join(run_dir, "parser_eval.json")
    check("parser correctness scored", os.path.exists(pe_path))
    if os.path.exists(pe_path):
        _pe = json.load(open(pe_path, encoding="utf-8"))
        check("parser scorer and vocab are pinned",
              len(_pe.get("_scorer_sha256") or "") == 64
              and len(_pe.get("_vocab_sha256") or "") == 64,
              "derived accuracy is a property of the alias map and the scorer")
    _vchecks = {c["name"] for c in checks}
    check("parser scoring is a validity check, not a print statement",
          "parser_quality_scored" in _vchecks,
          "scoring wrapped in try/except let a run report cost down, failures "
          "zero and validity PASS with no accuracy artifact at all")
    if os.path.exists(pe_path):
        pe = json.load(open(pe_path, encoding="utf-8"))
        check("parser weights scored",
              pe["derived"].get("weights_within_1e-3", {}).get("n", 0) > 0,
              f"n={pe['derived'].get('weights_within_1e-3', {}).get('n')}")
        check("every derived label resolvable",
              pe["derived"]["n_unresolvable_by_alias_map"] == 0,
              f"{pe['derived']['n_unresolvable_by_alias_map']} unresolvable")

    # live-process verification logic, exercised directly since the mock is not
    # a real vLLM process
    from agentops.preflight import _argv_agrees
    good = ["vllm", "serve", mock_llm_server.MODEL, "--port", str(PORT),
            "--dtype", "bfloat16", "--tensor-parallel-size", "1",
            "--no-enable-prefix-caching"]
    rec = {"model": mock_llm_server.MODEL, "prefix_caching": "off",
           "dtype": "bfloat16", "tensor_parallel_size": 1, "port": PORT}
    ok, _ = _argv_agrees(good, rec)
    check("argv check accepts a matching live process", ok)

    apc_on = [x for x in good if x != "--no-enable-prefix-caching"] + \
             ["--enable-prefix-caching"]
    ok2, why = _argv_agrees(apc_on, rec)
    check("argv check catches APC on while the record says off",
          ok2 is False, why)

    wrong_tp = good[:]
    wrong_tp[wrong_tp.index("--tensor-parallel-size") + 1] = "2"
    ok3, why3 = _argv_agrees(wrong_tp, rec)
    check("argv check catches a changed tensor-parallel size", ok3 is False, why3)

    warm_rows = [r for r in trace_rows if r.get("kind") == "query"
                 and str(r.get("query_id", "")).startswith("warmup-")]
    worker_warms = [r for r in warm_rows
                    if (r.get("meta") or {}).get("worker_warm")]
    check("every worker thread warmed before the clock started",
          len(worker_warms) >= m.get("n_workers", 0) > 0,
          f"{len(worker_warms)} barrier-synchronised warm-ups for "
          f"{m.get('n_workers')} workers")
    check("warm-up ran through the measuring pool",
          m.get("warmed_workers", 0) >= m.get("n_workers", 0) > 0,
          f"warmed={m.get('warmed_workers')} workers={m.get('n_workers')}")

    # ---- failure classification -------------------------------------------
    # The split exists so that a parser hallucination is reported as a baseline
    # property while a dead server still voids the run. If the classifier drifts
    # in either direction, one of those two guarantees is silently gone.
    from agentops.validity import classify_failure, split_failures
    check("snapshot miss classified as workflow",
          classify_failure("PipelineError: SnapshotMiss: MCD is not in the "
                           "frozen snapshot") == "workflow")
    check("bare snapshot miss classified as workflow",
          classify_failure("SnapshotMiss: SFM is not in the frozen snapshot")
          == "workflow")
    check("nested infra error beats the outer PipelineError",
          classify_failure("PipelineError: LLMError: Connection refused")
          == "infrastructure",
          "Pipeline.run wraps everything, so matching the outer type alone "
          "would file a dead server as a tolerated workflow failure")
    check("nested timeout beats the outer PipelineError",
          classify_failure("PipelineError: TimeoutError: read timed out")
          == "infrastructure")
    check("connection error classified as infrastructure",
          classify_failure("ConnectionError: [Errno 111] refused")
          == "infrastructure")
    check("unknown error classified as infrastructure (fails closed)",
          classify_failure("RuntimeError: something nobody has seen")
          == "infrastructure",
          "an unclassified failure must stop a run, not pass as a finding")
    _wf, _infra = split_failures([
        {"error": "PipelineError: SnapshotMiss: X"},
        {"error": "APIError: 503"},
    ])
    check("split_failures partitions without dropping rows",
          len(_wf) == 1 and len(_infra) == 1)

    # A parse error bad enough to crash the workflow must still be graded, or
    # parser accuracy flatters itself precisely on its worst errors.
    from agentops import parser_eval as _pe
    _vocab = _pe.load_vocab(os.path.join(ROOT, "data", "vocab.json"))
    # Fixture sentences are authored, not lifted from the corpus. An earlier
    # version used a real corpus query here, which put supplied text into a
    # published source file -- the same leak as the artifacts, through the
    # tests.
    _ok_row = {"query_id": "q1", "query": "AAPL 100%.",
               "parsed": {"holdings": {"AAPL": 1.0}, "lookback_days": None},
               "label": {"n_holdings": 1, "phrasing": "t",
                         "expected_lookback_days": None}}
    _bad_row = {"query_id": "q2", "query": "MSFT 100%.",
                "error": "PipelineError: SnapshotMiss: MCD not in snapshot",
                "parsed": {"holdings": {"MCD": 1.0}, "lookback_days": None},
                "label": {"n_holdings": 1, "phrasing": "t",
                          "expected_lookback_days": None}}
    _s_ok = _pe.score([_ok_row], _vocab)
    _s_all = _pe.score([_ok_row], _vocab, failures=[_bad_row])
    check("failed query with a parse is scored",
          _s_all["n_failed_scored"] == 1 and _s_all["n_scored"] == 2,
          f"scored {_s_all['n_scored']}, of which {_s_all['n_failed_scored']} failed")
    check("including the failure lowers ticker accuracy",
          _s_all["derived"]["ticker_set"]["accuracy"]
          < _s_ok["derived"]["ticker_set"]["accuracy"],
          f"{_s_all['derived']['ticker_set']['accuracy']:.2f} vs "
          f"{_s_ok['derived']['ticker_set']['accuracy']:.2f}")
    # ---- exclusion that correlates with the measurement -------------------
    #
    # Redaction strips query text from failures.jsonl. Failures are
    # disproportionately the parser's OWN errors -- a hallucinated ticker is
    # what crashes the pipeline. Re-scoring the redacted file therefore drops
    # the worst cases out of the denominator and raises the reported accuracy;
    # it took B0 from 95.3% to 98.0% with nothing measured differently.
    _redacted_fail = {"query_id": "q9", "query_redacted": True,
                      "error": "PipelineError: SnapshotMiss: MCD",
                      "parsed": {"holdings": {"MCD": 1.0}},
                      "label": {"n_holdings": 1}}
    _sc = _pe.score([_ok_row], _vocab, failures=[_redacted_fail])
    check("a row with no query text is counted, not silently dropped",
          _sc.get("n_rows_without_query_text", 0) >= 1,
          f"n_rows_without_query_text={_sc.get('n_rows_without_query_text')}")
    from agentops.validity import RunManifest as _RM2, check_run as _cr
    _m2 = _RM2(run_id="t", stage="B0", n_queries=1, concurrency=1)
    _names = {c.name: c for c in _cr(_m2, {"llm_calls": 0, "llm_failures": 0,
                                           "llm_calls_by_agent": {}},
                                     [], [], parser=_sc)}
    check("unscoreable rows fail the run rather than flattering it",
          _names["parser_scored_every_attempt"].ok is False,
          "excluding the parser's hallucinations from its own accuracy is the "
          "original bug of this module arriving by a new route")

    check("failure without a parse is not scored",
          _pe.score([_ok_row], _vocab,
                    failures=[{"query_id": "q3", "error": "APIError: 503"}]
                    )["n_failed_scored"] == 0,
          "an infrastructure failure before the parser ran has nothing to grade")

    # ---- A1 cascade: the two bugs the offline eval caught ------------------
    # Both were silent. The weight transposition is the dangerous one: every
    # ticker correct, holding count correct, lookback correct, and the wrong
    # portfolio analysed. Nothing downstream would have flagged it.
    import tempfile as _tf
    _cdir = _tf.mkdtemp(prefix="cascade-")
    os.makedirs(os.path.join(_cdir, "snap"), exist_ok=True)
    _LONG = {"V": "Visa Inc.", "PFE": "Pfizer Inc.", "NFLX": "Netflix, Inc.",
             "ADBE": "Adobe Inc.", "COST": "Costco Wholesale Corporation",
             "BA": "The Boeing Company",
             "DIS": "The Walt Disney Company", "KO": "The Coca-Cola Company",
             "JPM": "JP Morgan Chase & Co.", "INTC": "Intel Corporation",
             "META": "Meta Platforms, Inc.", "AMZN": "Amazon.com, Inc.",
             "ORCL": "Oracle Corporation"}
    json.dump({"source": "smoke-fixture",
               "names": {t: {"ticker": t, "longName": n, "shortName": n}
                         for t, n in _LONG.items()}},
              open(os.path.join(_cdir, "names.json"), "w"))
    json.dump({"tickers": {t: {} for t in _LONG}, "snapshot_id": "smoke"},
              open(os.path.join(_cdir, "snap", "MANIFEST.json"), "w"))
    sys.path.insert(0, os.path.join(ROOT, "workflow", "portfolio", "agents"))
    from parser_cascade import NameIndex, parse_holdings, parse_lookback
    _idx = NameIndex(os.path.join(_cdir, "names.json"),
                     os.path.join(_cdir, "snap"))

    # The sentence is invented, not lifted. An earlier version of this fixture
    # used the corpus query that first exposed the bug, which put a supplied
    # sentence into a published file -- the same leak as the template list,
    # hiding inside the regression test for a different bug entirely. The
    # property under test is structural, so it does not need the real sentence:
    # a percentage sits closer to the WRONG entity, and only sequence order
    # pairs them correctly.
    _h = parse_holdings("Weight me 61% in Adobe, 14% in Costco "
                        "and Boeing 25%.", _idx)
    check("cascade pairs weights by sequence, not distance",
          _h is not None and abs(_h["ADBE"] - 0.61) < 1e-6
          and abs(_h["COST"] - 0.14) < 1e-6 and abs(_h["BA"] - 0.25) < 1e-6,
          f"{_h} -- distance pairing binds 14% to Adobe, which is nearer to it "
          f"than 61% is; this shape covers a quarter of the corpus")
    check("cascade handles percent-after-holding",
          (parse_holdings("Split it JPM 28% and 72% in Pfizer.", _idx)
           or {}).get("JPM") == 0.28)
    check("cascade reads an unquantified window",
          parse_lookback("over the last month") == (30, True),
          str(parse_lookback("over the last month")))
    check("cascade reads a quantified window",
          parse_lookback("over the past 18 months") == (545, True),
          "18 months must be 545 days, the shipped label convention")
    check("cascade declines an unreadable window",
          parse_lookback("since the IPO")[1] is False,
          "emitting None would assert the user stated no window")
    check("cascade matches a shortened provider name",
          (parse_holdings("Evaluate Disney and KO.", _idx) or {}).keys()
          == {"DIS", "KO"},
          "'Disney' must reach 'The Walt Disney Company'")
    check("cascade declines an out-of-universe company",
          parse_holdings("Evaluate Rivian and Snowflake.", _idx) is None,
          "declining is correct: the LLM tier exists for exactly this")
    # Routing safety: the two rules added after the 91-query robustness set
    # measured an 8.8% false-accept rate. Both exist because false accept and
    # false decline are not symmetric -- one analyses the wrong portfolio, the
    # other costs a single LLM call.
    check("cascade declines when a holding cannot be resolved",
          parse_holdings("Evaluate Visa, Rivian and Pfizer.", _idx) is None,
          "resolving [V, PFE] and dropping Rivian silently analyses a "
          "portfolio nobody asked for -- the exact failure A1 was built to end")
    check("cascade declines an unpriceable symbol",
          parse_holdings("Split it 50% Visa and 50% GOOG.", _idx) is None,
          "GOOG is a real ticker outside the universe; dropping it is not an "
          "option and neither is guessing GOOGL")
    check("cascade declines weights that do not sum to a portfolio",
          parse_holdings("Review 120% Visa and 30% Pfizer.", _idx) is None,
          "normalising 120/30 to 80/20 invents an intent nobody expressed")
    check("weight rule tolerates honest rounding",
          parse_holdings("Review 33% Visa, 33% Pfizer and 33% Netflix.",
                         _idx) is not None,
          "33/33/33 sums to 99 and must still be accepted")

    _mixed = parse_holdings("60% in Visa and Pfizer.", _idx)
    for _bad, _why in [
        ("Analyze intelligent systems.", "intel inside intelligent"),
        ("What is the metadata risk?", "meta inside metadata"),
        ("Compare amazonian exposure.", "amazon inside amazonian"),
        ("Evaluate oracle-like systems.", "oracle before a hyphen"),
    ]:
        check(f"cascade rejects a name fragment: {_why}",
              parse_holdings(_bad, _idx) is None,
              "substring matching without word boundaries invents holdings "
              "in sentences nobody wrote")
    # ---- unknown company at the START of a clause -------------------------
    #
    # The most dangerous bug found in this project, and it survived three
    # audits. unknown_entity_evidence() skipped every sentence-initial capital
    # as grammar, so an unrecognised company in first position was invisible
    # and the remaining holdings were silently re-weighted into a portfolio
    # nobody asked for. Exactly what A1 exists to prevent, in the one position
    # the guard did not look.
    for _q in ("Rivian, Visa and Pfizer over the last month",
               "Snowflake and Visa over the last year",
               "Portfolio: Rivian, Visa and Pfizer over the last month",
               "RIVN, Visa and Pfizer over the last month",
               "Rivian 40% and Visa 60% over the last month",
               "Snowflake and 40% Visa over the last year"):
        check(f"cascade declines a clause-initial unknown: {_q.split(',')[0][:26]}",
              parse_holdings(_q, _idx) is None,
              "dropping the unknown name and re-weighting the rest is a wrong "
              "portfolio delivered confidently")
    # ...without turning every leading verb into a holding. This is the half
    # that broke first: a percentage AFTER a leading word binds FORWARD to the
    # holding that follows it, and reading it backwards declined 44 of the
    # 1,000 corpus queries and took Tier-1 coverage from 100% to 95.6%.
    check("a leading verb before a percentage is not a holding",
          parse_holdings("Evaluate 100% Visa over the last 6 months.", _idx)
          == {"V": 1.0},
          "'Evaluate 100% Visa' -- the 100% belongs to Visa, not to Evaluate")
    check("a recognised holding in first position still resolves",
          parse_holdings("Visa and Pfizer over the last month", _idx) is not None)

    check("cascade still matches a closed-up multiword name",
          (parse_holdings("Split it 40% JPMorgan and 60% Visa.", _idx)
           or {}).get("JPM") == 0.40,
          "'JPMorgan' must still reach 'JP Morgan Chase & Co.'")
    check("cascade declines partial weighting",
          _mixed is None,
          "one percentage for two holdings is ambiguous; guessing would trade "
          "a visible cost for an invisible error")
    shutil.rmtree(_cdir, ignore_errors=True)

    # ---- A2 quality gates, validated against known-good/known-bad text ----
    # The first real run of the grounding gate would otherwise be ambiguous: a
    # low score could mean the model invents figures OR that the checker is
    # broken. These two cases separate those readings before any A2 arm runs.
    from agentops import advisor_eval
    _h = {"AAPL": 0.6, "MSFT": 0.4}
    _m = {"AAPL": {"annualized_return": 0.18, "annualized_volatility": 0.25,
                   "sharpe": 0.72, "max_drawdown": -0.15},
          "MSFT": {"annualized_return": 0.14, "annualized_volatility": 0.22,
                   "sharpe": 0.64, "max_drawdown": -0.12}}
    _r = {"portfolio_annualized_return": 0.164,
          "portfolio_annualized_volatility": 0.21,
          "portfolio_sharpe": 0.78, "concentration_hhi": 0.52,
          "diversification_ratio": 1.12,
          "top_holding": {"ticker": "AAPL", "weight": 0.6}}

    _good = ("The portfolio returned 16.4% annualized against 21.0% volatility "
             "(Sharpe 0.78). AAPL is the largest position at 60.0%, leaving the "
             "book concentrated with an HHI of 0.52. Diversification ratio is "
             "1.12, so combining AAPL and MSFT reduces standalone risk only "
             "modestly. Max drawdown reached -15.0%.")
    _sg = advisor_eval.score_briefing(_good, _h, _m, _r)
    check("grounding gate accepts figures the workflow computed",
          _sg["n_ungrounded"] == 0 and _sg["n_numbers"] >= 6,
          f"{_sg['n_numbers']} numbers, {_sg['n_ungrounded']} ungrounded "
          f"{_sg['ungrounded']}")
    check("all four topics detected in a complete briefing",
          _sg["all_topics"], str(_sg["topics"]))
    check("holdings named", _sg["all_holdings_named"])

    # The regression for the tokenizer bug: a hyphenated range must not be
    # read as a negative number. This artifact alone accounted for the entire
    # apparent 6.6% fabrication rate on the first real scoring run.
    _range = advisor_eval.score_briefing(
        "In 3-4 words: returned 16.4% at 21.0% volatility, well diversified "
        "and concentrated (Sharpe 0.78).", _h, _m, _r)
    check("hyphenated range is not read as a negative number",
          _range["n_ungrounded"] == 0,
          f"flagged {_range['ungrounded']} -- '3-4' must give 3 and 4, "
          f"never 3 and MINUS 4")
    check("bare counts scored apart from statistics",
          _range["n_bare_integers"] == 2 and _range["n_numbers"] == 3,
          f"{_range['n_bare_integers']} counts, {_range['n_numbers']} figures")

    _bad = ("The portfolio returned 42.7% annualized against 88.3% volatility "
            "(Sharpe 9.91). It is well diversified.")
    _sb = advisor_eval.score_briefing(_bad, _h, _m, _r)
    check("grounding gate flags invented figures",
          _sb["n_ungrounded"] >= 3,
          f"flagged {_sb['ungrounded']} -- a confident brief summary with "
          f"fabricated numbers is worse than a verbose accurate one")
    check("topic gate notices a dropped topic",
          not _sb["all_topics"],
          f"covered {_sb['topics_covered']}/4")

    # fitted fixed overhead must reach the attribution, not vanish
    from agentops.cost import TokenWeights
    tw = TokenWeights(a_prefill=1e-5, b_decode=1e-3,
                      alpha_request_s=0.05, beta_decode_s=0.01)
    check("attribution includes fitted per-request overhead",
          tw.weight(100, 10) > tw.weight(100, 10, include_fixed=False),
          f"{tw.weight(100, 10):.4f} vs {tw.weight(100, 10, include_fixed=False):.4f}")

    # re-running the same tag must be refused, not silently appended to
    rerun = subprocess.run(cmd, env=env, capture_output=True, text=True)
    check("re-run into an existing dir refused",
          rerun.returncode == 2 and "already exists" in rerun.stderr,
          rerun.stderr.strip().splitlines()[0] if rerun.stderr.strip() else "")

    shutil.rmtree(tmp, ignore_errors=True)

    print()
    if failures:
        print(f"SMOKE TEST FAILED: {len(failures)} check(s) -- {failures}")
        return 1
    print("SMOKE TEST PASSED -- harness is wired correctly; nothing here is a measurement")
    return 0


def _jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    out = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


if __name__ == "__main__":
    raise SystemExit(main())
