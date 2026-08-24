#!/usr/bin/env python3
"""
Turn a run directory into the metrics the Systems prompt asks for.

Reads the trace and nothing else, so a run can be re-costed under a different
rate without re-executing a single query.

Cost per query is derived, not measured per request. The measured quantity is
the whole window's allocated GPU time; each query's share of it is its share
of the fitted token weight. Dividing the total evenly would hide the fact that
a six-holding five-year query costs more than a one-holding thirty-day one --
and that distribution is exactly what the prompt asks for.

    python scripts/report.py results/B0-offline-n100-c8-rep1
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)

from agentops import cost, validity  # noqa: E402


def jsonl(path: str) -> list[dict]:
    if not os.path.exists(path):
        return []
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def pct(xs, q):
    if not xs:
        return None
    s = sorted(xs)
    i = min(len(s) - 1, int(round(q * (len(s) - 1))))
    return s[i]


def dist(xs) -> dict:
    xs = [x for x in xs if x is not None]
    if not xs:
        return {}
    return {
        "n": len(xs),
        "mean": statistics.fmean(xs),
        "p50": pct(xs, 0.50), "p90": pct(xs, 0.90),
        "p95": pct(xs, 0.95), "p99": pct(xs, 0.99),
        "min": min(xs), "max": max(xs),
    }


def is_warmup(qid) -> bool:
    return isinstance(qid, str) and qid.startswith("warmup-")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir")
    ap.add_argument("--usd-per-gpu-hour", type=float, default=1.60)
    ap.add_argument("--rate-source", default="PLACEHOLDER -- cite a real rate")
    ap.add_argument("--n-gpus", type=int, default=1)
    ap.add_argument("--weights-from", default=None,
                    help="another run dir whose C=1 calls should supply the "
                         "token weights; strongly preferred over fitting on a "
                         "concurrent run")
    a = ap.parse_args()

    d = a.run_dir
    manifest = json.load(open(os.path.join(d, "manifest.json"), encoding="utf-8"))
    counters = json.load(open(os.path.join(d, "counters.json"), encoding="utf-8"))
    gpu = json.load(open(os.path.join(d, "gpu.json"), encoding="utf-8"))
    trace = jsonl(os.path.join(d, "trace.jsonl"))
    results = jsonl(os.path.join(d, "results.jsonl"))
    failures = jsonl(os.path.join(d, "failures.jsonl"))

    rows = [r for r in trace if not is_warmup(r.get("query_id"))]
    llm_rows = [r for r in rows if r.get("kind") == "llm"]
    ok_llm = [r for r in llm_rows if not r.get("error")]
    spans = [r for r in rows if r.get("kind") == "span"]
    queries = [r for r in rows if r.get("kind") == "query"]

    wall_s = counters["wall_s"]
    n_ok = len(results)

    # ---- the measured total ------------------------------------------------
    rate = cost.GpuRate(n_gpus=a.n_gpus, usd_per_gpu_hour=a.usd_per_gpu_hour,
                        source=a.rate_source)
    total = cost.AllocatedCost(wall_s=wall_s, n_queries=n_ok, rate=rate)

    # ---- the estimated split ----------------------------------------------
    weight_src = ok_llm
    weight_conc = manifest.get("concurrency") or 1
    note = ""
    if a.weights_from:
        src = [r for r in jsonl(os.path.join(a.weights_from, "trace.jsonl"))
               if r.get("kind") == "llm" and not r.get("error")
               and not is_warmup(r.get("query_id"))]
        if src:
            m = json.load(open(os.path.join(a.weights_from, "manifest.json"),
                               encoding="utf-8"))
            weight_src, weight_conc = src, m.get("concurrency") or 1
            note = f"weights fitted on {os.path.basename(a.weights_from)}"
    weights = cost.fit_token_weights(weight_src, concurrency=weight_conc)
    weights.note = (weights.note + " " + note).strip()

    stages = cost.attribute(ok_llm, weights, total)

    # ---- per-query cost, in proportion to token work -----------------------
    per_query_w: dict[str, float] = {}
    for r in ok_llm:
        q = r.get("query_id")
        if q is None:
            continue
        per_query_w[q] = per_query_w.get(q, 0.0) + weights.weight(
            r.get("prompt_tokens", 0), r.get("completion_tokens", 0),
            r.get("cached_prompt_tokens", 0))
    sum_w = sum(per_query_w.values()) or 1.0
    per_query_cost = {q: total.total_usd * w / sum_w for q, w in per_query_w.items()}

    # ---- per-agent time, deterministic stages included ---------------------
    # wall_s is INCLUSIVE (MetricsAgent contains PriceAgent), so those numbers
    # do not sum to the total. self_s is exclusive and does.
    agent_time: dict[str, dict[str, float]] = {}
    for s in spans:
        e = agent_time.setdefault(s["agent"], {"calls": 0, "wall_s": 0.0,
                                               "self_s": 0.0, "cpu_s": 0.0})
        e["calls"] += 1
        e["wall_s"] += s.get("wall_s", 0.0)
        e["self_s"] += s.get("self_s", 0.0)
        e["cpu_s"] += s.get("cpu_s", 0.0)
    total_self = sum(v["self_s"] for v in agent_time.values()) or 1.0

    e2e = [q["wall_s"] for q in queries if q.get("ok")]
    prompt_tok = sum(r.get("prompt_tokens", 0) for r in ok_llm)
    completion_tok = sum(r.get("completion_tokens", 0) for r in ok_llm)

    report = {
        "run": {
            "run_id": manifest["run_id"], "stage": manifest["stage"],
            "model": manifest["model"], "concurrency": manifest.get("concurrency"),
            "prefix_caching": manifest.get("prefix_caching"),
            "snapshot_id": manifest.get("snapshot_id"),
            "snapshot_source": manifest.get("snapshot_source"),
            "advisor_max_tokens": manifest.get("advisor_max_tokens"),
            "vllm": (manifest.get("env") or {}).get("vllm"),
            "n_queries_ok": n_ok, "n_queries_failed": len(failures),
            "wall_s": wall_s,
        },
        "reliability": _reliability(n_ok, failures),
        "cost": {
            **total.as_dict(),
            "n_attempted": n_ok + len(failures),
            "cost_per_attempted_query_usd": (
                total.total_usd / (n_ok + len(failures))
                if (n_ok + len(failures)) else float("nan")),
            "_attempted_note": "failed queries consumed GPU time too; dividing "
                               "only by successes understates cost per query. "
                               "Per-attempted is the headline figure: a system "
                               "that answers 97 of 100 queries has not become "
                               "cheaper by discarding the 3",
            "per_query_distribution_usd": dist(list(per_query_cost.values())),
            "note": "total is measured; per-query is the total split by fitted "
                    "token weight, never a per-request latency multiplication",
        },
        "token_weights": weights.as_dict(),
        "cost_by_stage": [
            {"stage": s.stage, "calls": s.calls, "pct_of_llm_cost": 100 * s.share,
             "_meaning": "token-work-weighted share of measured GPU allocation, "
                         "including fitted per-request overhead",
             "attributed_usd": s.attributed_usd, "usd_per_query": s.usd_per_query,
             "prompt_tokens": s.prompt_tokens,
             "completion_tokens": s.completion_tokens,
             "cached_tokens": s.cached_tokens}
            for s in stages
        ],
        "latency_s": {
            "end_to_end": dist(e2e),
            "ttft": dist([r.get("ttft_s") for r in ok_llm]),
            "tpot": dist([r.get("tpot_s") for r in ok_llm]),
            "by_llm_stage": {
                st: dist([r["latency_s"] for r in ok_llm if r["agent"] == st])
                for st in sorted({r["agent"] for r in ok_llm})
            },
        },
        "throughput": {
            "queries_per_s": n_ok / wall_s if wall_s else None,
            "llm_calls_per_s": len(ok_llm) / wall_s if wall_s else None,
            "prompt_tokens_per_s": prompt_tok / wall_s if wall_s else None,
            "output_tokens_per_s": completion_tok / wall_s if wall_s else None,
        },
        "tokens": {
            "prompt": prompt_tok, "completion": completion_tok,
            "cached_prompt": sum(r.get("cached_prompt_tokens", 0) for r in ok_llm),
            "per_query_prompt": prompt_tok / n_ok if n_ok else None,
            "per_query_completion": completion_tok / n_ok if n_ok else None,
            "output_share": completion_tok / max(1, prompt_tok + completion_tok),
        },
        "agents": {
            "_note": "wall_s is inclusive of nested agent calls and does NOT "
                     "sum to 100%; self_s is exclusive and does",
            **{k: {**v,
                   "wall_s_per_query": v["wall_s"] / n_ok if n_ok else None,
                   "self_s_per_query": v["self_s"] / n_ok if n_ok else None,
                   "cpu_s_per_query": v["cpu_s"] / n_ok if n_ok else None,
                   "pct_of_self_time": 100 * v["self_s"] / total_self}
               for k, v in sorted(agent_time.items(), key=lambda kv: -kv[1]["self_s"])},
        },
        "parser_eval": _load_parser_eval(d),
        "gpu": gpu,
        "llm_calls": {
            "total": len(llm_rows), "ok": len(ok_llm),
            "errors": len(llm_rows) - len(ok_llm),
            "by_agent": counters.get("llm_calls_by_agent", {}),
        },
    }

    out = os.path.join(d, "report.json")
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=str)

    _print(report)
    print(f"\nwrote {out}")
    return 0


def _reliability(n_ok: int, failures: list[dict]) -> dict:
    """Failures split by whether they invalidate the measurement or are part
    of what is being measured.

    An infrastructure failure -- server down, timeout, harness bug -- means the
    run did not measure what it claims, so the gate on it is zero. A workflow
    failure -- the parser hallucinated a ticker the snapshot does not hold --
    is a real property of this baseline. Voiding the run for it would delete
    the finding; hiding it in a success rate would flatter it. So it is
    reported here as a rate, and the cost of the GPU time it burned stays in
    the per-attempted figure.
    """
    attempted = n_ok + len(failures)
    wf, infra = validity.split_failures(failures)
    return {
        "n_attempted": attempted,
        "n_ok": n_ok,
        "workflow_failures": len(wf),
        "workflow_failure_rate": len(wf) / attempted if attempted else None,
        "infrastructure_failures": len(infra),
        "infrastructure_failure_rate": len(infra) / attempted if attempted else None,
        "_note": "infrastructure failures invalidate a run (gate: 0); workflow "
                 "failures are a measured property of the baseline",
        # No query text. The supplied corpus is not ours to republish, and
        # report.json is a published artifact -- so a failure is identified by
        # id, and what makes it INTERESTING (the error class and the parse the
        # model produced) is kept. Anyone with authorised access to
        # queries.json can recover the sentence from query_id.
        "workflow_failure_detail": [
            {"query_id": f.get("query_id"), "error": f.get("error"),
             "parsed_holdings": list((f.get("parsed") or {}).get("holdings") or {})}
            for f in wf
        ][:20],
        "infrastructure_failure_detail": [
            {"query_id": f.get("query_id"), "error": f.get("error")} for f in infra
        ][:20],
    }


def _load_parser_eval(run_dir: str):
    p = os.path.join(run_dir, "parser_eval.json")
    if not os.path.exists(p):
        return None
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def _acc(d: dict) -> str:
    a = d.get("accuracy")
    return "n/a" if a is None else f"{100*a:.1f}% (n={d['n']})"


def _print(r: dict) -> None:
    run, c = r["run"], r["cost"]
    print(f"\n{'='*72}\n{run['stage']}  {run['run_id']}\n{'='*72}")
    print(f"model            {run['model']}  vllm={run['vllm']}")
    print(f"concurrency      {run['concurrency']}   prefix_caching={run['prefix_caching']}")
    print(f"snapshot         {run['snapshot_id']} ({run['snapshot_source']})")
    print(f"queries          {run['n_queries_ok']} ok / {run['n_queries_failed']} failed"
          f"   wall {run['wall_s']:.1f}s")

    rel = r.get("reliability")
    if rel:
        print("\n-- reliability (failures split by what they mean) --")
        print(f"  infrastructure  {rel['infrastructure_failures']}/"
              f"{rel['n_attempted']} = "
              f"{100*(rel['infrastructure_failure_rate'] or 0):.1f}%"
              f"   -- any is disqualifying; the run did not measure what it claims")
        print(f"  workflow        {rel['workflow_failures']}/{rel['n_attempted']} = "
              f"{100*(rel['workflow_failure_rate'] or 0):.1f}%"
              f"   -- a measured property of the baseline, not a fault")
        for f in rel["workflow_failure_detail"]:
            print(f"    [{f['query_id']}] {(f['error'] or '')[:96]}")
            if f.get("parsed_holdings"):
                print(f"          parsed: {f['parsed_holdings']}")

    print(f"\n-- MEASURED: cost ({c['basis']}) --")
    print(f"  rate           ${c['usd_per_gpu_hour']}/GPU-h x {c['n_gpus']} GPU")
    print(f"  rate source    {c['rate_source']}")
    print(f"  TOTAL          ${c['total_usd']:.4f}")
    print(f"  per query      ${c['cost_per_query_usd']:.6f}")
    print(f"  per attempted  ${c['cost_per_attempted_query_usd']:.6f}"
          f"   ({c['n_attempted']} attempted, {c['n_queries']} succeeded)")
    d = c["per_query_distribution_usd"]
    if d:
        print(f"\n  ESTIMATED distribution (total split by attribution weight)")
        print(f"    p50 ${d['p50']:.6f}  p90 ${d['p90']:.6f}  "
              f"p95 ${d['p95']:.6f}  p99 ${d['p99']:.6f}  max ${d['max']:.6f}")

    w = r["token_weights"]
    print(f"\n-- ESTIMATED: attribution weights (fitted at C={w['fitted_at_concurrency']}) --")
    print(f"  these divide the measured total; they are NOT GPU-seconds/token")
    print(f"  marginal prefill  {w['marginal_s_per_prefill_token']:.3e} s/token  "
          f"(R2 {w['r2_prefill_fit']:.3f})")
    print(f"  marginal decode   {w['marginal_s_per_decode_token']:.3e} s/token  "
          f"(R2 {w['r2_decode_fit']:.3f})")
    print(f"  fixed per request {w['fixed_request_overhead_s']:.4f} s  "
          f"+ decode entry {w['fixed_decode_entry_s']:.4f} s")
    print(f"  decode:prefill    {w['decode_to_prefill_ratio']:.1f}x  "
          f"(ratio of marginal slopes)")
    if w.get("note"):
        print(f"  note              {w['note']}")
    if (w.get("r2_prefill_fit") or 0) < 0.85:
        print(f"  CAVEAT            the prefill fit is weak (R2 "
              f"{w['r2_prefill_fit']:.3f}). TTFT at these prompt lengths is "
              f"dominated by")
        print(f"                    fixed request overhead, so the prefill slope "
              f"is poorly determined and")
        print(f"                    the prefill share of the split carries real "
              f"uncertainty. The decode")
        print(f"                    side (R2 {w['r2_decode_fit']:.3f}) is where "
              f"the mass is, and it is tight.")

    print("\n-- ESTIMATED: token-work-weighted share of the measured GPU allocation --")
    print("   (not '% of system cost per agent': the deterministic agents burn")
    print("    CPU this split cannot see, and the weights are fitted)")
    print(f"  {'stage':<16}{'%':>7}{'$/query':>12}{'calls':>8}{'in tok':>10}{'out tok':>10}")
    for s in r["cost_by_stage"]:
        print(f"  {s['stage']:<16}{s['pct_of_llm_cost']:>6.1f}%"
              f"{s['usd_per_query']:>12.6f}{s['calls']:>8}"
              f"{s['prompt_tokens']:>10}{s['completion_tokens']:>10}")

    lat = r["latency_s"]
    print("\n-- latency (s) --")
    for name, dd in [("end-to-end", lat["end_to_end"]), ("TTFT", lat["ttft"]),
                     ("TPOT", lat["tpot"])]:
        if dd:
            print(f"  {name:<12} p50 {dd['p50']:.4f}  p95 {dd['p95']:.4f}  "
                  f"p99 {dd['p99']:.4f}  max {dd['max']:.4f}")

    t = r["throughput"]
    print("\n-- throughput --")
    print(f"  {t['queries_per_s']:.2f} queries/s   "
          f"{t['output_tokens_per_s']:.0f} output tok/s   "
          f"{t['prompt_tokens_per_s']:.0f} prompt tok/s")

    print("\n-- per-agent time (wall is inclusive; self is exclusive and additive) --")
    print(f"  {'agent':<16}{'calls':>8}{'wall/q':>10}{'self/q':>10}{'cpu/q':>9}{'% self':>9}")
    for k, v in r["agents"].items():
        if k.startswith("_"):
            continue
        print(f"  {k:<16}{v['calls']:>8}{v['wall_s_per_query']:>10.4f}"
              f"{v['self_s_per_query']:>10.4f}{v['cpu_s_per_query']:>9.4f}"
              f"{v['pct_of_self_time']:>8.1f}%")

    pe = r.get("parser_eval")
    if pe:
        sh, dv = pe["shipped"], pe["derived"]
        print("\n-- parser correctness --")
        print(f"  shipped labels : holding-count {_acc(sh['holding_count'])}   "
              f"lookback {_acc(sh['lookback_value'])}   "
              f"stated-vs-unstated {_acc(sh['lookback_stated_vs_unstated'])}")
        print(f"  derived labels : ticker-set {_acc(dv['ticker_set'])}   "
              f"({dv['n_unresolvable_by_alias_map']} unresolvable by alias map)")
        if pe.get("n_failed_scored"):
            print(f"  scored over ATTEMPTED queries: includes "
                  f"{pe['n_failed_scored']} that failed downstream because of "
                  f"the parse. Excluding them would make this metric most "
                  f"optimistic exactly where the parser is worst.")
        print("  a query can complete perfectly while the parser misread it; "
              "this is what stops that counting as a success")

    g = r["gpu"]
    if g.get("available") and g.get("gpus"):
        print("\n-- gpu --")
        for idx, v in g["gpus"].items():
            print(f"  gpu{idx}  util mean {v['util_gpu_mean']:.1f}%  "
                  f"p95 {v['util_gpu_p95']:.0f}%  busy {100*v['busy_fraction']:.1f}%  "
                  f"vram {v['mem_used_mb_max']:.0f}/{v['mem_total_mb']:.0f} MB")
        print("  NOTE: utilization.gpu is a duty cycle -- percent of the sample")
        print("  period with >=1 kernel resident. It is occupancy-blind, so 100%")
        print("  means the device was never idle, NOT that it was saturated.")
        print("  Decode at small batch is memory-bandwidth-bound with most SMs")
        print("  starved and still reads 100%. Headroom is judged from achieved")
        print("  tokens/s, not from this number.")
    else:
        print(f"\n-- gpu -- unavailable: {g.get('reason')}")


if __name__ == "__main__":
    raise SystemExit(main())
