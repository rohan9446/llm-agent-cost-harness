#!/usr/bin/env python3
"""
Build the frozen market-data snapshot. Run once, before any benchmark.

This is a CONTROL, not an optimization. PriceAgent as shipped tries yfinance
and silently falls back to a deterministic synthetic series when the fetch
fails. A rate limit part-way through a 1,000-query sweep would hand some
queries real prices and others synthetic ones, corrupting cost and quality
comparisons at the same time, with nothing in the output to show for it.

So every ticker is fetched exactly once here, checksummed, and written to
disk. Every measured run then reads from this snapshot and asserts it did.
B0 deliberately re-reads and re-slices on every single call -- caching that
work is the A3 experiment, and doing it here would quietly steal A3's result.

    python scripts/build_snapshot.py --out data/snapshot
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)


def tickers_from_vocab(path: str) -> list[str]:
    with open(path, encoding="utf-8") as fh:
        vocab = json.load(fh)
    return sorted(set(vocab["aliases"].values()))


def provider_symbol(ticker: str) -> str:
    """Canonical symbol -> the symbol Yahoo actually quotes.

    The corpus writes BRK.B; Yahoo quotes BRK-B. Keeping the two names apart
    matters: BRK.B stays the canonical id everywhere in the application and the
    evaluation, and only the fetch is translated.
    """
    return ticker.replace(".", "-")


def fetch(ticker: str, years: int, tries: int = 3) -> dict:
    import yfinance as yf

    symbol = provider_symbol(ticker)
    last = None
    for attempt in range(1, tries + 1):
        try:
            # auto_adjust is deliberately NOT passed. Upstream PriceAgent calls
            # history(period=f"{lookback_days}d") with no adjustment argument,
            # so it inherits whatever the installed yfinance defaults to.
            # Passing auto_adjust=False here would have given the snapshot
            # unadjusted closes against upstream's adjusted ones, quietly
            # changing every return around a split or dividend. Matching by
            # omission is more robust than matching by knowing the default.
            hist = yf.Ticker(symbol).history(period=f"{years}y")
            closes = [float(c) for c in hist["Close"].tolist()]
            dates = [str(d.date()) for d in hist.index]
            if len(closes) >= 100:
                return {"ticker": ticker, "provider_symbol": symbol,
                        "dates": dates, "closes": closes, "source": "yfinance"}
            last = RuntimeError(f"only {len(closes)} rows returned")
        except Exception as exc:  # noqa: BLE001
            last = exc
        print(f"  {ticker} (as {symbol}): attempt {attempt} failed ({last}); retrying",
              flush=True)
        time.sleep(3 * attempt)
    raise RuntimeError(
        f"{ticker} (as {symbol}): could not fetch after {tries} attempts: {last}")


def digest(obj: dict) -> str:
    payload = json.dumps(
        {"dates": obj["dates"], "closes": obj["closes"]},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--vocab", default=os.path.join(ROOT, "data", "vocab.json"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "snapshot"))
    ap.add_argument("--years", type=int, default=10,
                    help="history depth; the corpus needs 1825 days, 10y is headroom")
    ap.add_argument("--force", action="store_true",
                    help="rebuild even if a snapshot already exists")
    a = ap.parse_args()

    manifest_path = os.path.join(a.out, "MANIFEST.json")
    if os.path.exists(manifest_path) and not a.force:
        print(f"snapshot already present at {a.out}\n"
              f"refusing to rebuild -- a mid-project refetch would silently "
              f"change the inputs of runs already recorded. Use --force if "
              f"that is genuinely what you want.")
        return 1

    tickers = tickers_from_vocab(a.vocab)
    os.makedirs(a.out, exist_ok=True)
    print(f"fetching {len(tickers)} tickers, {a.years}y each")

    entries = {}
    for t in tickers:
        data = fetch(t, a.years)
        path = os.path.join(a.out, f"{t.replace('.', '-')}.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, separators=(",", ":"))
        entries[t] = {
            "file": os.path.basename(path),
            "provider_symbol": data["provider_symbol"],
            "rows": len(data["closes"]),
            "first": data["dates"][0],
            "last": data["dates"][-1],
            "sha256": digest(data),
        }
        print(f"  {t:6s} {entries[t]['rows']:5d} rows  "
              f"{entries[t]['first']} .. {entries[t]['last']}", flush=True)

    import yfinance as _yf
    manifest = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "years": a.years,
        "source": "yfinance",
        "yfinance_version": _yf.__version__,
        "auto_adjust": "library default (argument omitted, matching upstream)",
        "n_tickers": len(entries),
        "tickers": entries,
    }
    manifest["snapshot_id"] = hashlib.sha256(
        json.dumps({k: v["sha256"] for k, v in entries.items()},
                   sort_keys=True).encode()
    ).hexdigest()[:16]

    with open(manifest_path, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)

    print(f"\nsnapshot_id {manifest['snapshot_id']}")
    print(f"manifest    {manifest_path}")
    print("\nRecord that snapshot_id in every run manifest. If it ever changes, "
          "results from before and after are not comparable.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
