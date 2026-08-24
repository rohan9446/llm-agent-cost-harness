#!/usr/bin/env python3
"""
Company names for the A1 deterministic parser, from the market data provider.

WHY THIS FILE EXISTS AT ALL
---------------------------
The obvious way to build a rules-first parser is to look each company up in
`data/vocab.json`. That would be worthless as evidence. vocab.json was built by
enumerating the surface forms that appear in *these 1,000 queries* and mapping
each by hand. Measuring a lookup table against the corpus it was transcribed
from is measuring it against its own keys: coverage is 100% by construction,
and the number proves nothing about any query nobody has seen.

So A1's name table comes from somewhere the corpus did not write: yfinance's
own metadata for the tickers in the frozen snapshot. "Does the string
'Mastercard' match the provider's 'Mastercard Incorporated'?" is then a real
question with a real answer, and the queries this fails on are a real finding
rather than an artifact of how the table was made.

WHAT IS AND IS NOT ASSUMED KNOWN
--------------------------------
The *ticker universe* is legitimately known -- it is the frozen snapshot, and
any real deployment knows which assets it can price. Matching a bare symbol
like AAPL against that universe is fair.

The *surface forms* are not known. Nothing here enumerates how a user might
write a company name; the matcher normalises the provider's official name and
tries to match against it. Where that fails, the cascade falls through to the
LLM, which is the whole point of a cascade.

This script does NOT touch data/snapshot/. Prices stay frozen, every run
already recorded stays valid, and names.json carries its own provenance.

    python scripts/fetch_names.py
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


def provider_symbol(ticker: str) -> str:
    """Our ticker -> the provider's spelling. BRK.B is BRK-B at Yahoo."""
    return ticker.replace(".", "-")


def tickers_from_snapshot(snapshot_dir: str) -> list[str]:
    """The universe comes from the snapshot, not from the query corpus.

    This matters for the same reason the rest of the file exists: reading the
    ticker list out of vocab.json would import the corpus's own answer key.
    The snapshot is what the system can price, which a deployment knows
    independently of what anyone asks it.
    """
    manifest = os.path.join(snapshot_dir, "MANIFEST.json")
    with open(manifest, encoding="utf-8") as fh:
        m = json.load(fh)
    t = m.get("tickers")
    if isinstance(t, dict):          # build_snapshot writes {ticker: {...}}
        return sorted(t.keys())
    if isinstance(t, list):
        return sorted(t)
    raise SystemExit(f"cannot read a ticker universe from {manifest}")


def fetch_one(ticker: str, tries: int = 3) -> dict:
    import yfinance as yf
    symbol = provider_symbol(ticker)
    last: Exception | None = None
    for attempt in range(1, tries + 1):
        try:
            info = yf.Ticker(symbol).info or {}
            long_name = (info.get("longName") or "").strip()
            short_name = (info.get("shortName") or "").strip()
            if not long_name and not short_name:
                raise RuntimeError("provider returned no name fields")
            return {
                "ticker": ticker,
                "provider_symbol": symbol,
                "longName": long_name,
                "shortName": short_name,
            }
        except Exception as exc:  # noqa: BLE001
            last = exc
            print(f"  {ticker}: attempt {attempt} failed ({exc}); retrying",
                  flush=True)
            time.sleep(2 * attempt)
    raise RuntimeError(f"{ticker}: no name after {tries} attempts: {last}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--snapshot", default=os.path.join(ROOT, "data", "snapshot"))
    ap.add_argument("--out", default=os.path.join(ROOT, "data", "names.json"))
    ap.add_argument("--force", action="store_true",
                    help="refetch even though a names file already exists")
    a = ap.parse_args()

    if os.path.exists(a.out) and not a.force:
        print(f"{a.out} already exists; refusing to refetch.\n"
              f"A mid-experiment change to the name table would silently change "
              f"A1's coverage against runs already recorded. Use --force if that "
              f"is genuinely what you want.")
        return 1

    tickers = tickers_from_snapshot(a.snapshot)
    print(f"fetching provider names for {len(tickers)} tickers "
          f"(universe from the frozen snapshot, NOT from queries.json)")

    names = {}
    for t in tickers:
        rec = fetch_one(t)
        names[t] = rec
        print(f"  {t:<6} {rec['longName'] or rec['shortName']}")

    try:
        import yfinance
        yf_version = getattr(yfinance, "__version__", "unknown")
    except Exception:  # noqa: BLE001
        yf_version = "unknown"

    payload = {
        "_provenance": (
            "Fetched from yfinance Ticker.info for the tickers in the frozen "
            "price snapshot. NOT derived from queries.json in any way. This is "
            "what makes A1's coverage a measurement rather than a tautology -- "
            "see the module docstring in scripts/fetch_names.py."),
        "_universe_source": "data/snapshot/MANIFEST.json",
        "source": "yfinance Ticker.info (longName, shortName)",
        "yfinance_version": yf_version,
        "fetched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_tickers": len(names),
        "names": names,
    }
    payload["names_sha256"] = hashlib.sha256(
        json.dumps(names, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2, ensure_ascii=False)

    print(f"\nwrote {a.out}  ({len(names)} names, sha256 "
          f"{payload['names_sha256'][:16]})")
    print("record that sha256 in METHODS.md -- A1 coverage is only comparable "
          "across runs that used the same name table")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
