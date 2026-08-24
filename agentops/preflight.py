"""
Checks that run before a benchmark, not after it.

A validity check on the trace can only tell you a run was wrong once the GPU
time is already spent. These two run first and refuse to start:

  * the snapshot's file checksums are recomputed and compared. Writing hashes
    into a manifest and never verifying them means a ticker file edited after
    the snapshot was built would carry the same snapshot_id forever. Thirty
    files is negligible next to a benchmark.

  * the live server is interrogated, not merely declared. results/server.json
    records what the launch script intended; it is written before vLLM starts
    and says nothing about which process currently owns the port. So the model
    list is fetched from the running server and, where the process is visible,
    its command line is read back from /proc.
"""

from __future__ import annotations

import hashlib
import json
import os
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any


class PreflightError(RuntimeError):
    pass


# --------------------------------------------------------------------------
# snapshot integrity
# --------------------------------------------------------------------------

def _digest(obj: dict) -> str:
    payload = json.dumps(
        {"dates": obj["dates"], "closes": obj["closes"]},
        separators=(",", ":"), sort_keys=True,
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def verify_snapshot(snapshot_dir: str) -> dict[str, Any]:
    """Recompute every ticker's digest and compare it to the manifest."""
    mpath = os.path.join(snapshot_dir, "MANIFEST.json")
    if not os.path.exists(mpath):
        raise PreflightError(f"no snapshot manifest at {mpath}")
    with open(mpath, encoding="utf-8") as fh:
        manifest = json.load(fh)

    mismatched, missing = [], []
    for ticker, entry in manifest.get("tickers", {}).items():
        path = os.path.join(snapshot_dir, entry["file"])
        if not os.path.exists(path):
            missing.append(ticker)
            continue
        with open(path, encoding="utf-8") as fh:
            data = json.load(fh)
        if _digest(data) != entry["sha256"]:
            mismatched.append(ticker)

    if missing or mismatched:
        raise PreflightError(
            f"snapshot integrity failed at {snapshot_dir}: "
            f"{len(missing)} missing {missing[:5]}, "
            f"{len(mismatched)} modified {mismatched[:5]}. "
            f"The snapshot_id would still have matched, which is exactly why "
            f"this check exists. Rebuild with --force, or restore the files."
        )

    return {
        "snapshot_id": manifest.get("snapshot_id", ""),
        "source": manifest.get("source", "unknown"),
        "n_tickers": len(manifest.get("tickers", {})),
        "yfinance_version": manifest.get("yfinance_version"),
        "verified": True,
    }


# --------------------------------------------------------------------------
# live server
# --------------------------------------------------------------------------

@dataclass
class ServerFacts:
    reachable: bool
    base_url: str
    models_advertised: list[str]
    model_present: bool
    record_matches_url: bool
    process_argv: list[str] | None
    process_matches_record: bool | None
    process_detail: str
    detail: str

    def as_dict(self) -> dict[str, Any]:
        return self.__dict__.copy()


def _get_json(url: str, timeout: float = 10.0) -> Any:
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return json.loads(r.read().decode())


def probe_server(base_url: str, expected_model: str,
                 server_record: dict | None = None) -> ServerFacts:
    """Ask the running server what it is, instead of trusting a file."""
    base = base_url.rstrip("/")
    record = server_record or {}

    try:
        listing = _get_json(f"{base}/models")
        advertised = [m.get("id") for m in listing.get("data", []) if m.get("id")]
    except (urllib.error.URLError, OSError, ValueError) as exc:
        return ServerFacts(
            reachable=False, base_url=base, models_advertised=[],
            model_present=False, record_matches_url=False,
            process_argv=None, process_matches_record=None,
            process_detail="server unreachable", 
            detail=f"could not reach {base}/models: {exc}",
        )

    # the launch record has to describe the port we are actually talking to
    record_port = record.get("port")
    url_port = None
    for part in base.split(":"):
        head = part.split("/")[0]
        if head.isdigit():
            url_port = int(head)
    record_matches = (record_port is None or url_port is None
                      or int(record_port) == url_port)

    argv, argv_matches, argv_detail = None, None, "no pid recorded"
    pid = record.get("pid")
    if pid:
        try:
            with open(f"/proc/{int(pid)}/cmdline", "rb") as fh:
                argv = [x for x in fh.read().decode().split("\x00") if x]
        except OSError as exc:
            argv, argv_matches = None, False
            argv_detail = (f"cannot read /proc/{pid}/cmdline ({exc}); the "
                           f"recorded server process is not running, or is not "
                           f"visible from here")
        else:
            argv_matches, argv_detail = _argv_agrees(argv, record)

    return ServerFacts(
        reachable=True,
        base_url=base,
        models_advertised=advertised,
        model_present=expected_model in advertised,
        record_matches_url=record_matches,
        process_argv=argv,
        process_matches_record=argv_matches,
        process_detail=argv_detail,
        detail=("ok" if expected_model in advertised
                else f"{expected_model!r} not advertised; server offers {advertised}"),
    )


def _argv_agrees(argv: list[str], record: dict) -> tuple[bool, str]:
    """Does the live command line carry the configuration we claim to be testing?

    Checking only that the model name appears would miss the case that matters
    most: same model, same port, prefix caching left on. Every setting that
    changes what is being measured is checked against the live process.
    """
    line = " ".join(argv)
    problems = []

    model = record.get("model")
    if model and model not in line:
        problems.append(f"model {model!r} absent from live argv")

    apc = record.get("prefix_caching")
    if apc == "off" and "--no-enable-prefix-caching" not in argv:
        problems.append("record says APC off but the live process was not "
                        "launched with --no-enable-prefix-caching")
    if apc == "on" and "--enable-prefix-caching" not in argv:
        problems.append("record says APC on but the live process was not "
                        "launched with --enable-prefix-caching")

    for flag, key in (("--dtype", "dtype"),
                      ("--tensor-parallel-size", "tensor_parallel_size"),
                      ("--port", "port")):
        want = record.get(key)
        if want is None:
            continue
        try:
            got = argv[argv.index(flag) + 1]
        except (ValueError, IndexError):
            problems.append(f"{flag} not present in live argv")
            continue
        if str(got) != str(want):
            problems.append(f"{flag} is {got!r} live but {want!r} in the record")

    return (not problems), ("live process matches the record"
                            if not problems else "; ".join(problems))


def enforce_preflight(facts: ServerFacts, snapshot: dict[str, Any],
                      require_real_data: bool,
                      require_process_match: bool = True) -> None:
    problems = []
    if not facts.reachable:
        # One cause, one message. Port and process checks are meaningless
        # against a server that isn't there, and stacking them buries the
        # actual problem.
        problems.append(facts.detail)
        raise PreflightError("preflight failed:\n  - " + "\n  - ".join(problems))
    if not facts.model_present:
        problems.append(facts.detail)
    if not facts.record_matches_url:
        problems.append(
            f"results/server.json describes port {facts.base_url} differently "
            f"from the URL under test -- the launch record may belong to an "
            f"older server"
        )
    if require_process_match and facts.process_matches_record is not True:
        problems.append(
            f"live server configuration not verified: {facts.process_detail}. "
            f"results/server.json states intent, not fact -- a stale record "
            f"beside a server restarted with different flags would pass every "
            f"other check. Restart via scripts/serve_vllm.sh so the PID is "
            f"recorded, or pass --allow-unverified-server (which is written "
            f"into the manifest)."
        )
    if require_real_data and snapshot.get("source") != "yfinance":
        problems.append(
            f"snapshot source is {snapshot.get('source')!r}, not real market data"
        )
    if problems:
        raise PreflightError("preflight failed:\n  - " + "\n  - ".join(problems))
