#!/usr/bin/env python3
"""
A fake OpenAI-compatible server, for exercising the harness without a GPU.

Streams SSE chunks and a final usage block exactly the way vLLM does, so the
TTFT capture, token accounting, retry path and validity checks all run against
something shaped like the real thing. It answers parser prompts with JSON and
advisor prompts with prose.

This exists so the harness can be debugged on a laptop and so a broken run
never burns a booked GPU window. Numbers it produces are meaningless -- the
snapshot-provenance check is what stops a smoke run being mistaken for a
measurement.

    python scripts/mock_llm_server.py --port 8000
"""

from __future__ import annotations

import argparse
import json
import random
import re
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"

TICKERS = ["AAPL", "MSFT", "AMZN", "GOOGL", "META", "NVDA", "TSLA", "AMD",
           "INTC", "QCOM", "CRM", "ADBE", "ORCL", "CSCO", "NFLX", "DIS",
           "JPM", "BAC", "V", "MA", "BRK.B", "UNH", "JNJ", "PFE", "XOM",
           "CVX", "WMT", "COST", "KO", "PEP"]

NAME_TO_TICKER = {
    "apple": "AAPL", "microsoft": "MSFT", "amazon": "AMZN", "alphabet": "GOOGL",
    "meta": "META", "nvidia": "NVDA", "tesla": "TSLA", "intel": "INTC",
    "qualcomm": "QCOM", "salesforce": "CRM", "adobe": "ADBE", "oracle": "ORCL",
    "cisco": "CSCO", "netflix": "NFLX", "disney": "DIS", "jpmorgan": "JPM",
    "bank of america": "BAC", "visa": "V", "mastercard": "MA",
    "berkshire hathaway": "BRK.B", "unitedhealth": "UNH",
    "johnson & johnson": "JNJ", "pfizer": "PFE", "exxonmobil": "XOM",
    "chevron": "CVX", "walmart": "WMT", "costco": "COST",
    "coca-cola": "KO", "pepsico": "PEP",
}

WINDOWS = [("30 days", 30), ("3 months", 90), ("90 days", 90), ("6 months", 180),
           ("12 months", 365), ("year", 365), ("18 months", 545),
           ("2 years", 730), ("5 years", 1825), ("month", 30)]


def fake_parse(query: str) -> str:
    """Good-enough extraction so downstream stages get realistic input."""
    q = query.lower()
    found: list[tuple[int, str, float | None]] = []

    for m in re.finditer(r"(\d+(?:\.\d+)?)\s*%\s*(?:in\s+)?([A-Za-z][\w.&\- ]*)", query):
        w, name = float(m.group(1)), m.group(2).strip().lower()
        t = _resolve(name)
        if t:
            found.append((m.start(), t, w / 100.0))
    for m in re.finditer(r"([A-Za-z][\w.&\- ]*?)\s+(\d+(?:\.\d+)?)\s*%", query):
        name, w = m.group(1).strip().lower(), float(m.group(2))
        t = _resolve(name)
        if t and not any(f[1] == t for f in found):
            found.append((m.start(), t, w / 100.0))

    if not found:
        for name, t in NAME_TO_TICKER.items():
            i = q.find(name)
            if i >= 0:
                found.append((i, t, None))
        for t in TICKERS:
            i = query.find(t)
            if i >= 0 and not any(f[1] == t for f in found):
                found.append((i, t, None))

    seen, holdings = set(), {}
    for _, t, w in sorted(found):
        if t in seen:
            continue
        seen.add(t)
        holdings[t] = w
    if not holdings:
        holdings = {"AAPL": None}

    if all(v is None for v in holdings.values()):
        holdings = {k: round(1.0 / len(holdings), 6) for k in holdings}
    else:
        holdings = {k: (v if v is not None else 0.0) for k, v in holdings.items()}

    lookback = None
    m = re.search(r"(?:over|for|in|during)\s+the\s+(?:past|last|trailing)\s+([^.,?]*)", q)
    if m:
        tail = m.group(1)
        for phrase, days in sorted(WINDOWS, key=lambda x: -len(x[0])):
            if phrase in tail:
                lookback = days
                break

    return json.dumps({"holdings": holdings, "lookback_days": lookback})


def _resolve(name: str) -> str | None:
    name = name.strip().lower().rstrip(".,")
    if name.upper() in TICKERS:
        return name.upper()
    return NAME_TO_TICKER.get(name)


ADVISOR_TEXT = (
    "The portfolio carries an estimated annualized return against moderate "
    "volatility, producing a Sharpe ratio in the mid range. Concentration is "
    "meaningful, with the largest position accounting for a substantial share "
    "of total weight, so idiosyncratic risk in that name drives much of the "
    "aggregate outcome. The diversification ratio indicates that combining "
    "these holdings reduces standalone risk only modestly, which is consistent "
    "with holdings drawn from overlapping sectors. Rebalancing toward the "
    "smaller positions would lower concentration without materially changing "
    "expected return."
)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):  # quiet
        pass

    def do_GET(self):
        if self.path.rstrip("/").endswith("/v1/models"):
            self._json({"object": "list", "data": [{"id": MODEL, "object": "model"}]})
        elif self.path.rstrip("/").endswith("/health"):
            self._json({"status": "ok"})
        else:
            self.send_error(404)

    def do_POST(self):
        if not self.path.rstrip("/").endswith("/chat/completions"):
            self.send_error(404)
            return
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        messages = body.get("messages", [])
        prompt = "\n".join(m.get("content") or "" for m in messages)

        is_parser = "Extract the holdings" in prompt
        if is_parser:
            query = prompt.split("Query:", 1)[-1].strip()
            text = fake_parse(query)
        else:
            text = ADVISOR_TEXT

        max_tokens = int(body.get("max_tokens") or 400)
        words = text.split()
        # crude token proxy: keep the reply inside the caller's budget
        if len(words) > max_tokens:
            words = words[:max_tokens]
        prompt_tokens = max(1, len(prompt) // 4)
        completion_tokens = max(1, len(" ".join(words)) // 4)

        if body.get("stream"):
            self._stream(words, prompt_tokens, completion_tokens)
        else:
            self._json({
                "id": "cmpl-mock", "object": "chat.completion", "model": MODEL,
                "choices": [{"index": 0, "finish_reason": "stop",
                             "message": {"role": "assistant", "content": " ".join(words)}}],
                "usage": {"prompt_tokens": prompt_tokens,
                          "completion_tokens": completion_tokens,
                          "total_tokens": prompt_tokens + completion_tokens},
            })

    # -- helpers ----------------------------------------------------------

    def _json(self, obj):
        raw = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def _stream(self, words, prompt_tokens, completion_tokens):
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        def chunk(payload: dict):
            data = f"data: {json.dumps(payload)}\n\n".encode()
            self.wfile.write(f"{len(data):X}\r\n".encode() + data + b"\r\n")
            self.wfile.flush()

        base = {"id": "cmpl-mock", "object": "chat.completion.chunk", "model": MODEL}

        # a small prefill delay, then per-token delay, so TTFT and TPOT are
        # distinguishable in the trace rather than both landing at zero
        time.sleep(random.uniform(0.02, 0.05))
        for i, w in enumerate(words):
            chunk({**base, "choices": [
                {"index": 0, "delta": {"content": (" " if i else "") + w},
                 "finish_reason": None}]})
            time.sleep(0.002)

        chunk({**base, "choices": [{"index": 0, "delta": {}, "finish_reason": "stop"}]})
        chunk({**base, "choices": [], "usage": {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens}})

        done = b"data: [DONE]\n\n"
        self.wfile.write(f"{len(done):X}\r\n".encode() + done + b"\r\n")
        self.wfile.write(b"0\r\n\r\n")
        self.wfile.flush()


def serve(port: int) -> ThreadingHTTPServer:
    srv = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    return srv


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8000)
    a = ap.parse_args()
    s = serve(a.port)
    print(f"mock LLM on http://127.0.0.1:{a.port}/v1  (not a measurement device)")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        s.shutdown()
