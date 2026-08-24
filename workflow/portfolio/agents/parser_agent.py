# Parser Agent
#
# NEW STAGE -- not part of the handout. Documented in PATCHES.md section 2.
#
# The handout's entry point takes structured arguments:
#
#     main(holdings: dict, lookback_days: int)
#
# but queries.json is natural language:
#
#     <a natural-language portfolio question>

import json
import os
import re


class ParseError(ValueError):
    """Model output that cannot be read as a portfolio.

    The message deliberately does NOT contain the query. Exception text ends up
    in failures.jsonl and, via the reliability section, in the published
    report.json -- so embedding the supplied sentence here quietly republished
    the corpus through a channel that no redaction keyed on `query` fields
    would ever find. It took three passes to notice: the .gitignore rule, then
    the filename-level audit, then a redactor that scrubbed keys and not
    strings.

    What a failure needs in order to be debuggable is the query_id (recorded
    alongside) and the model's actual output. Neither requires the input text.
    """


SYSTEM = (
    "You extract portfolio parameters from a user's question. "
    "Reply with one JSON object and nothing else."
)

INSTRUCTION = """Extract the holdings and the lookback window from the query.

Rules:
- "holdings" maps stock ticker symbols to weights that sum to 1.0.
- Use the official ticker symbol, never the company name. Berkshire Hathaway
  class B is BRK.B.
- If the query gives percentages, convert them to fractions.
- If the query says the holdings are equal, split evenly.
- If the query lists holdings with no weights at all, split evenly.
- "lookback_days" is the stated window converted to days: a month is 30, a
  quarter or 3 months is 90, 6 months is 180, a year or 12 months is 365,
  18 months is 545, 2 years is 730, 5 years is 1825.
- If the query states no window, set "lookback_days" to null. Do not guess.

Reply with exactly this shape:
{"holdings": {"TICKER": 0.0}, "lookback_days": null}

Query: """


class ParserAgent(object):
    def __init__(self):
        self.tools = [self.parse]
        self.max_tokens = int(os.environ.get("PARSER_MAX_TOKENS", "200"))
        self.temperature = float(os.environ.get("PARSER_TEMPERATURE", "0.0"))

    def parse(self, query: str) -> dict:
        """Natural-language query -> {"holdings": {...}, "lookback_days": int|None}."""
        from agentops import llm

        result = llm.chat(
            [
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": INSTRUCTION + query},
            ],
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            tag="parser",
        )
        return self._decode(result.text, query)

    # -- decoding ---------------------------------------------------------

    def _decode(self, text: str, query: str) -> dict:
        obj = _first_json_object(text)
        if obj is None:
            raise ParseError(f"no JSON object in model output: {text[:200]!r}")

        holdings = obj.get("holdings")
        if not isinstance(holdings, dict) or not holdings:
            raise ParseError(f"no holdings parsed: {text[:200]!r}")

        clean = {}
        for k, v in holdings.items():
            if not isinstance(k, str) or not k.strip():
                continue
            try:
                w = float(v)
            except (TypeError, ValueError):
                raise ParseError(f"non-numeric weight {v!r} for {k!r}")
            if w < 0:
                raise ParseError(f"negative weight {w} for {k!r}")
            clean[k.strip().upper()] = w
        if not clean:
            raise ParseError("holdings empty after cleaning")

        total = sum(clean.values())
        if total <= 0:
            # Every weight zero: the query almost certainly listed bare
            # tickers. Equal-weight rather than fail, and say so.
            clean = {k: 1.0 / len(clean) for k in clean}
        else:
            clean = {k: w / total for k, w in clean.items()}

        lookback = obj.get("lookback_days", None)
        if lookback is not None:
            try:
                lookback = int(lookback)
            except (TypeError, ValueError):
                raise ParseError(f"non-integer lookback {lookback!r}")
            if lookback <= 0:
                raise ParseError(f"non-positive lookback {lookback}")

        return {"holdings": clean, "lookback_days": lookback}


def normalize(parsed: dict, default_lookback_days: int = 365) -> dict:
    """Policy layer: turn a parse into workflow arguments.

    Kept separate from parse() so the parser stays scoreable against the
    shipped nulls. This function is where our declared default lives, and it
    is not ground truth for anything.
    """
    return {
        "holdings": parsed["holdings"],
        "lookback_days": (
            default_lookback_days if parsed["lookback_days"] is None
            else parsed["lookback_days"]
        ),
    }


_JSON_START = re.compile(r"\{")


def _first_json_object(text: str):
    """Pull the first balanced {...} out of the model's reply.

    Small models wrap JSON in prose or fences often enough that a bare
    json.loads() would misreport working parses as failures -- which would
    inflate the very error rate we are trying to measure.
    """
    for m in _JSON_START.finditer(text):
        depth, in_str, esc = 0, False, False
        for i in range(m.start(), len(text)):
            c = text[i]
            if in_str:
                if esc:
                    esc = False
                elif c == "\\":
                    esc = True
                elif c == '"':
                    in_str = False
                continue
            if c == '"':
                in_str = True
            elif c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[m.start():i + 1])
                    except json.JSONDecodeError:
                        break
    return None
