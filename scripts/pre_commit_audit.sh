#!/usr/bin/env bash
# What would actually get committed, and does any of it leak?
#
# Run this BEFORE `git add`. It is cheaper to find a hostname now than to
# rewrite history after a push, and a public repository is the one place where
# "I'll clean it up later" is not true.
#
#   ./scripts/pre_commit_audit.sh
#
# HISTORY, kept because it explains section 1b.
# The first version of this script checked FILENAMES. It asked whether
# data/queries.json was in the commit set and answered no, correctly, while
# four other files carried the same sentences inside them: failures.jsonl on
# every row, report.json inside reliability.*_failure_detail, parser_eval.py as
# a transcribed list of the corpus's sentence openings, and perturbed.json
# wholesale. A filename check cannot see any of that. It passed four times
# while the corpus was being published.
#
# So section 1b reads the corpus and looks for its WORDS in the bytes that
# would be committed. That is the only check here that could have caught what
# actually happened.
#
set -uo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

FAIL=0
ok()   { printf '  \033[32m[ ok ]\033[0m %s\n' "$1"; }
bad()  { printf '  \033[31m[FAIL]\033[0m %s\n' "$1"; FAIL=1; }
warn() { printf '  \033[33m[warn]\033[0m %s\n' "$1"; }

echo "== 0. housekeeping =="
STRAY=$(find . -path ./.venv -prune -o \( -name "*.tar.gz" -o -name "__pycache__" \
        -o -name ".ipynb_checkpoints" -o -name "nohup.out" \) -print 2>/dev/null | wc -l)
if [[ "$STRAY" -gt 0 ]]; then
  warn "$STRAY stray build artifacts present (gitignored, but clutter the tree)"
  echo "     remove with: find . -path ./.venv -prune -o \\( -name '*.tar.gz' \\"
  echo "       -o -name '__pycache__' -o -name '.ipynb_checkpoints' \\) -print0 \\"
  echo "       | xargs -0 rm -rf"
else
  ok "no stray tarballs or caches"
fi

if [[ ! -d .git ]]; then
  warn "not a git repo yet -- run: git init && git add -A"
  echo "     then re-run this script to audit the actual staged set"
fi

# The file list git would really use, so the audit matches reality rather than
# a find(1) approximation of it.
if [[ -d .git ]]; then
  FILES=$(git ls-files --cached --others --exclude-standard)
else
  FILES=$(git -c init.defaultBranch=main init -q . >/dev/null 2>&1; \
          git ls-files --others --exclude-standard)
fi

echo
echo "== 1. supplied material must NOT be tracked =="
for f in data/queries.json docs/upstream \
         workflow/portfolio/agents/price_agent.py \
         workflow/portfolio/agents/metrics_agent.py \
         workflow/portfolio/agents/risk_agent.py \
         workflow/portfolio/agents/advisor_agent.py \
         workflow/portfolio/workflows/portfolio_workflow.py; do
  if grep -qx -- "$f" <<<"$FILES" || grep -q "^$f/" <<<"$FILES"; then
    bad "$f WOULD BE COMMITTED -- this is Canyon Code's material"
  fi
done
[[ "$FAIL" -eq 0 ]] && ok "no supplied workflow or corpus in the commit set"

echo
echo "== 1b. supplied TEXT must not appear inside any committed file =="
echo "     (the check that section 1 cannot make: filenames vs contents)"
CORPUS_HITS=$(python3 - "$FILES" <<'PY'
import json, os, re, sys

CORPUS = "data/queries.json"
if not os.path.exists(CORPUS):
    print("!!NOCORPUS")
    raise SystemExit(0)

# Shingle length. Six words is long enough that ordinary English -- "the risk
# of the portfolio over the past" -- will not collide by accident, and short
# enough to catch a fragment of a query embedded in an error message or a
# regex. It is deliberately shorter than a whole sentence: the leak that
# actually happened was sentence OPENINGS, not whole queries.
K = 6

def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())

shingles = {}
with open(CORPUS, encoding="utf-8") as fh:
    for q in json.load(fh):
        text = q.get("query", "")
        w = words(text)
        for i in range(max(0, len(w) - K + 1)):
            shingles.setdefault(" ".join(w[i:i + K]), q.get("id"))

if not shingles:
    print("!!NOCORPUS")
    raise SystemExit(0)

files = [f for f in sys.argv[1].split("\n") if f.strip() and os.path.isfile(f)]
hits = []
for path in files:
    try:
        with open(path, "rb") as fh:
            blob = fh.read()
    except OSError:
        continue
    if b"\0" in blob[:4096]:
        continue                       # binary
    w = words(blob.decode("utf-8", "replace"))
    seen = set()
    for i in range(max(0, len(w) - K + 1)):
        sh = " ".join(w[i:i + K])
        qid = shingles.get(sh)
        if qid is not None and sh not in seen:
            seen.add(sh)
            hits.append((path, qid, sh))
            if len(seen) >= 3:
                break

for path, qid, sh in hits[:20]:
    print(f"{path}: matches query {qid}: ...{sh}...")
print(f"!!COUNT {len({h[0] for h in hits})} {len(hits)}")
PY
)
if grep -q '^!!NOCORPUS' <<<"$CORPUS_HITS"; then
  warn "data/queries.json not present -- cannot check contents against the corpus"
  echo "     This check is the one that matters. Run it on a machine that has"
  echo "     the corpus before pushing, or you are auditing filenames again."
else
  SUMMARY=$(grep '^!!COUNT' <<<"$CORPUS_HITS")
  NFILES=$(awk '{print $2}' <<<"$SUMMARY")
  if [[ "${NFILES:-0}" -gt 0 ]]; then
    bad "supplied query text found inside $NFILES committed file(s):"
    grep -v '^!!' <<<"$CORPUS_HITS" | sed 's/^/       /'
    echo "       fix with: python scripts/redact_artifacts.py --apply"
    echo "       or exclude the file in .gitignore -- but check WHY it has"
    echo "       corpus text before deciding which."
  else
    ok "no 6-word run from any supplied query appears in the commit set"
  fi
fi

echo
echo "== 1c. JSON artifacts must not carry a query field =="
JHITS=$(python3 - "$FILES" <<'PY'
import json, os, sys

KEYS = {"query", "queries", "query_text"}

def walk(o, path=""):
    if isinstance(o, dict):
        for k, v in o.items():
            if k in KEYS and isinstance(v, str) and v.strip():
                yield f"{path}.{k}"
            else:
                yield from walk(v, f"{path}.{k}")
    elif isinstance(o, list):
        for i, x in enumerate(o[:200]):
            yield from walk(x, f"{path}[{i}]")

out = []
for p in [f for f in sys.argv[1].split("\n") if f.strip()]:
    if not (p.endswith(".json") or p.endswith(".jsonl")) or not os.path.isfile(p):
        continue
    try:
        with open(p, encoding="utf-8") as fh:
            docs = ([json.loads(l) for l in fh if l.strip()]
                    if p.endswith(".jsonl") else [json.load(fh)])
    except Exception:
        continue
    found = [w for d in docs for w in walk(d)]
    if found:
        out.append(f"{p}: {len(found)} query field(s), e.g. {found[0]}")
print("\n".join(out))
PY
)
if [[ -n "$JHITS" ]]; then
  bad "query text present as a JSON value:"; sed 's/^/       /' <<<"$JHITS"
  echo "       fix with: python scripts/redact_artifacts.py --apply"
else
  ok "no query fields in committed JSON"
fi

echo
echo "== 2. secrets =="
HITS=$(grep -rInE "postgres(ql)?://[^ ]*:[^ ]*@|mysql://[^ ]*:[^ ]*@|AKIA[0-9A-Z]{16}|hf_[A-Za-z0-9]{34,}|sk-[A-Za-z0-9]{32,}|BEGIN [A-Z ]*PRIVATE KEY" \
       $(echo "$FILES" | tr '\n' ' ') 2>/dev/null | head -5)
if [[ -n "$HITS" ]]; then bad "credential-shaped strings:"; sed 's/^/       /' <<<"$HITS"
else ok "no credentials"; fi

echo
echo "== 3. hostnames, IPs, personal paths =="
HITS=$(grep -rInE "\b([0-9]{1,3}\.){3}[0-9]{1,3}\b|/home/[a-z]+/|jupyter-[a-z0-9-]+|[a-z0-9._%-]+@[a-z0-9.-]+\.[a-z]{2,}" \
       $(echo "$FILES" | tr '\n' ' ') 2>/dev/null \
       | grep -vE "127\.0\.0\.1|0\.0\.0\.0|localhost|example\.(com|org)|huggingface\.co|/home/claude/" \
       | head -8)
if [[ -n "$HITS" ]]; then
  warn "host or path references -- check each is generic, not identifying:"
  sed 's/^/       /' <<<"$HITS"
  echo "       (/home/jovyan is the standard Jupyter user and is fine;"
  echo "        a cluster hostname or an email address is not)"
else ok "no hostnames, IPs or personal paths"; fi

echo
echo "== 4. run manifests: hostname must be hashed, never recorded =="
if compgen -G "results/*/manifest.json" >/dev/null; then
  RAW=$(python3 - <<'PY'
import glob, json
bad = []
for p in glob.glob("results/*/manifest.json"):
    env = (json.load(open(p)).get("env") or {})
    h = env.get("host_id", "")
    if not h or len(h) > 16 or not all(c in "0123456789abcdef" for c in h):
        bad.append(f"{p}: host_id={h!r}")
print("\n".join(bad))
PY
)
  [[ -z "$RAW" ]] && ok "$(ls -d results/*/ | wc -l) manifests carry a hashed host_id" \
                  || { bad "manifest host_id is not a hash:"; sed 's/^/       /' <<<"$RAW"; }
else
  warn "no run manifests found"
fi

echo
echo "== 5. size =="
N=$(wc -l <<<"$FILES")
SZ=$(du -ch $(echo "$FILES" | tr '\n' ' ') 2>/dev/null | tail -1 | cut -f1)
echo "     $N files, $SZ total"
BIG=$(du -h $(echo "$FILES" | tr '\n' ' ') 2>/dev/null | sort -rh | head -3)
echo "     largest:"; sed 's/^/       /' <<<"$BIG"

echo
if [[ "$FAIL" -eq 0 ]]; then
  echo "READY to commit. Re-read section 3 by eye -- a grep cannot tell you"
  echo "whether a path identifies you, and this repo is public."
else
  echo "DO NOT COMMIT -- fix the [FAIL] lines above."
  exit 1
fi
