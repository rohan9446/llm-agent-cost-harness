# One command per stage. Every target is safe to re-run.
PY ?= python3
N  ?= 100
C  ?= 8
STAGE ?= B0
APC ?= off

.PHONY: help bootstrap smoke snapshot serve calibrate bench report verify audit recheck compare clean

help:
	@echo "make bootstrap  one-time setup + preflight; run this first on the lab box"
	@echo "make smoke      harness check, no GPU or network needed"
	@echo "make snapshot   build the frozen price snapshot (once, first)"
	@echo "make serve      launch vLLM for B0 (prefix caching OFF)"
	@echo "make calibrate  C=1 run, supplies attribution weights for the report"
	@echo "make bench      run the frozen systems_100 set  [C=8 STAGE=B0 APC=off]"
	@echo "make report     metrics for the newest run"
	@echo "make verify     confirm the handout patches still apply cleanly"
	@echo "make audit      what would be committed, and does any of it leak"
	@echo "make recheck    re-judge archived runs under the current validity rules"
	@echo "make compare    is the headline table a valid comparison at all"

bootstrap:
	./scripts/bootstrap.sh

smoke:
	$(PY) scripts/smoke_test.py

snapshot:
	$(PY) scripts/build_snapshot.py

serve:
	./scripts/serve_vllm.sh

calibrate:
	$(PY) scripts/run_bench.py --stage $(STAGE)cal --concurrency 1 \
	  --query-set data/query_sets/systems_100.json \
	  --expect-apc $(APC) --max-failure-rate 0

bench:
	$(PY) scripts/run_bench.py --stage $(STAGE) --concurrency $(C) \
	  --query-set data/query_sets/systems_100.json \
	  --expect-apc $(APC) --max-failure-rate 0

# Attribution weights are fitted at C=1 and carried into the C=8 report, because
# at C>1 the measured times contain queueing delay. If no calibration run
# exists the report still works and labels its own weights as contaminated.
# STAGE is part of the selector, not just the label. `make report C=8` used to
# glob results/*-c8-* and take the newest, so running A1 and then asking for a
# B0 report handed back the A1 directory with a B0 heading. The calibration run
# is matched on the same stage for the same reason: attribution weights fitted
# on one system do not describe another.
report:
	@d=$$(ls -dt results/$(STAGE)-*-c$(C)-* 2>/dev/null | head -1); \
	 cal=$$(ls -dt results/$(STAGE)cal-*-c1-* 2>/dev/null | head -1); \
	 if [ -z "$$d" ]; then \
	   echo "no run directory matching results/$(STAGE)-*-c$(C)-*"; \
	   echo "available:"; ls -1dt results/*/ 2>/dev/null | head -8; \
	   exit 1; \
	 fi; \
	 if [ -n "$$cal" ]; then \
	   echo "reporting $$d with weights from $$cal"; \
	   $(PY) scripts/report.py $$d --weights-from $$cal; \
	 else \
	   echo "reporting $$d (no C=1 calibration found -- run 'make calibrate')"; \
	   $(PY) scripts/report.py $$d; \
	 fi

# Run this before every `git add`. Section 1b needs data/queries.json present
# to work at all, so run it on the lab box, not from a clone.
audit:
	./scripts/pre_commit_audit.sh

recheck:
# rep2, not rep1: the A1 rep1 run predates advisor_eval.py and has no briefing
# scores, so it cannot be the thing a brevity arm is held against.
	$(PY) scripts/rescore_parser.py --write
	$(PY) scripts/recheck_runs.py --write \
	  --advisor-reference results/A1-offline-n1000-c8-rep2
	$(PY) scripts/results_index.py --write

# Every headline number is a DIFFERENCE between two runs. This asks whether
# those runs differ only in the treatment being studied.
compare:
	$(PY) scripts/compare_runs.py --headline

verify:
	@echo "--- upstream does not compile (expected) ---"
	-@$(PY) -m py_compile docs/upstream/portfolio/agents/metrics_agent.py
	@echo "--- patched tree compiles ---"
	@$(PY) -m py_compile $$(find workflow -name '*.py') && echo OK
	@echo "--- diff vs upstream ---"
	-@diff -ru -x '__pycache__' -x '*.pyc' docs/upstream/portfolio workflow/portfolio

clean:
	rm -rf results/*/ logs/*.log
	find . -name '__pycache__' -type d -prune -exec rm -rf {} + 2>/dev/null || true
	find . -name '*.pyc' -delete 2>/dev/null || true
