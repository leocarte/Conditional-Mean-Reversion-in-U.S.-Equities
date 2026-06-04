PYTHON ?= python
PIP ?= $(PYTHON) -m pip

.DEFAULT_GOAL := help

.PHONY: help setup \
	crsp-parquet build-panel refresh-baseline-outputs baseline-smoke \
	linear-benchmarks linear-benchmarks-smoke \
	nonlinear-benchmarks nonlinear-benchmarks-smoke \
	audit-costs-universe audit-costs-universe-smoke \
	compustat-schema-audit compustat-schema-audit-smoke \
	compustat-diagnostics compustat-diagnostics-smoke \
	flow-compustat-linear flow-compustat-linear-smoke \
	flow-compustat-nonlinear flow-compustat-nonlinear-smoke \
	walkforward walkforward-smoke walkforward-aggregate \
	classical-baselines classical-baselines-smoke \
	walkforward-robustness walkforward-robustness-smoke \
	final-report-outputs report-pdf checks test lint clean

help:
	@echo "Available targets:"
	@echo "  setup                         Install project dependencies."
	@echo "  crsp-parquet                  Convert data/raw/monthly_crsp.csv to parquet."
	@echo "  build-panel                   Rebuild the processed panel and baseline outputs."
	@echo "  refresh-baseline-outputs      Recompute baseline outputs from an existing panel."
	@echo "  baseline-smoke                Run the baseline integration smoke test."
	@echo "  linear-benchmarks             Run the locked-split Ridge / Elastic Net pipeline."
	@echo "  linear-benchmarks-smoke       Run the linear benchmark smoke tests."
	@echo "  nonlinear-benchmarks          Run the locked-split XGBoost / MLP pipeline."
	@echo "  nonlinear-benchmarks-smoke    Run the nonlinear benchmark smoke tests."
	@echo "  audit-costs-universe          Run the cost and universe robustness audit."
	@echo "  audit-costs-universe-smoke    Run the cost and universe audit smoke tests."
	@echo "  compustat-schema-audit        Run the Compustat schema feasibility audit."
	@echo "  compustat-schema-audit-smoke  Run the Compustat schema smoke test."
	@echo "  compustat-diagnostics         Run the Compustat integration diagnostics."
	@echo "  compustat-diagnostics-smoke   Run the Compustat diagnostics smoke test."
	@echo "  flow-compustat-linear         Run the flow-Compustat linear benchmark pipeline."
	@echo "  flow-compustat-linear-smoke   Run the flow-Compustat linear smoke test."
	@echo "  flow-compustat-nonlinear      Run the flow-Compustat nonlinear benchmark pipeline."
	@echo "  flow-compustat-nonlinear-smoke Run the flow-Compustat nonlinear smoke test."
	@echo "  walkforward                   Run the final audited expanding-window pipeline."
	@echo "  walkforward-smoke             Run the walk-forward smoke test."
	@echo "  walkforward-aggregate         Aggregate existing walk-forward shards."
	@echo "  classical-baselines           Run the classical baseline comparison."
	@echo "  classical-baselines-smoke     Run the classical baseline smoke test."
	@echo "  walkforward-robustness        Build walk-forward robustness summaries."
	@echo "  walkforward-robustness-smoke  Run the walk-forward robustness smoke test."
	@echo "  final-report-outputs          Regenerate curated report tables and figures."
	@echo "  report-pdf                    Compile report/report.tex."
	@echo "  checks                        Run pytest, ruff, and black --check."
	@echo "  test                          Run pytest."
	@echo "  lint                          Run ruff and black --check."
	@echo "  clean                         Remove generated local artefacts."

setup:
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(PIP) install -e ".[dev]"

crsp-parquet:
	$(PYTHON) -m mlfinance.run.convert_crsp_to_parquet data/raw/monthly_crsp.csv

build-panel:
	$(PYTHON) -m mlfinance.run.baseline_pipeline

refresh-baseline-outputs:
	$(PYTHON) -m mlfinance.run.baseline_pipeline runtime.rebuild_panel=false

baseline-smoke:
	$(PYTHON) -m pytest tests/integration/test_baseline_pipeline.py -q

linear-benchmarks:
	$(PYTHON) -m mlfinance.run.linear_benchmarks

linear-benchmarks-smoke:
	$(PYTHON) -m pytest tests/run/test_linear_benchmarks.py tests/integration/test_linear_benchmarks.py -q

nonlinear-benchmarks:
	$(PYTHON) -m mlfinance.run.nonlinear_benchmarks

nonlinear-benchmarks-smoke:
	PYTHON="$(PYTHON)" bash scripts/run_nonlinear_smoke.sh

audit-costs-universe:
	$(PYTHON) -m mlfinance.run.audit_costs_universe

audit-costs-universe-smoke:
	$(PYTHON) -m pytest tests/data/test_universe.py tests/integration/test_audit_costs_universe.py -q

compustat-schema-audit:
	$(PYTHON) -m mlfinance.run.compustat_schema_audit

compustat-schema-audit-smoke:
	$(PYTHON) -m pytest tests/integration/test_compustat_schema_audit.py -q

compustat-diagnostics:
	$(PYTHON) -m mlfinance.run.compustat_integration_smoke

compustat-diagnostics-smoke:
	$(PYTHON) -m pytest tests/integration/test_compustat_integration_smoke.py -q

flow-compustat-linear:
	$(PYTHON) -m mlfinance.run.flow_compustat_linear

flow-compustat-linear-smoke:
	$(PYTHON) -m pytest tests/integration/test_flow_compustat_linear.py -q

flow-compustat-nonlinear:
	$(PYTHON) -m mlfinance.run.flow_compustat_nonlinear

flow-compustat-nonlinear-smoke:
	PYTHON="$(PYTHON)" bash scripts/run_flow_compustat_nonlinear_smoke.sh

walkforward:
	$(PYTHON) -m mlfinance.run.walkforward

walkforward-smoke:
	$(PYTHON) -m pytest tests/integration/test_walkforward_smoke.py -q

walkforward-aggregate:
	$(PYTHON) -m mlfinance.run.walkforward walkforward.mode=aggregate

classical-baselines:
	$(PYTHON) -m mlfinance.run.classical_baselines

classical-baselines-smoke:
	$(PYTHON) -m pytest tests/run/test_classical_baselines.py tests/integration/test_classical_baselines_smoke.py -q

walkforward-robustness:
	$(PYTHON) -m mlfinance.run.walkforward_robustness

walkforward-robustness-smoke:
	$(PYTHON) -m pytest tests/eval/test_walkforward_robustness.py tests/integration/test_walkforward_robustness_smoke.py -q

final-report-outputs:
	$(PYTHON) scripts/make_final_report_outputs.py

report-pdf:
	cd report && latexmk -pdf -interaction=nonstopmode report.tex

checks:
	$(PYTHON) -m pytest -q
	$(PYTHON) -m ruff check mlfinance tests scripts
	$(PYTHON) -m black --check mlfinance tests scripts

test:
	$(PYTHON) -m pytest

lint:
	$(PYTHON) -m ruff check mlfinance tests scripts
	$(PYTHON) -m black --check mlfinance tests scripts

clean:
	rm -rf data/processed/*.parquet
	rm -rf data/outputs
	rm -rf outputs
	rm -f report/*.aux report/*.log report/*.out report/*.toc report/*.fls report/*.fdb_latexmk report/*.synctex.gz
