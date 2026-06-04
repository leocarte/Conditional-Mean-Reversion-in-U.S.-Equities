#!/usr/bin/env bash
# feature-audit flow-Compustat NONLINEAR smoke runner.
#
# Mirrors scripts/run_nonlinear_smoke.sh: probes whether torch + xgboost can
# safely share one Python process (they cannot on macOS conda due to duplicate
# OpenMP runtimes) and runs the strict pytest suite only when the probe passes.
# On platforms where the probe fails (or when SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE=1 is
# set) we skip cleanly with exit 0. Reuses the existing probe at
# ``scripts/_nonlinear_smoke_probe.py`` because the underlying issue is identical
# (torch + xgboost imports + DMatrix build in the same process).
#
# Run from the repo root (the Makefile target does).
set -uo pipefail

PYTHON="${PYTHON:-python}"
PROBE="scripts/_nonlinear_smoke_probe.py"
TARGETS=(tests/integration/test_flow_compustat_nonlinear.py)

_skip() {
  echo "[flow-compustat-nonlinear-smoke] SKIPPED: $1"
  echo "[flow-compustat-nonlinear-smoke] feature-audit nonlinear runs target the Linux or CI environment (GPU=1)."
  exit 0
}

# (1) Explicit opt-out -- mirror the nonlinear-benchmark smoke wrapper's env var, AND
#     accept SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE for a feature-audit-specific opt-out.
if [ -n "${SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE:-}" ]; then
  _skip "SKIP_FLOW_COMPUSTAT_NONLINEAR_SMOKE is set."
fi
if [ -n "${SKIP_NONLINEAR_BENCHMARK_SMOKE:-}" ]; then
  _skip "SKIP_NONLINEAR_BENCHMARK_SMOKE is set (reusing nonlinear benchmark opt-out for feature-audit)."
fi

# (2) Probe in an isolated child shell so a segfault cannot kill this script.
if ! bash -c '"$0" "$1"' "${PYTHON}" "${PROBE}" >/dev/null 2>&1; then
  _skip "torch + xgboost cannot share this process (likely macOS conda duplicate OpenMP)."
fi

# (3) Healthy stack -> strict run.
echo "[flow-compustat-nonlinear-smoke] torch + xgboost OK in one process; running strict suite."
exec "${PYTHON}" -m pytest "${TARGETS[@]}" -q
