#!/usr/bin/env bash
# Nonlinear benchmark smoke runner that avoids macOS conda OpenMP crashes.
#
# torch and xgboost each bundle an OpenMP runtime; on macOS conda, using both in
# one Python process crashes (segfault at xgboost/data.py _meta_from_numpy) when
# xgboost builds a DMatrix from numpy. Importing alone does not crash, so a
# subprocess probe both imports the stack AND builds a small DMatrix:
#   - probe ok    -> healthy stack (Linux / cluster / CI): run the strict suite.
#   - probe fails -> skip cleanly (exit 0) with a clear message.
# Force a skip regardless with SKIP_NONLINEAR_BENCHMARK_SMOKE=1.
#
# Run from the repo root (the Makefile target does).
set -uo pipefail

PYTHON="${PYTHON:-python}"
PROBE="scripts/_nonlinear_smoke_probe.py"
TARGETS=(tests/run/test_nonlinear_benchmarks.py tests/integration/test_nonlinear_benchmarks.py)

_skip() {
  echo "[nonlinear-smoke] SKIPPED: $1"
  echo "[nonlinear-smoke] Run the strict suite on Linux or CI if the local stack is incompatible."
  exit 0
}

# (1) Explicit opt-out.
if [ -n "${SKIP_NONLINEAR_BENCHMARK_SMOKE:-}" ]; then
  _skip "SKIP_NONLINEAR_BENCHMARK_SMOKE is set."
fi

# (2) Probe in an isolated child shell so a segfault cannot kill this script, and
#     its "Segmentation fault" diagnostic is swallowed by the redirect.
if ! bash -c '"$0" "$1"' "${PYTHON}" "${PROBE}" >/dev/null 2>&1; then
  _skip "torch + xgboost cannot share this process (likely macOS conda duplicate OpenMP)."
fi

# (3) Healthy stack -> strict run. This is the cluster / CI path.
echo "[nonlinear-smoke] torch + xgboost OK in one process; running strict suite."
exec "${PYTHON}" -m pytest "${TARGETS[@]}" -q
