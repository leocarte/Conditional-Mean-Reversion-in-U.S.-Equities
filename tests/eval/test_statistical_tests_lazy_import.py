"""statsmodels must stay a lazy import so statsmodels-free stages stay importable.

scipy 1.15 removed ``scipy._lib._util._lazywhere``, which statsmodels < 0.14.5
imports at load time -- so an image that pairs new scipy with old statsmodels makes
``import statsmodels`` raise. We now pin statsmodels >= 0.14.5, but keeping the
import lazy means stages that never compute HAC statistics (e.g. the feature-audit
flow-Compustat smoke) still import and run even where statsmodels is broken or
absent (such as the current pre-rebuild cluster image). ``mlfinance.eval`` must
therefore not import statsmodels at module load. These tests pin that contract
while confirming the HAC helpers still work when statsmodels is present.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

from mlfinance.eval.statistical_tests import market_alpha_beta, newey_west_mean_tstat

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_eval_and_smoke_runner_import_without_statsmodels() -> None:
    code = textwrap.dedent(
        """
        import sys
        # Simulate the cluster image where 'import statsmodels' fails.
        sys.modules["statsmodels"] = None
        sys.modules["statsmodels.api"] = None
        import mlfinance.eval  # package __init__ must not eager-import statsmodels
        import mlfinance.run.compustat_integration_smoke  # full transitive chain
        assert sys.modules["statsmodels"] is None
        print("import-ok")
        """
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        cwd=REPO_ROOT,
    )
    assert result.returncode == 0, result.stderr
    assert "import-ok" in result.stdout


def test_hac_helpers_compute_when_statsmodels_present() -> None:
    rng = np.random.default_rng(0)
    rp = pd.Series(0.01 + 0.02 * rng.standard_normal(60))
    rm = pd.Series(0.008 + 0.015 * rng.standard_normal(60))

    mean_stats = newey_west_mean_tstat(rp, lags=6)
    assert mean_stats["n"] == 60
    assert np.isfinite(mean_stats["tstat"])

    ab = market_alpha_beta(rp, rm, lags=6)
    assert ab["n"] == 60
    assert np.isfinite(ab["alpha"]) and np.isfinite(ab["beta"])
