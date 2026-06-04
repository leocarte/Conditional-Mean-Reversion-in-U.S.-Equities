"""Transaction-cost simulators, portfolio construction, and backtest grid."""

from __future__ import annotations

from mlfinance.backtest.costs import (
    IBKR_TIERS,
    BaseCostModel,
    IBKRTieredCostModel,
)
from mlfinance.backtest.metrics import (
    cumulative_return,
    max_drawdown,
    newey_west_tstat,
    sharpe,
    summary_table,
    turnover,
)
from mlfinance.backtest.portfolio import (
    compute_realized_returns,
    decile_sort,
    long_short_weights,
)
from mlfinance.backtest.robust_costs import FlatPlusImpactCostModel
from mlfinance.backtest.run_backtest import run_full_backtest

__all__ = [
    "BaseCostModel",
    "FlatPlusImpactCostModel",
    "IBKRTieredCostModel",
    "IBKR_TIERS",
    "compute_realized_returns",
    "cumulative_return",
    "decile_sort",
    "long_short_weights",
    "max_drawdown",
    "newey_west_tstat",
    "run_full_backtest",
    "sharpe",
    "summary_table",
    "turnover",
]
