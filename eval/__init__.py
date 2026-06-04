"""Active evaluation exports for the current return-prediction workflow."""

from mlfinance.eval.diebold_mariano import diebold_mariano
from mlfinance.eval.fama_french import (
    alpha_regression,
    bootstrap_alpha_ci,
    holm_bonferroni,
)
from mlfinance.eval.metrics import feature_target_corr, fold_aggregated_r2, oos_r_squared
from mlfinance.eval.newey_west import (
    all_three_lags,
    andrews_1991_plugin_lag,
    newey_west_lag_nw1994,
    newey_west_lag_sqrt_t,
    newey_west_se,
    report_t_stats_3lags,
)
from mlfinance.eval.portfolio_metrics import (
    max_drawdown,
    summarize_backtest,
    summarize_monthly_returns,
)
from mlfinance.eval.predictive_metrics import (
    decile_spread,
    information_coefficient,
    mae,
    rmse,
    summarize_predictions,
)
from mlfinance.eval.statistical_tests import (
    market_alpha_beta,
    newey_west_mean_tstat,
)

__all__ = [
    "alpha_regression",
    "all_three_lags",
    "andrews_1991_plugin_lag",
    "bootstrap_alpha_ci",
    "decile_spread",
    "diebold_mariano",
    "feature_target_corr",
    "fold_aggregated_r2",
    "holm_bonferroni",
    "information_coefficient",
    "mae",
    "market_alpha_beta",
    "max_drawdown",
    "newey_west_lag_nw1994",
    "newey_west_lag_sqrt_t",
    "newey_west_mean_tstat",
    "newey_west_se",
    "oos_r_squared",
    "report_t_stats_3lags",
    "rmse",
    "summarize_backtest",
    "summarize_monthly_returns",
    "summarize_predictions",
]
