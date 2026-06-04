"""Top-level orchestrator for the (scheme x cost_model) backtest grid."""

from __future__ import annotations

import pandas as pd

from mlfinance.backtest.costs import BaseCostModel
from mlfinance.backtest.metrics import summary_table, turnover
from mlfinance.backtest.portfolio import (
    compute_realized_returns,
    decile_sort,
    long_short_weights,
)


def _cost_drag_per_period(cost_df: pd.DataFrame, gross_notional: pd.Series) -> pd.Series:
    """Per-date cost drag: ``sum_i total_cost / gross_notional`` (0 where gross==0)."""
    if cost_df.empty:
        return pd.Series(0.0, index=gross_notional.index)
    per_date_cost = cost_df.groupby("date")["total_cost"].sum()
    drag = per_date_cost.reindex(gross_notional.index).fillna(0.0)
    denom = gross_notional.replace(0.0, pd.NA)
    drag = (drag / denom).fillna(0.0).astype(float)
    return drag


def run_full_backtest(
    predictions: pd.DataFrame,
    returns_panel: pd.DataFrame,
    mktcap_panel: pd.DataFrame,
    prices_panel: pd.DataFrame,
    cost_models: dict[str, BaseCostModel],
    schemes: list[str] | None = None,
    n_deciles: int = 10,
    top_decile: int | None = None,
    bottom_decile: int = 1,
) -> pd.DataFrame:
    """Run the full backtest grid; one summary row per (scheme, cost_model).

    VW schemes propagate the ``NotImplementedError`` from
    :func:`long_short_weights`; only EW is supported with the provided CRSP file.
    """
    _ = mktcap_panel  # accepted for API parity / future use
    if schemes is None:
        schemes = ["EW"]
    top = top_decile if top_decile is not None else n_deciles

    if isinstance(returns_panel.index, pd.DatetimeIndex):
        calendar_month = pd.Series(returns_panel.index.to_period("M"), index=returns_panel.index)
    else:
        # Generic fallback: each row is its own month.
        calendar_month = pd.Series(range(len(returns_panel.index)), index=returns_panel.index)

    rows: list[pd.DataFrame] = []
    decile_panel = decile_sort(predictions, n_deciles=n_deciles)

    for scheme in schemes:
        weights = long_short_weights(
            decile_panel, scheme=scheme, top_decile=top, bottom_decile=bottom_decile
        )
        if weights.empty:
            continue
        weights = weights.reindex(returns_panel.index).fillna(0.0)
        gross_returns = compute_realized_returns(weights, returns_panel)
        turn = turnover(weights)
        gross_notional = weights.abs().sum(axis=1)

        net_dict: dict[str, pd.Series] = {}
        for model_name, model in cost_models.items():
            cost_df = model.simulate(
                weights=weights,
                prices=prices_panel.reindex_like(weights),
                calendar_month=calendar_month,
            )
            drag = _cost_drag_per_period(cost_df, gross_notional)
            # gross_returns and drag are BOTH timestamped at the rebalance date t
            # under the new convention (weights[t] * R_{t+1} stored at row t,
            # cost paid at the same rebalance event). No drag.shift(1) needed;
            # shifting again would re-introduce the off-by-one we fixed upstream.
            net_dict[model_name] = gross_returns - drag.fillna(0.0)

        summary = summary_table(
            gross_returns=gross_returns,
            net_returns_dict=net_dict,
            turnover_series=turn,
            weights=weights,
        )
        summary.insert(0, "scheme", scheme)
        rows.append(summary)

    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)
