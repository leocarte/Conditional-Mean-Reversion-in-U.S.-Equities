"""Half-spread + sqrt-impact cost model (Almgren-Chriss / FIM 2018)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from mlfinance.backtest.costs import BaseCostModel


class FlatPlusImpactCostModel(BaseCostModel):
    """Flat half-spread (bps) plus sqrt market-impact (eta * sigma * sqrt(q/ADV))."""

    name = "FlatPlusImpact"

    def __init__(
        self,
        spread_bps: float = 10.0,
        eta: float = 0.1,
        vol_window: int = 60,
        adv_window: int = 60,
    ) -> None:
        self.spread_bps = float(spread_bps)
        self.eta = float(eta)
        self.vol_window = int(vol_window)
        self.adv_window = int(adv_window)

    def cost_per_order(
        self,
        abs_dw: float,
        sigma: float,
        adv_dollars: float,
        trade_value: float,
    ) -> float:
        """Single-trade dollar cost: half-spread + sqrt-impact."""
        half_spread = (self.spread_bps / 10_000.0) * trade_value
        if adv_dollars > 0:
            impact = self.eta * sigma * np.sqrt(trade_value / adv_dollars) * trade_value
        else:
            impact = 0.0
        return float(half_spread + impact)

    def simulate(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        calendar_month: pd.Series,
        *,
        returns: pd.DataFrame | None = None,
        dollar_volume: pd.DataFrame | None = None,
    ) -> pd.DataFrame:
        """Per-(date, permno) costs under the spread + impact model.

        ``gross_commission`` is NaN; this model has no headline commission.
        Returns is used for the trailing-volatility estimate; falls back to
        log-price changes if absent.
        """
        _ = calendar_month  # accepted for API parity; path-independent across months.

        if weights.empty:
            return pd.DataFrame(
                columns=[
                    "date",
                    "permno",
                    "shares_traded",
                    "trade_value",
                    "gross_commission",
                    "total_cost",
                    "side",
                ]
            )

        w = weights.reindex_like(prices).fillna(0.0)
        dw = w.diff()
        dw.iloc[0] = w.iloc[0]
        abs_dw = dw.abs()
        safe_price = prices.where(prices > 0)
        shares = (abs_dw / safe_price).fillna(0.0)
        trade_value = abs_dw.fillna(0.0)
        side_signed = np.sign(dw.values)

        if returns is None:
            returns = np.log(prices.where(prices > 0)).diff()
        sigma = returns.rolling(self.vol_window, min_periods=2).std()
        sigma = sigma.reindex_like(weights).ffill().fillna(0.0)

        if dollar_volume is None:
            # Proxy: produces ratio ~1 everywhere; only meaningful in tests.
            adv = trade_value.replace(0.0, np.nan).ffill().fillna(1.0)
        else:
            adv = dollar_volume.rolling(self.adv_window, min_periods=2).mean()
            adv = adv.reindex_like(weights).ffill().fillna(0.0)

        half_spread = (self.spread_bps / 10_000.0) * trade_value
        # Guard sqrt(0/0): where ADV == 0, impact = 0.
        ratio = np.where(adv.to_numpy() > 0, trade_value.to_numpy() / adv.to_numpy(), 0.0)
        impact_per_dollar = self.eta * sigma.to_numpy() * np.sqrt(np.clip(ratio, 0.0, None))
        impact_dollar = impact_per_dollar * trade_value.to_numpy()
        total_panel = pd.DataFrame(
            half_spread.to_numpy() + impact_dollar,
            index=weights.index,
            columns=weights.columns,
        )

        shares_long = shares.stack(future_stack=True).rename("shares_traded")
        value_long = trade_value.stack(future_stack=True).rename("trade_value")
        sign_long = (
            pd.DataFrame(side_signed, index=weights.index, columns=weights.columns)
            .stack(future_stack=True)
            .rename("sign")
        )
        total_long = total_panel.stack(future_stack=True).rename("total_cost")

        df = pd.concat([shares_long, value_long, sign_long, total_long], axis=1)
        df = df[df["trade_value"] > 0].copy()
        df = df.reset_index().rename(columns={"level_0": "date", "level_1": "permno"})
        if df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "permno"})

        side = np.where(df["sign"].to_numpy() > 0, "buy", "sell")
        out = pd.DataFrame(
            {
                "date": df["date"].to_numpy(),
                "permno": df["permno"].to_numpy(),
                "shares_traded": df["shares_traded"].to_numpy(),
                "trade_value": df["trade_value"].to_numpy(),
                "gross_commission": np.nan,
                "total_cost": df["total_cost"].to_numpy(),
                "side": side,
            }
        )
        return out
