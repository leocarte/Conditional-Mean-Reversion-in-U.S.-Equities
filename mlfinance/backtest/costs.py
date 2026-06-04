"""IBKR Pro Tiered transaction cost simulator with NYSE/FINRA/SEC pass-throughs."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Final

import numpy as np
import pandas as pd

#: IBKR Pro Tiered schedule: ``(upper bound on cumulative monthly shares, $/share)``.
#: Top tier is open-ended via ``math.inf``.
IBKR_TIERS: Final[list[tuple[float, float]]] = [
    (300_000.0, 0.0035),
    (3_000_000.0, 0.0020),
    (20_000_000.0, 0.0015),
    (100_000_000.0, 0.0010),
    (float("inf"), 0.0005),
]


class BaseCostModel(ABC):
    """Protocol every cost simulator implements; orchestrator only uses ``simulate``."""

    name: str = "base"

    @abstractmethod
    def simulate(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        calendar_month: pd.Series,
    ) -> pd.DataFrame:
        """Compute per-(date, permno) trade costs for an entire backtest path."""


def _tier_cost_for_one_order(
    shares: float,
    monthly_so_far: float,
    tiers: list[tuple[float, float]] = IBKR_TIERS,
) -> float:
    """Scalar reference implementation of the IBKR marginal-tier commission.

    Kept as the slow but obviously-correct oracle the vectorised path is
    tested against.
    """
    shares = abs(shares)
    remaining = shares
    gross = 0.0
    cumulative = monthly_so_far
    prev_upper = 0.0
    for upper, rate in tiers:
        tier_capacity = upper - max(prev_upper, cumulative)
        if tier_capacity <= 0:
            prev_upper = upper
            continue
        shares_in_tier = min(remaining, tier_capacity)
        gross += shares_in_tier * rate
        remaining -= shares_in_tier
        cumulative += shares_in_tier
        prev_upper = upper
        if remaining <= 0:
            break
    return float(gross)


def _vectorized_tier_gross(
    shares: np.ndarray,
    monthly_so_far: np.ndarray,
    tiers: list[tuple[float, float]] = IBKR_TIERS,
) -> np.ndarray:
    """Fully-vectorised gross commission for many orders at once.

    For each order i and tier t the in-tier share count is
    ``max(0, min(U_t, m_i + S_i) - max(U_{t-1}, m_i))``; gross is
    ``sum_t s_{i,t} * r_t``. With a fixed (small) tier count we broadcast
    over orders and reduce along the tier axis.
    """
    shares = np.asarray(shares, dtype=np.float64)
    monthly_so_far = np.asarray(monthly_so_far, dtype=np.float64)
    uppers = np.array([t[0] for t in tiers], dtype=np.float64)
    rates = np.array([t[1] for t in tiers], dtype=np.float64)

    lowers = np.concatenate(([0.0], uppers[:-1]))

    m = monthly_so_far[:, None]
    s = shares[:, None]
    eff_lower = np.maximum(lowers[None, :], m)
    eff_upper = np.minimum(uppers[None, :], m + s)
    shares_in_tier = np.maximum(0.0, eff_upper - eff_lower)
    return (shares_in_tier * rates[None, :]).sum(axis=1)


class IBKRTieredCostModel(BaseCostModel):
    """IBKR Pro Tiered commission with $0.35 floor, 1% cap, and venue pass-throughs."""

    name = "IBKRTiered"

    def __init__(
        self,
        min_per_order: float = 0.35,
        max_pct: float = 0.01,
        clearing_per_share: float = 0.0002,
        # IBKR Pro Tiered: average exchange / ECN liquidity-removing fee
        # is ~$0.0003/share (varies by venue and side; can be a rebate on
        # adds). The legacy default of $0.003/share was 10x too high; it
        # inflated round-trip cost by ~6 bps on a typical 40k-share
        # rebalance, biasing every reported net-of-cost Sharpe and alpha
        # downward. See IBKR's tiered fee schedule.
        exchange_per_share: float = 0.0003,
        sec_fee_rate: float = 0.0000278,
        finra_taf_per_share: float = 0.0000278,
        passthru: float = 0.000175,
        tiers: list[tuple[float, float]] | None = None,
    ) -> None:
        self.min_per_order = float(min_per_order)
        self.max_pct = float(max_pct)
        self.clearing_per_share = float(clearing_per_share)
        self.exchange_per_share = float(exchange_per_share)
        self.sec_fee_rate = float(sec_fee_rate)
        self.finra_taf_per_share = float(finra_taf_per_share)
        self.passthru = float(passthru)
        self.tiers: list[tuple[float, float]] = list(tiers) if tiers is not None else IBKR_TIERS

    def cost_per_order(
        self,
        shares: np.ndarray,
        price: np.ndarray,
        monthly_shares_so_far: np.ndarray,
        side: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute per-order total cost (commission + venue + regulatory passthrough).

        Returns (total_cost, gross_commission, updated_monthly_shares).
        """
        shares = np.asarray(shares, dtype=np.float64)
        price = np.asarray(price, dtype=np.float64)
        monthly = np.asarray(monthly_shares_so_far, dtype=np.float64)
        side_arr = np.asarray(side)

        gross = _vectorized_tier_gross(shares, monthly, self.tiers)
        trade_value = shares * price

        # Floor then cap (notebook applies in this order).
        commission = np.maximum(self.min_per_order, gross)
        commission = np.minimum(commission, self.max_pct * trade_value)

        clearing = self.clearing_per_share * shares
        exchange = self.exchange_per_share * shares
        venue_fees = clearing + exchange

        is_sell = side_arr == "sell"
        sec_fee = np.where(is_sell, self.sec_fee_rate * trade_value, 0.0)
        finra_taf = np.where(is_sell, self.finra_taf_per_share * shares, 0.0)

        # IBKR's regulatory passthru (the SEC/FINRA filing-fee component
        # that IBKR passes through to the client) is 17.5 bp of the
        # COMMISSION ONLY, not commission + venue. Legacy code mistakenly
        # applied passthru to commission+venue_fees, slightly over-charging
        # by 0.000175 * venue_fees per order. The error is small in
        # absolute terms but propagates into every reported cost number.
        ibkr_passthru = self.passthru * commission

        total = commission + venue_fees + sec_fee + finra_taf + ibkr_passthru
        total = np.where(shares > 0, total, 0.0)

        updated_monthly = monthly + shares
        return total, gross, updated_monthly

    def simulate(
        self,
        weights: pd.DataFrame,
        prices: pd.DataFrame,
        calendar_month: pd.Series,
    ) -> pd.DataFrame:
        """Simulate IBKR costs over a full (date, permno) weight path.

        Trades are derived from ``weights.diff()`` with the implicit
        pre-period weight of zero, so the first row generates a "buy from
        cash" trade. The monthly tier accumulator resets per calendar month.
        """
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

        # capital == 1 convention; |dw| is the dollar trade value per name.
        abs_dw = dw.abs()
        safe_price = prices.where(prices > 0)
        shares = (abs_dw / safe_price).fillna(0.0)
        trade_value = abs_dw.fillna(0.0)
        side_signed = np.sign(dw.values)

        shares_long = shares.stack(future_stack=True).rename("shares_traded")
        value_long = trade_value.stack(future_stack=True).rename("trade_value")
        sign_long = (
            pd.DataFrame(
                side_signed,
                index=weights.index,
                columns=weights.columns,
            )
            .stack(future_stack=True)
            .rename("sign")
        )
        price_long = prices.stack(future_stack=True).rename("price")

        df = pd.concat([shares_long, value_long, sign_long, price_long], axis=1).dropna(
            subset=["price"]
        )

        df = df[df["shares_traded"] > 0].copy()
        df = df.reset_index().rename(columns={"level_0": "date", "level_1": "permno"})
        if df.columns[0] != "date":
            df = df.rename(columns={df.columns[0]: "date", df.columns[1]: "permno"})

        # Per-month cumsum-shift for the tier accumulator.
        df["calendar_month"] = df["date"].map(calendar_month)
        df = df.sort_values(["calendar_month", "date", "permno"]).reset_index(drop=True)
        cumsum = df.groupby("calendar_month", sort=False)["shares_traded"].cumsum()
        df["monthly_so_far"] = (cumsum - df["shares_traded"]).to_numpy()

        side = np.where(df["sign"].to_numpy() > 0, "buy", "sell")
        total, gross, _ = self.cost_per_order(
            df["shares_traded"].to_numpy(),
            df["price"].to_numpy(),
            df["monthly_so_far"].to_numpy(),
            side,
        )

        out = pd.DataFrame(
            {
                "date": df["date"].to_numpy(),
                "permno": df["permno"].to_numpy(),
                "shares_traded": df["shares_traded"].to_numpy(),
                "trade_value": df["trade_value"].to_numpy(),
                "gross_commission": gross,
                "total_cost": total,
                "side": side,
            }
        )
        return out
