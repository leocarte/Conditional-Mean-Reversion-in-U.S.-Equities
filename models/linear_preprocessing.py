"""Training-window preprocessing for linear benchmark linear models."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class LinearFeaturePreprocessor:
    """Winsorize, median-impute, standardize, and append missing flags.

    The transformer is fit on one estimation sample only, then reused on
    validation or test rows without refitting.
    """

    def __init__(
        self,
        feature_cols: list[str],
        *,
        date_col: str = "date",
        winsor_lower: float = 0.01,
        winsor_upper: float = 0.99,
    ) -> None:
        if not feature_cols:
            raise ValueError("feature_cols must be non-empty.")
        if not 0.0 <= winsor_lower < winsor_upper <= 1.0:
            raise ValueError(
                f"winsor bounds must satisfy 0 <= lower < upper <= 1, got "
                f"{winsor_lower}, {winsor_upper}."
            )
        self.feature_cols = list(feature_cols)
        self.date_col = str(date_col)
        self.winsor_lower = float(winsor_lower)
        self.winsor_upper = float(winsor_upper)

        self.lower_bounds_: pd.Series | None = None
        self.upper_bounds_: pd.Series | None = None
        self.medians_: pd.Series | None = None
        self.means_: pd.Series | None = None
        self.stds_: pd.Series | None = None
        self.matrix_columns_: list[str] | None = None
        self.fit_metadata_: dict[str, Any] | None = None

    def _validate_input(self, df: pd.DataFrame) -> None:
        missing = [col for col in [self.date_col, *self.feature_cols] if col not in df.columns]
        if missing:
            raise KeyError(f"Missing required columns for preprocessing: {missing}")

    def fit(self, df: pd.DataFrame) -> LinearFeaturePreprocessor:
        self._validate_input(df)
        x = df[self.feature_cols].astype("float64")

        lower = x.quantile(self.winsor_lower)
        upper = x.quantile(self.winsor_upper)
        clipped = x.clip(lower=lower, upper=upper, axis=1)

        medians = clipped.median().fillna(0.0)
        imputed = clipped.fillna(medians)

        means = imputed.mean()
        stds = imputed.std(ddof=0).replace(0.0, 1.0).fillna(1.0)

        self.lower_bounds_ = lower.astype("float64")
        self.upper_bounds_ = upper.astype("float64")
        self.medians_ = medians.astype("float64")
        self.means_ = means.astype("float64")
        self.stds_ = stds.astype("float64")
        self.matrix_columns_ = [
            *self.feature_cols,
            *[f"{col}_missing" for col in self.feature_cols],
        ]

        dates = pd.to_datetime(df[self.date_col])
        self.fit_metadata_ = {
            "n_rows": int(len(df)),
            "fit_start": pd.Timestamp(dates.min()).date().isoformat(),
            "fit_end": pd.Timestamp(dates.max()).date().isoformat(),
            "winsor_lower": self.winsor_lower,
            "winsor_upper": self.winsor_upper,
        }
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        self._validate_input(df)
        if any(
            attr is None
            for attr in (
                self.lower_bounds_,
                self.upper_bounds_,
                self.medians_,
                self.means_,
                self.stds_,
                self.matrix_columns_,
            )
        ):
            raise RuntimeError("LinearFeaturePreprocessor.transform called before fit.")

        x = df[self.feature_cols].astype("float64")
        missing_flags = x.isna().astype("float64")
        missing_flags.columns = [f"{col}_missing" for col in self.feature_cols]

        clipped = x.clip(lower=self.lower_bounds_, upper=self.upper_bounds_, axis=1)
        imputed = clipped.fillna(self.medians_)
        scaled = (imputed - self.means_) / self.stds_

        out = pd.concat([scaled, missing_flags], axis=1)
        return out[self.matrix_columns_].astype("float64")

    def metadata(self) -> dict[str, Any]:
        if self.fit_metadata_ is None:
            raise RuntimeError("LinearFeaturePreprocessor.metadata called before fit.")
        return dict(self.fit_metadata_)

    def to_dict(self) -> dict[str, Any]:
        meta = self.metadata()
        return {
            **meta,
            "feature_cols": list(self.feature_cols),
            "matrix_columns": list(self.matrix_columns_ or []),
        }


def write_feature_list(path: str | Path, feature_cols: list[str]) -> Path:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    pd.Series(feature_cols, name="feature").to_json(p, orient="values", indent=2)
    return p
