"""Model exports with lazy imports so lightweight modules stay importable."""

from __future__ import annotations

from importlib import import_module

from mlfinance.models.base import BaseModel, r_squared_oos

_LAZY_IMPORTS: dict[str, tuple[str, str]] = {
    "ConstantForecast": ("mlfinance.models.baselines", "ConstantForecast"),
    "baseline_predictions_from_panel": (
        "mlfinance.models.baselines",
        "baseline_predictions_from_panel",
    ),
    "RidgeEig": ("mlfinance.models.ridge", "RidgeEig"),
    "ridge_eig": ("mlfinance.models.ridge", "ridge_eig"),
    "RidgelessModel": ("mlfinance.models.ridge", "RidgelessModel"),
    "ridgeless_min_norm": ("mlfinance.models.ridge", "ridgeless_min_norm"),
    "ElasticNetModel": ("mlfinance.models.elastic_net", "ElasticNetModel"),
    "LassoModel": ("mlfinance.models.lasso", "LassoModel"),
    "LinearFeaturePreprocessor": (
        "mlfinance.models.linear_preprocessing",
        "LinearFeaturePreprocessor",
    ),
    "write_feature_list": ("mlfinance.models.linear_preprocessing", "write_feature_list"),
    "RandomFeaturesModel": ("mlfinance.models.random_features", "RandomFeaturesModel"),
    "compute_random_features": ("mlfinance.models.random_features", "compute_random_features"),
    "generate_random_weights": ("mlfinance.models.random_features", "generate_random_weights"),
    "MLPRegressor": ("mlfinance.models.mlp_tabular", "MLPRegressor"),
    "XGBoostModel": ("mlfinance.models.xgb", "XGBoostModel"),
    "LightGBMModel": ("mlfinance.models.xgb", "LightGBMModel"),
}

__all__ = [
    "BaseModel",
    "r_squared_oos",
    *sorted(_LAZY_IMPORTS.keys()),
]


def __getattr__(name: str):
    if name in {"BaseModel", "r_squared_oos"}:
        return globals()[name]
    if name not in _LAZY_IMPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    module_path, attr_name = _LAZY_IMPORTS[name]
    value = getattr(import_module(module_path), attr_name)
    globals()[name] = value
    return value
