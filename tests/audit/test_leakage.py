"""Synthetic-data tests for the temporal-leakage audit suite."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mlfinance.audit.cv_audit import (
    audit_cv_disjointness,
    audit_no_inner_shuffle,
    audit_walk_forward_expansion,
)
from mlfinance.audit.leakage import (
    LeakageAuditError,
    audit_feature_provenance,
    audit_prediction_dates,
    audit_preprocessor_fit_train_only,
    audit_target_alignment,
    fit_with_audit,
)
from mlfinance.audit.negative_control import run_negative_control

N_STOCKS = 30
N_MONTHS = 24


def _clean_panel(seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    permnos = list(range(10001, 10001 + N_STOCKS))
    months = pd.date_range("2010-01-31", periods=N_MONTHS, freq="ME")
    records = []
    for permno in permnos:
        for date in months:
            row = {
                "permno": permno,
                "date": date,
                "yyyymm": int(date.strftime("%Y%m")),
                "mktcap": float(rng.lognormal(8.0, 1.5)),
                "ret_exc": float(rng.normal(0.005, 0.05)),
            }
            for i in range(5):
                row[f"jkp_{i:02d}"] = float(rng.standard_normal())
            records.append(row)
    df = pd.DataFrame.from_records(records)
    df = df.sort_values(["permno", "date"]).reset_index(drop=True)
    df["ret_exc_next"] = df.groupby("permno", observed=True)["ret_exc"].shift(-1)
    df = df.dropna(subset=["ret_exc_next"]).reset_index(drop=True)
    return df


def test_feature_provenance_passes_on_clean_panel() -> None:
    """A clean panel with lag-appropriate features produces no violations."""
    panel = _clean_panel()
    feature_cols = [c for c in panel.columns if c.startswith("jkp_")]
    violations = audit_feature_provenance(panel, feature_cols, as_of_lag_days={"jkp_*": 30})
    assert violations.empty, f"Unexpected violations on a clean panel: {violations}"


def test_feature_provenance_catches_future_compustat() -> None:
    """A Compustat row whose availability_date is AFTER the signal_date must be flagged."""
    panel = _clean_panel()
    panel["compustat_saleq"] = np.arange(len(panel), dtype=float)
    panel["compustat_saleq_availability_date"] = pd.to_datetime(panel["date"])
    # Push availability_date 1 day past signal_date for the first 100 rows: textbook leak.
    mask = np.zeros(len(panel), dtype=bool)
    mask[:100] = True
    panel.loc[mask, "compustat_saleq_availability_date"] = panel.loc[mask, "date"] + pd.Timedelta(
        days=1
    )

    violations = audit_feature_provenance(panel, ["compustat_saleq"])
    assert not violations.empty, "Future Compustat row was not detected."
    assert int((violations["column"] == "compustat_saleq").sum()) == 100

    with pytest.raises(LeakageAuditError):
        audit_feature_provenance(panel, ["compustat_saleq"], raise_on_fail=True)


def test_target_alignment_passes_on_clean_panel() -> None:
    """ret_exc_next built via shift(-1) within permno must pass the audit."""
    panel = _clean_panel()
    violations = audit_target_alignment(panel, target_col="ret_exc_next")
    assert violations.empty, f"Clean panel flagged: {violations.head()}"


def test_target_alignment_catches_off_by_one() -> None:
    """target == ret_exc (current) instead of next must flag every off-by-one row."""
    panel = _clean_panel().copy()
    panel["target_leak"] = panel["ret_exc"]
    violations = audit_target_alignment(panel, target_col="target_leak")
    assert not violations.empty, "Contemporaneous-target leak was not detected."

    with pytest.raises(LeakageAuditError):
        audit_target_alignment(panel, target_col="target_leak", raise_on_fail=True)


def test_prediction_dates_passes_on_clean_set() -> None:
    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]),
            "return_date": pd.to_datetime(["2020-02-29", "2020-03-31", "2020-04-30"]),
            "y_pred": [0.1, -0.1, 0.05],
        }
    )
    violations = audit_prediction_dates(preds)
    assert violations.empty


def test_prediction_dates_catches_signal_after_return() -> None:
    """Predictions where signal_date >= return_date must be flagged."""
    preds = pd.DataFrame(
        {
            "date": pd.to_datetime(["2020-01-31", "2020-03-31", "2020-04-30"]),
            "return_date": pd.to_datetime(["2020-02-29", "2020-02-29", "2020-04-30"]),
            "y_pred": [0.1, -0.1, 0.05],
        }
    )
    violations = audit_prediction_dates(preds)
    # Row 1 (signal > return) and Row 2 (equality) violate.
    assert len(violations) == 2
    assert set(violations["row_index"].tolist()) == {1, 2}

    with pytest.raises(LeakageAuditError):
        audit_prediction_dates(preds, raise_on_fail=True)


def test_preprocessor_fit_with_audit_passes() -> None:
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler

    n = 100
    X = np.random.default_rng(0).standard_normal((n, 4))
    panel = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:60] = True

    scaler = StandardScaler()
    fit_with_audit(scaler, X, train_mask)
    assert audit_preprocessor_fit_train_only(scaler, train_mask, panel) is True


def test_preprocessor_fit_full_panel_is_detected() -> None:
    """Stamping the wrong count (fit on full panel) must trigger the audit."""
    pytest.importorskip("sklearn")
    from sklearn.preprocessing import StandardScaler

    n = 100
    X = np.random.default_rng(0).standard_normal((n, 4))
    panel = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    train_mask = np.zeros(n, dtype=bool)
    train_mask[:60] = True

    scaler = StandardScaler().fit(X)  # fit on full panel - the bug
    with pytest.raises(LeakageAuditError):
        audit_preprocessor_fit_train_only(scaler, train_mask, panel)


def _make_folds(n: int, splits: list[tuple[int, int, int]]) -> list[dict]:
    folds: list[dict] = []
    for k, (n_tr, n_va, n_te) in enumerate(splits, start=1):
        tr = np.zeros(n, dtype=bool)
        va = np.zeros(n, dtype=bool)
        te = np.zeros(n, dtype=bool)
        tr[:n_tr] = True
        va[n_tr : n_tr + n_va] = True
        te[n_tr + n_va : n_tr + n_va + n_te] = True
        folds.append(
            {
                "fold_index": k,
                "train_mask": tr,
                "val_mask": va,
                "test_mask": te,
            }
        )
    return folds


def test_cv_disjointness_passes_on_clean_folds() -> None:
    n = 100
    folds = _make_folds(n, [(60, 20, 20)])
    out = audit_cv_disjointness(folds)
    assert out["verdict"] == "PASS"
    assert out["n_violations"] == 0


def test_cv_disjointness_catches_overlap() -> None:
    n = 100
    tr = np.zeros(n, dtype=bool)
    va = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[:70] = True
    va[60:80] = True  # overlaps train [60..70)
    te[80:100] = True
    folds = [
        {
            "fold_index": 1,
            "train_mask": tr,
            "val_mask": va,
            "test_mask": te,
        }
    ]
    out = audit_cv_disjointness(folds)
    assert out["verdict"] == "FAIL"
    assert out["n_violations"] >= 1
    types = {v["type"] for v in out["details"]}
    assert "train_val_overlap" in types

    with pytest.raises(LeakageAuditError):
        audit_cv_disjointness(folds, raise_on_fail=True)


def test_cv_disjointness_catches_date_misordering() -> None:
    """Mask-disjoint folds with train AFTER val must still FAIL."""
    dates = pd.Series(pd.date_range("2020-01-31", periods=6, freq="ME").repeat(2))
    n = len(dates)
    # Train = dates 3..5; val = dates 0..1; test = date 2. Train is AFTER val.
    tr = np.zeros(n, dtype=bool)
    va = np.zeros(n, dtype=bool)
    te = np.zeros(n, dtype=bool)
    tr[6:] = True
    va[0:4] = True
    te[4:6] = True
    folds = [
        {
            "fold_index": 1,
            "train_mask": tr,
            "val_mask": va,
            "test_mask": te,
        }
    ]
    out = audit_cv_disjointness(folds, dates=dates)
    assert out["verdict"] == "FAIL"
    types = {v["type"] for v in out["details"]}
    assert "train_after_val_start" in types


def test_walk_forward_expansion_passes_on_expanding_folds() -> None:
    n = 100
    folds = []
    for k, train_end in enumerate([40, 60, 80], start=1):
        tr = np.zeros(n, dtype=bool)
        tr[:train_end] = True
        va = np.zeros(n, dtype=bool)
        va[train_end : train_end + 10] = True
        te = np.zeros(n, dtype=bool)
        te[train_end + 10 : train_end + 20] = True
        folds.append(
            {
                "fold_index": k,
                "train_mask": tr,
                "val_mask": va,
                "test_mask": te,
            }
        )
    out = audit_walk_forward_expansion(folds)
    assert out["verdict"] == "PASS"
    assert out["n_violations"] == 0


def test_walk_forward_expansion_catches_shrinking_train() -> None:
    """Sliding-window (shrinking train) is a violation."""
    n = 100
    folds = []
    for k, end in enumerate([60, 50], start=1):
        tr = np.zeros(n, dtype=bool)
        tr[:end] = True
        va = np.zeros(n, dtype=bool)
        va[end : end + 10] = True
        te = np.zeros(n, dtype=bool)
        te[end + 10 : end + 20] = True
        folds.append(
            {
                "fold_index": k,
                "train_mask": tr,
                "val_mask": va,
                "test_mask": te,
            }
        )
    out = audit_walk_forward_expansion(folds)
    assert out["verdict"] == "FAIL"
    types = {v["type"] for v in out["details"]}
    assert "train_lost_rows" in types or "train_did_not_grow" in types

    with pytest.raises(LeakageAuditError):
        audit_walk_forward_expansion(folds, raise_on_fail=True)


def test_no_inner_shuffle_passes_on_contiguous_masks() -> None:
    n = 100
    folds = _make_folds(n, [(60, 20, 20)])
    out = audit_no_inner_shuffle(folds)
    assert out["verdict"] == "PASS"


def test_no_inner_shuffle_catches_random_sampling() -> None:
    """Randomly-sampled (non-contiguous) train rows must FAIL."""
    n = 100
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr = np.zeros(n, dtype=bool)
    tr[perm[:60]] = True
    va = np.zeros(n, dtype=bool)
    va[perm[60:80]] = True
    te = np.zeros(n, dtype=bool)
    te[perm[80:]] = True
    folds = [
        {
            "fold_index": 1,
            "train_mask": tr,
            "val_mask": va,
            "test_mask": te,
        }
    ]
    out = audit_no_inner_shuffle(folds)
    assert out["verdict"] == "FAIL"
    assert out["n_violations"] >= 1


def _signal_panel(n_months: int = 36, n_stocks: int = 50, seed: int = 0) -> pd.DataFrame:
    """Panel where the next-period target is a noisy linear function of current features."""
    rng = np.random.default_rng(seed)
    permnos = list(range(20001, 20001 + n_stocks))
    months = pd.date_range("2010-01-31", periods=n_months, freq="ME")
    records = []
    for permno in permnos:
        for date in months:
            row = {
                "permno": permno,
                "date": date,
                "yyyymm": int(date.strftime("%Y%m")),
                "mktcap": float(rng.lognormal(8.0, 1.5)),
            }
            for i in range(4):
                row[f"jkp_{i:02d}"] = float(rng.standard_normal())
            records.append(row)
    df = pd.DataFrame.from_records(records).sort_values(["permno", "date"]).reset_index(drop=True)

    # ret_exc_next = linear combo of CURRENT features + noise. A wrong-shift
    # label is NOT predictable because features at t are uncorrelated with
    # the return at t-2 - the negative control must catch any planted leak.
    f = df[[f"jkp_{i:02d}" for i in range(4)]].to_numpy()
    coefs = np.array([0.10, -0.07, 0.05, 0.03])
    eps = rng.normal(0.0, 0.02, size=len(df))
    df["ret_exc_next"] = f @ coefs + eps
    df["ret_exc"] = rng.normal(0.005, 0.05, size=len(df))
    return df


def test_negative_control_collapses() -> None:
    """Clean signal panel should give real_r2 >> neg_r2 (PASS verdict)."""
    panel = _signal_panel(n_months=36, n_stocks=60, seed=1)
    feature_cols = [c for c in panel.columns if c.startswith("jkp_")]

    out = run_negative_control(
        panel,
        target_col="ret_exc_next",
        feature_cols=feature_cols,
        n_lags_wrong=-2,
        train_frac=0.7,
    )
    assert out["real_r2"] > 0.01, f"Real R² {out['real_r2']} too low - synth panel bug."
    assert out["verdict"] == "PASS", (
        f"Negative control FAILED on clean panel: {out}. "
        "This would indicate a leak in the audit logic itself."
    )


def test_negative_control_fails_when_leak_present() -> None:
    """Leaking the target into the features must make the WRONG-shifted run also score."""
    panel = _signal_panel(n_months=36, n_stocks=60, seed=2)
    # Inject the same shift the negative control uses (-2).
    leak = panel.sort_values(["permno", "date"]).groupby("permno")["ret_exc_next"].shift(-2)
    panel = panel.sort_values(["permno", "date"]).reset_index(drop=True).copy()
    panel["jkp_leak"] = leak.values
    panel = panel.dropna(subset=["jkp_leak"]).reset_index(drop=True)
    feature_cols = [c for c in panel.columns if c.startswith("jkp_")]

    out = run_negative_control(
        panel,
        target_col="ret_exc_next",
        feature_cols=feature_cols,
        n_lags_wrong=-2,
        train_frac=0.7,
    )
    assert out["verdict"] == "FAIL", (
        f"Negative control PASSED despite a planted leak: {out}. " "Audit is not sensitive enough."
    )
