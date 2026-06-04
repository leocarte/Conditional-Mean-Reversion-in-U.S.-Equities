"""Raw-file loaders for datasets on the active project surface.

Each loader takes a path (and optionally a start date) and returns a
strictly-typed DataFrame. The active CRSP, Compustat, and factor/regime
pipelines consume these frames directly; imputation / winsorisation / ranking
live in ``mlfinance.data.preprocessing`` so the lookahead audit stays local.
"""
from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# JKP factor name -> Fama-French 5 + Momentum naming. Add aliases here if a
# JKP file uses a different spelling.
JKP_TO_FF5_MAP: dict[str, str] = {
    "market_equity": "mkt_rf",
    "mkt": "mkt_rf",
    "mkt_rf": "mkt_rf",
    "size": "smb",
    "smb": "smb",
    "book_to_market": "hml",
    "hml": "hml",
    "operating_profits_to_book_equity": "rmw",
    "rmw": "rmw",
    "assets_growth": "cma",
    "cma": "cma",
    "momentum_12_1": "umd",
    "umd": "umd",
    "mom": "umd",
}


def _read_table(path: str | Path) -> pd.DataFrame:
    """Dispatch on file extension: csv/csv.gz/parquet/pkl/tsv."""
    p = Path(path)
    suffix = "".join(p.suffixes).lower()
    if suffix.endswith(".parquet") or suffix.endswith(".pq"):
        return pd.read_parquet(p)
    if suffix.endswith(".pkl") or suffix.endswith(".pickle"):
        return pd.read_pickle(p)  # nosec - trusted internal data
    if suffix.endswith(".csv.gz"):
        return pd.read_csv(p, compression="gzip", low_memory=False)
    if suffix.endswith(".csv"):
        return pd.read_csv(p, low_memory=False)
    if suffix.endswith(".tsv") or suffix.endswith(".txt"):
        return pd.read_csv(p, sep="\t", low_memory=False)
    raise ValueError(f"Unsupported file extension for {p}; expected csv/parquet/pickle.")


def _read_parquet_or_csv(path: str | Path) -> pd.DataFrame:
    """Try ``path`` first; fall back to sibling with ``.csv`` extension."""
    p = Path(path)
    if p.exists():
        return _read_table(p)
    alt = p.with_suffix(".csv")
    if alt.exists():
        logger.info("Parquet '%s' not found - falling back to CSV '%s'.", p, alt)
        return _read_table(alt)
    raise FileNotFoundError(f"Neither '{p}' nor '{alt}' exists.")


def _coerce_datetime(s: pd.Series) -> pd.Series:
    """Convert ISO strings, YYYYMMDD ints/strings, or timestamps to tz-naive datetimes."""
    if pd.api.types.is_datetime64_any_dtype(s):
        return pd.to_datetime(s, errors="coerce").dt.tz_localize(None)
    if pd.api.types.is_integer_dtype(s) or pd.api.types.is_float_dtype(s):
        return pd.to_datetime(s.astype("Int64").astype("string"), format="%Y%m%d", errors="coerce")
    sample = s.dropna().astype("string").head(20)
    if not sample.empty and sample.str.match(r"^\d{8}$").all():
        return pd.to_datetime(s.astype("string"), format="%Y%m%d", errors="coerce")
    return pd.to_datetime(s, errors="coerce")


def _lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).strip().lower() for c in out.columns]
    return out


def load_crsp_monthly(path: str, start_date: str = "1957-01-31") -> pd.DataFrame:
    """Load the simplified monthly CRSP panel.

    The Drive ships CRSP without PRC, SHROUT, dlret, or a risk-free series,
    so we cannot reconstruct market cap, splice delisting returns, or compute
    a true excess return. Instead we expose ``ret_market_adj = MthRet -
    sprtrn``. Downstream code should consume ``ret_market_adj`` rather than
    the removed ``ret_exc``.
    """
    raw = _read_parquet_or_csv(path)
    df = _lower_columns(raw)

    date_col = next(
        (c for c in ("mthcaldt", "date", "month", "datadate") if c in df.columns),
        None,
    )
    if date_col is None:
        raise KeyError("Monthly CRSP file must have a 'MthCalDt' / 'date' column.")
    df["date"] = _coerce_datetime(df[date_col]) + pd.offsets.MonthEnd(0)

    if "permno" not in df.columns:
        raise KeyError("Monthly CRSP file must have a 'PERMNO' column.")
    df["permno"] = pd.to_numeric(df["permno"], errors="coerce").astype("Int64")
    if "permco" in df.columns:
        df["permco"] = pd.to_numeric(df["permco"], errors="coerce").astype("Int64")
    else:
        df["permco"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    ret_col = next(
        (c for c in ("mthret", "ret", "r_1") if c in df.columns),
        None,
    )
    if ret_col is None:
        raise KeyError("Monthly CRSP file must have a 'MthRet' return column.")
    df["ret"] = pd.to_numeric(df[ret_col], errors="coerce")

    if "sprtrn" in df.columns:
        df["sprtrn"] = pd.to_numeric(df["sprtrn"], errors="coerce").astype("float64")
    else:
        warnings.warn(
            "load_crsp_monthly: 'sprtrn' column absent - market-adjusted return will be NaN.",
            stacklevel=2,
        )
        df["sprtrn"] = np.nan

    df["ret_market_adj"] = df["ret"].astype("float64") - df["sprtrn"].astype("float64")

    # Prefer HdrCUSIP (header/current 8-char CUSIP) since CRSP's CUSIP rotates on re-CUSIP.
    cusip_src = "hdrcusip" if "hdrcusip" in df.columns else (
        "cusip" if "cusip" in df.columns else None
    )
    if cusip_src is None:
        warnings.warn(
            "load_crsp_monthly: neither 'HdrCUSIP' nor 'CUSIP' present - CUSIP-based "
            "joins downstream will be no-ops.",
            stacklevel=2,
        )
        df["cusip"] = pd.Series([pd.NA] * len(df), dtype="string")
    else:
        df["cusip"] = df[cusip_src].astype("string").str.strip()
    df["cusip8"] = df["cusip"].str[:8]

    ticker_src = "ticker" if "ticker" in df.columns else ("tradingsymbol" if "tradingsymbol" in df.columns else None)
    if ticker_src is not None:
        df["ticker"] = df[ticker_src].astype("string").str.strip()
    else:
        df["ticker"] = pd.Series([pd.NA] * len(df), dtype="string")

    if "siccd" in df.columns:
        df["siccd"] = pd.to_numeric(df["siccd"], errors="coerce").astype("Int64")
    else:
        df["siccd"] = pd.Series([pd.NA] * len(df), dtype="Int64")
    if "naics" in df.columns:
        df["naics"] = pd.to_numeric(df["naics"], errors="coerce").astype("Int64")
    else:
        df["naics"] = pd.Series([pd.NA] * len(df), dtype="Int64")

    df = df.dropna(subset=["ret"]).copy()
    df = df.loc[df["date"] >= pd.to_datetime(start_date)].copy()

    df = df.dropna(subset=["permno"]).copy()
    df["permno"] = df["permno"].astype("int64")
    df["ret"] = df["ret"].astype("float64")
    df["sprtrn"] = df["sprtrn"].astype("float64")
    df["ret_market_adj"] = df["ret_market_adj"].astype("float64")

    keep = [
        "permno", "date", "ret", "sprtrn", "ret_market_adj",
        "cusip", "cusip8", "ticker", "permco", "siccd", "naics",
    ]
    df = df[keep].sort_values(["permno", "date"]).reset_index(drop=True)
    logger.info(
        "load_crsp_monthly: %d rows, %d unique permnos, %d unique cusip8s.",
        len(df), df["permno"].nunique(), df["cusip8"].nunique(),
    )
    return df


def load_compustat_quarterly(path: str) -> pd.DataFrame:
    """Load the quarterly Compustat (CompFirmCharac) dump.

    Adds ``availability_date = datadate + 6 months`` per Gu-Kelly-Xiu (2020)
    reporting-lag convention. All downstream merge_asof joins key off it.
    Any non-identifier numeric column is kept as a candidate fundamental.
    """
    raw = _read_parquet_or_csv(path)
    df = _lower_columns(raw)

    if "datadate" not in df.columns:
        raise KeyError("Compustat file must have a 'datadate' column.")
    df["datadate"] = _coerce_datetime(df["datadate"])
    df = df.dropna(subset=["datadate"]).copy()

    if "gvkey" not in df.columns:
        raise KeyError("Compustat file must have a 'gvkey' column.")
    df["gvkey"] = (
        df["gvkey"].astype("string").str.strip().str.replace(r"\.0$", "", regex=True).str.zfill(6)
    )

    if "cusip" in df.columns:
        df["cusip"] = df["cusip"].astype("string").str.strip()
    else:
        df["cusip"] = pd.Series([pd.NA] * len(df), dtype="string")
    df["cusip8"] = df["cusip"].str[:8]

    if "cik" in df.columns:
        df["cik"] = (
            df["cik"]
            .astype("string")
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
            .str.zfill(10)
        )
        # zfill turns '<NA>' into 10 zeros; map all-zeros back to NA.
        df["cik"] = df["cik"].mask(df["cik"].eq("0" * 10), pd.NA)
    else:
        df["cik"] = pd.Series([pd.NA] * len(df), dtype="string")

    df["availability_date"] = df["datadate"] + pd.DateOffset(months=6)

    id_cols = ["gvkey", "cik", "cusip", "cusip8", "datadate", "availability_date"]
    str_context_cols = [c for c in ("tic", "conm", "indfmt", "consol", "popsrc", "datafmt",
                                    "costat", "fic", "exchg", "curcdq")
                        if c in df.columns]
    keep_non_numeric = set(id_cols) | set(str_context_cols)

    numeric_cols: list[str] = []
    for col in df.columns:
        if col in keep_non_numeric:
            continue
        coerced = pd.to_numeric(df[col], errors="coerce")
        if coerced.notna().any():
            df[col] = coerced.astype("float64")
            numeric_cols.append(col)

    df = df[id_cols + str_context_cols + numeric_cols].copy()
    df = df.sort_values(["gvkey", "availability_date"]).reset_index(drop=True)
    logger.info(
        "load_compustat_quarterly: %d rows, %d gvkeys, %d ciks, %d cusip8s, %d numeric fundamentals.",
        len(df),
        df["gvkey"].nunique(),
        df["cik"].dropna().nunique(),
        df["cusip8"].dropna().nunique(),
        len(numeric_cols),
    )
    return df


def _standardise_factor_long(df: pd.DataFrame, factor_label: str) -> pd.DataFrame:
    """Normalise a long-format factor table to (date, name, n_stocks, ret)."""
    out = _lower_columns(df)

    date_col = next(
        (c for c in ("date", "eom", "month", "datadate", "yyyymm") if c in out.columns),
        None,
    )
    if date_col is None:
        raise KeyError(f"{factor_label}: no date-like column found among {list(out.columns)}.")
    out["date"] = _coerce_datetime(out[date_col]) + pd.offsets.MonthEnd(0)

    name_col = next(
        (c for c in ("name", "factor", "characteristic", "signalname") if c in out.columns),
        None,
    )
    if name_col is None:
        raise KeyError(f"{factor_label}: no name-like column found among {list(out.columns)}.")
    out["name"] = out[name_col].astype("string").str.strip()

    ret_col = next(
        (c for c in ("ret", "return", "r", "long_short", "ls") if c in out.columns),
        None,
    )
    if ret_col is None:
        raise KeyError(f"{factor_label}: no return-like column found among {list(out.columns)}.")
    out["ret"] = pd.to_numeric(out[ret_col], errors="coerce").astype("float64")

    if "n_stocks" in out.columns:
        out["n_stocks"] = pd.to_numeric(out["n_stocks"], errors="coerce").astype("Int64")
    elif "n" in out.columns:
        out["n_stocks"] = pd.to_numeric(out["n"], errors="coerce").astype("Int64")
    else:
        out["n_stocks"] = pd.Series([pd.NA] * len(out), dtype="Int64")

    return out[["date", "name", "n_stocks", "ret"]].dropna(subset=["date", "name", "ret"]).copy()


def _pivot_to_wide(long_df: pd.DataFrame, label: str) -> pd.DataFrame:
    """Pivot a long (date, name, ret) table to wide date x factor_name."""
    dup_mask = long_df.duplicated(subset=["date", "name"], keep=False)
    if dup_mask.any():
        warnings.warn(
            f"{label}: {int(dup_mask.sum())} duplicate (date, name) rows averaged.",
            stacklevel=3,
        )
        long_df = long_df.groupby(["date", "name"], as_index=False)["ret"].mean()

    wide = long_df.pivot(index="date", columns="name", values="ret")
    wide.columns.name = None
    wide = wide.sort_index().reset_index()
    return wide


def load_jkp(path: str) -> pd.DataFrame:
    """Load the JKP factor panel (long format) and pivot to wide ``date x name``."""
    long_df = _standardise_factor_long(
        _read_parquet_or_csv(path), factor_label="load_jkp",
    )
    wide = _pivot_to_wide(long_df, label="load_jkp")
    logger.info(
        "load_jkp: %d dates, %d factors.",
        len(wide),
        wide.shape[1] - 1,
    )
    return wide


def load_chen_zimmerman(path: str) -> pd.DataFrame:
    """DEPRECATED: load the Chen-Zimmerman anomaly panel (not on Drive)."""
    long_df = _standardise_factor_long(
        _read_table(path), factor_label="load_chen_zimmerman",
    )
    wide = _pivot_to_wide(long_df, label="load_chen_zimmerman")
    logger.info(
        "load_chen_zimmerman: %d dates, %d anomalies.",
        len(wide),
        wide.shape[1] - 1,
    )
    return wide


def load_ff5_umd_from_jkp(jkp_wide: pd.DataFrame) -> pd.DataFrame:
    """Construct a Fama-French 5 + UMD panel from the wide JKP table.

    Mapping lives in :data:`JKP_TO_FF5_MAP`. Missing FF columns are filled
    with NaN and a warning is logged.
    """
    if "date" not in jkp_wide.columns:
        raise KeyError("load_ff5_umd_from_jkp: input is missing the 'date' column.")
    out = pd.DataFrame({"date": jkp_wide["date"].values})
    targets = ("mkt_rf", "smb", "hml", "rmw", "cma", "umd")

    reverse: dict[str, list[str]] = {t: [] for t in targets}
    for jkp_name, ff_name in JKP_TO_FF5_MAP.items():
        if ff_name in reverse:
            reverse[ff_name].append(jkp_name)

    for ff_name in targets:
        chosen: str | None = None
        for alias in reverse[ff_name]:
            if alias in jkp_wide.columns:
                chosen = alias
                break
        if chosen is None:
            warnings.warn(
                f"load_ff5_umd_from_jkp: no JKP column found for FF factor '{ff_name}'.",
                stacklevel=2,
            )
            out[ff_name] = np.nan
        else:
            out[ff_name] = pd.to_numeric(jkp_wide[chosen], errors="coerce").astype("float64")

    return out.sort_values("date").reset_index(drop=True)
