# Data

This repository does not commit raw vendor files or generated model outputs.
Only placeholders such as `.gitkeep` are tracked under `data/raw/` and
`data/processed/`.

## Expected Raw Inputs

The project expects the following local files when reproducing results:

- `data/raw/monthly_crsp.parquet` or `data/raw/monthly_crsp.csv`
- `data/raw/CompFirmCharac.parquet` for the Compustat-augmented runs

Accepted Monthly CRSP fields include:

- `PERMNO`
- `HdrCUSIP` or `CUSIP`
- `Ticker` or `TradingSymbol`
- `PERMCO`
- `SICCD`
- `NAICS`
- `MthCalDt`
- `MthRet`
- `sprtrn`

The loaders standardize these into the internal panel schema used by the
package.

## Targets

Main target:

```text
target_ret_fwd_1m = R_{i,t+1}
```

Supplementary market-adjusted target:

```text
target_excess_ret_fwd_1m = R_{i,t+1} - sprtrn_{t+1}
```

Future `sprtrn` values are used only for target construction, never as model
features.

## Generated Data Policy

Typical local outputs include:

- `data/processed/model_panel.parquet`
- `data/outputs/predictions/`
- `data/outputs/backtests/`
- `data/outputs/metrics/`

These files are generated locally and should remain untracked.

## Scope Limitation

The provided Monthly CRSP extract does not contain:

- `prc`
- `shrout`
- `shrcd`
- `exchcd`

That prevents true price, share-code, exchange, microcap, and liquidity
screens. The project therefore reports flat-cost robustness on the provided
universe and treats tradability screening as an explicit limitation.
