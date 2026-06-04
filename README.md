# Conditional Mean Reversion in U.S. Equities

Authors: Leonardo Cartesegna, Giacomo Case

Course: FIN-407 Machine Learning in Finance

Institution: EPFL

## Abstract

This repository contains the final submission for a cross-sectional U.S.
equity return-prediction project based only on course-provided data. We study
whether machine-learning models can improve next-month stock ranking relative
to classical reversal and momentum signals, and whether those rankings survive
flat transaction costs in a market-neutral long-short implementation. The
final audited expanding-window results over 2017-2024 show that nonlinear
models materially outperform classical reversal benchmarks. The strongest
reported specification is the flow-Compustat MLP, with net Sharpe about `2.03`
at `10` bps, annualized net return about `21.4%`, and breakeven costs near
`90` bps. Momentum remains a strong classical benchmark, regime dependence is
mostly an evaluation heterogeneity result rather than a consistent feature
gain, and missing CRSP fields limit tradability screening.

## Research Question

1. Can nonlinear machine learning improve next-month cross-sectional stock
   ranking?
2. Does it beat classical reversal and momentum baselines under a shared
   portfolio protocol?
3. Does the signal remain attractive after transaction costs?
4. Does performance vary across market states?
5. What are the main limits imposed by the provided-data constraint?

## Repository Structure

```text
MLfinance/
├── README.md
├── LICENSE
├── Makefile
├── pyproject.toml
├── requirements.txt
├── configs/
├── data/
├── mlfinance/
├── report/
├── scripts/
└── tests/
```

Key directories:

- `configs/`: central configuration, with [`configs/main.yaml`](configs/main.yaml)
  as the main source of truth.
- `data/`: data location conventions and `.gitkeep` placeholders. Raw vendor
  files and generated outputs are intentionally not committed.
- `mlfinance/`: package code for data loading, feature engineering, models,
  walk-forward evaluation, backtesting, and report-output generation.
- `report/`: final LaTeX source, compiled PDF, and curated tables/figures used
  in the submission.
- `scripts/`: small executable entrypoints used by the Makefile.
- `tests/`: unit and integration tests for the cleaned submission codebase.

## Installation and Dependencies

Python `3.10+` is required.

Install with the project metadata:

```bash
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If you prefer the pinned requirements file:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev]"
```

## Data

No external datasets are allowed. Raw input files are not committed.

Expected local inputs:

- `data/raw/monthly_crsp.parquet` or `data/raw/monthly_crsp.csv`
- `data/raw/CompFirmCharac.parquet` for the Compustat-augmented runs

Important limitation:

- The provided Monthly CRSP extract does not include `prc`, `shrout`, `shrcd`,
  or `exchcd`.
- As a result, true price, share-code, exchange, microcap, and liquidity
  screens are outside feasible scope in this project and are discussed as a
  limitation rather than claimed as completed robustness evidence.

## Reproducing the Results

Minimal checks:

```bash
make test
make lint
make checks
```

Build the CRSP parquet input from CSV if needed:

```bash
make crsp-parquet
```

Rebuild the processed panel and baseline pipeline:

```bash
make build-panel
```

Run the locked-split benchmark families:

```bash
make linear-benchmarks
make nonlinear-benchmarks
```

Run the main expanding-window evaluation and supporting report tables:

```bash
make walkforward
make classical-baselines
make walkforward-robustness
make final-report-outputs
```

Compile the final PDF:

```bash
make report-pdf
```

## Key Components

- `mlfinance/data/`: raw-data loading, preprocessing, and universe helpers.
- `mlfinance/features/`: return, regime, and flow-Compustat feature blocks.
- `mlfinance/models/`: baselines, linear models, XGBoost, and compact MLP.
- `mlfinance/run/`: executable pipelines for panel construction, benchmark
  runs, walk-forward evaluation, robustness summaries, and final report
  outputs.
- `mlfinance/backtest/` and `mlfinance/eval/`: portfolio construction,
  transaction-cost handling, predictive diagnostics, and statistical tests.

## Report

The final report assets are in [`report/`](report/):

- [`report/report.pdf`](report/report.pdf)
- [`report/report.tex`](report/report.tex)
- [`report/report_body.tex`](report/report_body.tex)
- [`report/tables/`](report/tables/)
- [`report/figures/`](report/figures/)

## License / Course Note

This repository is academic coursework submitted at EPFL for `FIN-407 Machine
Learning in Finance`. The code is provided for grading and academic reference.
