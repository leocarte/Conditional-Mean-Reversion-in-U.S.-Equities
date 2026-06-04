# Conditional Mean Reversion in U.S. Equities

## Abstract

This project asks whether nonlinear machine-learning models can improve
next-month cross-sectional U.S. equity return prediction and whether those
signals remain economically useful after transaction costs. Using only the
course-provided CRSP and Compustat-style data, the evidence comes from the
final audited expanding-window walk-forward package over 2017-2024.
The main strategy converts monthly forecasts into equal-weight top-minus-bottom
decile portfolios. The strongest specification is the
flow-Compustat MLP, which reaches net Sharpe `2.03` at `10` bps, annualized
net return `21.4%`, rank-IC HAC `t = 6.36`, and breakeven costs near
`90` bps. It clearly outperforms same-protocol reversal and z-score reversal.
Momentum remains the strongest classical benchmark; ML is economically stronger
but not cleanly significant against momentum in the same-protocol
mean-difference tests. Performance is better in bear and high-volatility
states, but direct regime-feature ablations do not improve the strongest
flow-Compustat models. The report's key limitation is that the provided
Monthly CRSP extract lacks `prc`, `shrout`, `shrcd`, and `exchcd`, so the
result is best interpreted as a flat-cost academic finding rather than a full
tradability proof.

## 1. Introduction

The economic question is not whether raw one-month reversal exists in the
abstract, but whether conditional next-month return forecasts can rank stocks
better than fixed-rule reversal portfolios once turnover and costs are taken
seriously. This project treats machine learning as a forecasting and ranking
tool rather than as black-box trading magic.

The report uses a final audited expanding-window protocol designed to prevent
leakage and align all model comparisons. It compares nonlinear models, linear
benchmarks, and same-protocol classical strategies under a shared long-short
backtest engine. The report focuses on
five issues: out-of-sample predictive ranking, comparison with classical
reversal and momentum benchmarks, transaction-cost survival, regime
dependence, and the limits imposed by the provided data.

## 2. Data and Prediction Problem

The main label is `target_ret_fwd_1m`, the next-month stock return from
Monthly CRSP. A supplementary robustness exercise uses
`target_excess_ret_fwd_1m`, but the headline result remains the raw
return target. Predictors come from two approved sources: lagged CRSP return
features and an audited flow-only Compustat block merged point-in-time.

The sample spans `1984-01-31` to `2024-11-30`, with `3,646,731`
stock-months, `33,884` unique stocks, and `7,427` stocks per month on
average. The final audited walk-forward test window covers 2017-2024, with
training starting at `1985-01-31`.

The key data limitation is structural. The provided Monthly CRSP file omits
`prc`, `shrout`, `shrcd`, and `exchcd`, so true
price/share-code/exchange/microcap/liquidity filtering is not feasible under
the no-external-data rule. The report therefore claims flat-cost robustness on
the provided universe, not a complete executable tradability result.

## 3. Features, Models, and Validation Design

The baseline feature set, `return_only`, contains `15` lagged-return
predictors, including recent returns, momentum windows, volatility, drawdown,
beta, idiosyncratic volatility, and rank transforms. The main feature set,
`flow_compustat`, adds audited flow-only firm characteristics such as sales,
capex, research and development, SG&A, margins, growth rates, and intensity
ratios.

The model stack has three layers. Classical baselines are reversal,
z-score reversal, momentum, and reversal plus momentum. Linear benchmarks are
Ridge and Elastic Net. Nonlinear models are XGBoost and a compact MLP. The
final report centers on four ML strategies: return-only XGBoost,
return-only MLP, flow-Compustat XGBoost, and flow-Compustat MLP.

Validation is strict. For each test year from 2017 through 2024, training
uses data through `Y-6`, validation uses `Y-5` through `Y-1`, and the
calendar year `Y` remains untouched until final evaluation. Hyperparameters
are selected on validation data only, and only out-of-sample predictions enter
the backtest.

## 4. Portfolio Construction and Transaction Costs

Forecasts are converted into monthly cross-sectional ranks. The main strategy
buys the top `10%` of names and shorts the bottom `10%`, equal-weighting each
side. The portfolio is rebalanced monthly, and performance is reported gross
and net of flat transaction costs. The main reported cost level is `10` bps,
with robustness over a wider grid up to `100` bps.

This cost model is intentionally simple. It is informative for comparing
signals with a shared turnover penalty, but it is not a substitute for a full
liquidity-aware execution model.

## 5. Results

The central final hierarchy is in
[table_final_model_hierarchy.csv](tables/table_final_model_hierarchy.csv).
The strongest specification is `flow_compustat x mlp`, with net Sharpe `2.028`,
gross Sharpe `2.281`, annualized net return `21.4%`, turnover `2.21`,
breakeven costs `90.5` bps, and rank-IC HAC `t = 6.36`.

Three comparisons matter most.

1. Nonlinear models dominate the linear benchmarks in the detailed
   walk-forward table
   [table_walkforward_model_summary.csv](tables/table_walkforward_model_summary.csv).
2. Adding audited flow-Compustat features materially helps the nonlinear
   models. Net Sharpe rises from `1.192` to `2.028` for the MLP and from
   `1.313` to `1.770` for XGBoost.
3. The strongest models remain robust to flat costs. In
   [table_cost_breakeven.csv](tables/table_cost_breakeven.csv),
   the two flow-Compustat nonlinear models still have positive net Sharpe at
   `75` bps and only turn slightly negative near `100` bps.

The visual evidence in
[figure_cumulative_net_returns.png](figures/figure_cumulative_net_returns.png)
and
[figure_net_sharpe_vs_cost.png](figures/figure_net_sharpe_vs_cost.png)
matches the same ranking.

## 6. Robustness and Ablations

Same-protocol classical baselines sharpen the interpretation. Reversal and
z-score reversal are weak under the final 2017-2024 protocol, with net
Sharpe only `0.062`. Momentum is a real benchmark at net Sharpe `0.636` and
very high breakeven costs. The Newey-West mean-difference tests in
[table_classical_model_difference_tests.csv](tables/table_classical_model_difference_tests.csv)
show that the leading ML strategies strongly beat reversal and z-score
reversal, but not momentum at conventional significance thresholds.
Same-protocol classical baselines close the main benchmark gap, and momentum
is a stronger classical benchmark than reversal.

Prediction sorting is economically credible. In
[table_prediction_deciles.csv](tables/table_prediction_deciles.csv),
the flow-Compustat MLP produces a gross top-minus-bottom spread of about
`2.00%` per month, with decile `1` at `-1.02%` and decile `10` at `1.05%`.
The corresponding XGBoost spread is about `1.99%`.

Performance is also regime-dependent. From
[table_regime_robustness.csv](tables/table_regime_robustness.csv),
the flow-Compustat MLP reaches net Sharpe `3.41` in bear months and `2.56` in
high-volatility months, while still remaining positive in calmer states.
However, the direct regime-feature ablation in
[table_supplementary_robustness.csv](tables/table_supplementary_robustness.csv)
does not improve the strongest models: `flow_compustat_regime x mlp` falls to
net Sharpe `1.520` from `2.028`, and `flow_compustat_regime x xgboost` falls
to `1.439` from `1.770`. The best reading is that regimes matter for
evaluation, but simple regime features do not robustly improve the strongest
flow-Compustat specifications. We therefore treat regime dependence mainly as
an evaluation dimension.

Additional robustness is supportive but secondary. The excess-return target
raises the main flow-Compustat MLP to net Sharpe `2.970` and breakeven costs
near `124` bps. A three-seed MLP robustness check yields mean net Sharpe
`1.968`, standard deviation `0.227`, minimum `1.717`, and maximum `2.159`.
These checks do not replace the main raw-target final audited result.

## 7. Limitations

This is a provided-data flat-cost universe study, not a complete tradability
exercise. Missing CRSP fields prevent true tradability screens, and the cost
model is simplified. Monthly frequency also compresses implementation frictions
into a single turnover penalty.

Even with a strict out-of-sample protocol, ML finance results remain exposed
to model-selection and data-mining risk. The final walk-forward design
reduces leakage risk, but it does not eliminate broader research-process
selection concerns.

## 8. Conclusion

Within the provided-data setting, conditional nonlinear models improve
cross-sectional mean-reversion evidence relative to simple reversal
benchmarks. The strongest strategy is the flow-Compustat MLP, with
net Sharpe `2.03`, breakeven costs near `90` bps, and strong rank-IC evidence.
Flow-Compustat XGBoost confirms that the result is not model-specific.

The conclusion remains measured. ML strongly dominates reversal and z-score
reversal, but momentum remains an important benchmark. Regime dependence
appears in realized performance heterogeneity, while direct regime features do
not help the strongest model. The report therefore provides credible academic
evidence for conditional mean reversion under the assignment constraints, not a
production-ready implementation claim.
