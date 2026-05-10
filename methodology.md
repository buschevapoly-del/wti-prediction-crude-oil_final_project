# Methodology

This document explains the methodology in detail. It expands on what's in the
README and is designed to support academic citations or replication.

## Research question

> Can a multi-source predictive model (price technicals + LLM-scored news
> sentiment + macro indicators) produce a tradeable edge for WTI crude oil
> 5-day directional prediction at retail scale?

## Approach

We replicate the methodology of Dai et al. (2026) — LightGBM on multi-dimensional
GPT-scored sentiment features — with two refinements:

1. **Optuna hyperparameter optimization** (50 trials, TPE sampler, 5-fold time-series CV)
2. **5-model ensemble averaging** (different random seeds, same Optuna-tuned hyperparameters)

We add two macro indicators not in the original paper:
- **OVX**: CBOE Crude Oil Volatility Index (oil-specific fear gauge)
- **DXY**: US Dollar Index (well-documented inverse correlation with oil)

## Data

| Source | Series | Frequency | Period |
|---|---|---|---|
| FRED | DCOILWTICO (WTI spot price) | Daily | 2020-05-08 → 2026-05-08 |
| FRED | OVXCLS (Oil VIX) | Daily | Same |
| Yahoo Finance | DX-Y.NYB (Dollar Index) | Daily | Same |
| AlphaVantage News API | Energy/transportation articles | As-published | 2020-2026 |
| OpenAI GPT-4o-mini | Sentiment scoring of 4,231 articles | Per article | One-time |

**Data window: 6 years, 1,510 trading days, 707 walk-forward predictions.**

## Features

### Price technicals (6)
- `feat_RSI`: 14-day Relative Strength Index
- `feat_MACD`: MACD line (EMA12 - EMA26)
- `feat_Vol20`: 20-day rolling standard deviation of price
- `feat_Mom5`: 5-day price momentum (% change)
- `feat_Mom20`: 20-day price momentum (% change)
- `feat_BB`: Bollinger Band position (price relative to ±2σ band)

### GPT-4o-mini sentiment (5)
Each article scored on a structured prompt extracting:
- `sent_relevance`: 0-1, how related to WTI crude oil markets
- `sent_polarity`: -1 to +1, bearish to bullish
- `sent_intensity`: 0-1, weak to strong sentiment
- `sent_uncertainty`: 0-1, level of hedging language
- `sent_forwardness`: 0-1, past events vs future outlook

Daily aggregation:
- Relevance-weighted mean for each dimension
- Exponential decay smoothing (half-life = 3 days)

### OVX features (3)
- `ovx_raw`: daily OVX close
- `ovx_smooth5`: 5-day rolling mean
- `ovx_change5`: 5-day change

### DXY features (3)
- `dxy_raw`, `dxy_smooth5`, `dxy_change5` (same structure as OVX)

**Total: 17 features.**

## Target

Binary classification:
```
y_t = 1   if (price_{t+5} - price_t) / price_t > +0.30%
y_t = 0   if (price_{t+5} - price_t) / price_t < -0.30%
y_t = -1  otherwise (excluded as ambiguous)
```

The ±0.3% deadband matches Dai et al. and excludes ~5% of days as too noisy
to label confidently.

## Model

**LightGBM** binary classifier with the following Optuna search space:

| Parameter | Range | Type |
|---|---|---|
| learning_rate | [0.01, 0.15] | log-uniform |
| num_leaves | [8, 64] | integer |
| max_depth | [3, 10] | integer |
| min_child_samples | [5, 50] | integer |
| subsample | [0.6, 1.0] | uniform |
| colsample_bytree | [0.6, 1.0] | uniform |
| reg_alpha | [1e-8, 10.0] | log-uniform |
| reg_lambda | [1e-8, 10.0] | log-uniform |

Fixed: `n_estimators=300`, early stopping on validation set (patience=30).
`scale_pos_weight` set to (n_neg / n_pos) per fold for class imbalance.

## Tuning protocol

1. Fit `StandardScaler` and `SimpleImputer` on initial 50% training window
2. Run Optuna for 50 trials (or 30 minutes, whichever first)
3. For each trial: 5-fold `TimeSeriesSplit` CV, optimize average AUC
4. Take best params, freeze them for all walk-forward folds

## Walk-forward backtest

Strict temporal validation:

| Setting | Value |
|---|---|
| Initial training fraction | 50% |
| Test step size | 30 days |
| Purge gap (train → test) | 5 days |
| Validation fraction (within train) | 20% |
| Validation purge gap | 5 days |

For each fold:
1. Refit `StandardScaler` and `SimpleImputer` on training data only
2. Train 5 LightGBM models with seeds 42, 43, 44, 45, 46
3. Average predicted probabilities across the 5 models
4. Apply confidence filter: trade only if `max(p, 1-p) ≥ 0.52` AND `|p - 0.5| ≥ 0.02`
5. Apply non-overlap filter: don't open new trade until previous trade closes
6. Apply 0.2% round-trip fee to all trades

## Metrics

- **Accuracy**: fraction of correct predictions (when traded)
- **AUC**: ROC area under curve (across all predictions, traded or not)
- **Macro F1**: balanced precision/recall across UP and DOWN
- **Sharpe ratio**: annualized risk-adjusted return
  ```
  sharpe = (mean(r) / std(r)) × sqrt(252 / horizon)
  ```
- **Up recall / Down recall**: per-class recall
- **Total return**: cumulative product of trade returns (after fees)

## Baselines

- **Always-up baseline**: predict UP every day. Accuracy = fraction of UP days
- **Buy & hold**: passive long position over the full backtest period

## Honest limitations

1. **Cross-fold variance is high.** Single-fold AUC ranges from 0.50 to 0.62 in
   our experiments. The mean is ~0.58, but reporting the best fold would be
   misleading.

2. **GPT-4o-mini is mid-tier sentiment extraction.** Dai et al. show that
   GPT-4o + FinBERT ensemble adds ~3pp AUC over GPT alone. We did not implement
   this in this version due to setup cost.

3. **Walk-forward is stricter than CV.** Dai et al. report CV AUC of 0.65; our
   walk-forward AUC of 0.576 is not directly comparable but is closer to what
   you'd see in real deployment.

4. **The 2020-2026 oil regime had a strong rally.** WTI went from ~$25 (April
   2020 low) to ~$70+ (2026), a +285% return. The "always-up" baseline benefits
   disproportionately from this trend.

5. **Transaction cost assumption (0.2%) may be optimistic.** Actual retail
   futures or CFD trading costs can be higher, especially with slippage on
   illiquid intraday bars.

## What would improve results

Honest paths to better performance:

1. **Add FinBERT and/or Llama 3.2 sentiment** (Dai et al. ensemble) — likely +2-3pp
2. **Longer historical window** — 10+ years smooths regime variance
3. **Higher-frequency data** — weekly aggregation may reduce daily noise
4. **EIA inventory data** — actual fundamentals released weekly, more causal
5. **CFTC Commitments of Traders** — large trader positioning, mean-reverting

What likely won't help:
- More feature engineering on existing features (already saturated)
- Different model class (LightGBM is a strong tabular baseline)
- More Optuna trials (>100 risks overfitting validation)

## References

Dai, D., Ma, D., Liu, D., Geng, K., & Wang, Y. (2026). Beyond Polarity:
Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures Return
Prediction. *arXiv:2603.11408v2*.

Akiba, T., Sano, S., Yanase, T., Ohta, T., & Koyama, M. (2019). Optuna: A
next-generation hyperparameter optimization framework. *Proceedings of the
25th ACM SIGKDD International Conference on Knowledge Discovery & Data
Mining*, 2623-2631.

Ke, G., Meng, Q., Finley, T., Wang, T., Chen, W., Ma, W., Ye, Q., & Liu, T.
(2017). LightGBM: A highly efficient gradient boosting decision tree.
*Advances in Neural Information Processing Systems*, 30.
