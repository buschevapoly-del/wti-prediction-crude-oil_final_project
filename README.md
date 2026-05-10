# 🛢️ WTI Crude Oil Forecasting Dashboard

An honest replication and extension of [Dai et al. (2026)](https://arxiv.org/abs/2603.11408)
"Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for WTI Crude Oil Futures
Return Prediction" — with two methodological refinements: **Optuna hyperparameter
tuning** and **5-model ensemble averaging**.

> **Live dashboard**: [your-app-name.streamlit.app](https://share.streamlit.io/) *(deploy your own copy — see below)*

---

## Honest result

| Metric | Value | Reference |
|---|---|---|
| **Accuracy** | 55.56% | Dai et al.: 56.08% |
| **AUC** | 0.576 | Dai et al.: 0.65 |
| **Sharpe** | -0.30 | — |
| **Total return** | -29.9% | Buy & Hold: +285.7% |
| **Edge vs always-up** | +0.10 pp | — |

The methodology (Optuna + 5-model ensemble + 17 features) closely matches the
published frontier, but produces a **negligible edge over the trivial "always-up"
baseline** (55.46%). After realistic 0.2% round-trip transaction fees, the strategy
underperforms buy & hold by a wide margin.

This is a useful negative finding. It's consistent with the broader literature
showing that retail-accessible sentiment data provides limited tradeable edge for
liquid commodities like WTI crude oil.

---

## Why this project exists

Most "AI predicts oil prices" claims fail honest replication. This project
deliberately uses rigorous methodology and reports the actual result rather
than cherry-picked best folds:

- **Walk-forward validation** with 5-day purge gap (no future data leaks into training)
- **0.2% round-trip fees** modeled (actual retail trading cost)
- **5-model ensemble** to reduce single-model variance
- **Optuna** hyperparameter tuning over 50 trials with time-series CV
- **Honest baselines**: always-up (~55%) and buy-and-hold (+286% over 2020-2026)
- **Reported limitations**: GPT-4o-mini sentiment is mid-tier; cross-fold variance is high

---

## Repository structure

```
.
├── README.md                       this file
├── requirements.txt                Python dependencies
├── LICENSE                         MIT
├── .gitignore                      excludes secrets and cache
│
├── wti_optuna_ensemble.py          the pipeline: data → model → outputs
├── dashboard.py                    Streamlit app reading the outputs
│
├── outputs_optuna_ensemble/        pre-computed results (committed for dashboard)
│   ├── predictions.csv             per-day predictions
│   ├── feature_importance.csv      averaged across folds
│   └── metadata.json               run config + final metrics
│
├── docs/
│   └── methodology.md              detailed methodology, citations
│
└── .streamlit/
    └── config.toml                 dashboard theme
```

---

## Architecture

Two-part design separates slow/expensive computation from fast UI:

### Pipeline (`wti_optuna_ensemble.py`)
- Fetches WTI price data (FRED, free)
- Fetches OVX volatility index (FRED, free)
- Fetches DXY dollar index (Yahoo Finance, free)
- Loads cached GPT-4o-mini sentiment scores (4,231 articles from AlphaVantage)
- Runs Optuna hyperparameter search (50 trials, ~10 min)
- Walks forward through ~2 years of test data with 5-model ensemble
- Writes results to `outputs_optuna_ensemble/`

**Cost: ~$0.50 in OpenAI fees for the original sentiment scoring (one time).**
**Runtime: ~30-45 minutes total.**

### Dashboard (`dashboard.py`)
- Reads pre-computed CSVs from `outputs_optuna_ensemble/`
- Calls **no paid APIs** at runtime — fast and free to host
- Four tabs: Performance, Methodology, Feature Importance, Honest Findings

---

## How to run locally

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/wti-forecast-dashboard.git
cd wti-forecast-dashboard

# Install dependencies
pip3 install -r requirements.txt

# (Optional) Re-run the pipeline to regenerate outputs.
# The repo already includes the outputs, so this is only needed if you want to
# update or retrain. Set API keys first:
export OPENAI_API_KEY="sk-..."        # only needed if regenerating cache
export ALPHAVANTAGE_KEY="..."         # only needed if regenerating cache
python3 wti_optuna_ensemble.py

# Run the dashboard
streamlit run dashboard.py
```

The dashboard opens at `http://localhost:8501`.

---

## How to deploy to Streamlit Community Cloud (free)

Streamlit Cloud hosts your dashboard at a public URL for free.

### Step 1: Create a public GitHub repo

```bash
git init
git add .
git commit -m "Initial commit: WTI forecasting dashboard"
git branch -M main
git remote add origin https://github.com/YOUR_USERNAME/wti-forecast-dashboard.git
git push -u origin main
```

**Important**: confirm `.gitignore` is preventing API keys and cache files from being committed:

```bash
git ls-files | grep -i "key\|env\|cache"   # should return nothing sensitive
```

### Step 2: Deploy on Streamlit Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io/)
2. Sign in with GitHub
3. Click **"New app"**
4. Repository: `YOUR_USERNAME/wti-forecast-dashboard`
5. Branch: `main`
6. Main file path: `dashboard.py`
7. Click **"Deploy"**

Live URL ready in 2-5 minutes at `https://YOUR_USERNAME-wti-forecast-dashboard.streamlit.app/`.

### Step 3: Update predictions later (optional)

```bash
# Locally:
python3 wti_optuna_ensemble.py
git add outputs_optuna_ensemble/
git commit -m "Update predictions $(date +%Y-%m-%d)"
git push

# Streamlit Cloud auto-redeploys within 1-2 minutes.
```

---

## Methodology summary

| Component | Choice | Justification |
|---|---|---|
| Model | LightGBM | Strong on tabular data, Dai et al. baseline |
| Tuning | Optuna (50 trials, TPE sampler) | Bayesian search beats grid/random |
| Validation | TimeSeriesSplit (5 folds) inside Optuna | No future leakage during tuning |
| Backtest | Walk-forward, 30-day step, 5-day purge | Stricter than 5-fold CV |
| Ensemble | 5 LightGBMs with different seeds | Reduces single-model variance |
| Features | 17 (price + GPT + OVX + DXY) | Matches Dai et al. core methodology |
| Threshold | Trade only at confidence ≥ 52% | Avoid uncertain predictions |
| Fees | 0.2% round-trip | Realistic retail commission + slippage |

See [`docs/methodology.md`](docs/methodology.md) for full details.

---

## Limitations (honest)

- **6 years of data** (2020-2026) is short for robust ML conclusions
- **GPT-4o-mini sentiment is mid-tier**; the published paper used GPT-4o + FinBERT
  ensemble which may add 2-3pp accuracy
- **Walk-forward variance is high** — single-fold results swing ±5pp
- **Transaction cost assumed at 0.2%** — real retail costs may be higher
- **Survivor bias**: 2020-2026 had a strong oil rally; "always-up" looks artificially good
- **Not real trading**: backtest assumes executable close-to-close fills with no slippage
- **Dai et al.'s headline finding** (uncertainty/intensity rank highest) does not
  fully replicate in our data — DXY and price volatility rank highest

---

## Comparison to prior experiments

This methodology is the strongest of 7 configurations tested. Results across runs:

| Configuration | Accuracy | Notes |
|---|---|---|
| Run 1: price + F&G + GPT sentiment | 52.86% | Original full pipeline |
| Run 2: same as Run 1 (sentiment bug) | 57.67% | Sentiment was effectively zero |
| Run 3: price + F&G only | 50.65% | No sentiment baseline |
| OVX + DXY only (no sentiment) | 55.10% | Macro indicators alone |
| Price + sentiment + OVX + DXY | 52.35% | All features, default LightGBM |
| **Optuna + ensemble (this work)** | **55.56%** | Methodology refinement |
| Always-up baseline | 55.46% | Predict UP every day |
| Dai et al. (2026) published | 56.08% | GPT-4o + FinBERT, 5-fold CV |

---

## Citation

If you use or reference this work:

```bibtex
@misc{wti_forecast_2026,
  title  = {WTI Crude Oil Forecasting Dashboard: An Honest Replication of Dai et al.},
  author = {Your Name},
  year   = {2026},
  url    = {https://github.com/YOUR_USERNAME/wti-forecast-dashboard}
}
```

Original paper:

```bibtex
@article{dai2026beyond,
  title   = {Beyond Polarity: Multi-Dimensional LLM Sentiment Signals for
             WTI Crude Oil Futures Return Prediction},
  author  = {Dai, Dehao and Ma, Ding and Liu, Dou and Geng, Kerui and Wang, Yiqing},
  journal = {arXiv preprint arXiv:2603.11408},
  year    = {2026}
}
```

---

## License

MIT — see [`LICENSE`](LICENSE).

---

## Disclaimer

This is a research project. **Not financial advice.** Do not trade real money
based on the predictions in this dashboard. The model has demonstrated only a
+0.10 percentage point edge over the always-up baseline, which is well within
statistical noise. After realistic transaction costs, the strategy returns
-29.9% while buy & hold returned +285.7% over the same period.
