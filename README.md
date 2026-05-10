[README (3).md](https://github.com/user-attachments/files/27573079/README.3.md)
# WTI Crude Oil 5-Day Direction Forecast — Dashboard

LightGBM with Chain-of-Thought GPT sentiment + Fear & Greed Index features. Walk-forward backtested on 2020–2026 WTI data.

## Result Summary

- **Accuracy:** 57.24% (random walk baseline: 50.00% — edge: +7.24 pp)
- **Total return:** +86.13% across 168 trades
- **Sharpe ratio:** 0.535 (annualized, 5-day horizon)

## Local development

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Files

```
.
├── dashboard.py
├── requirements.txt
└── outputs_optuna_ensemble/
    ├── metadata.json
    └── predictions.csv
```

The dashboard fetches recent WTI price data from yfinance at runtime to overlay prediction outcomes.
