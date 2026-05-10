# WTI Crude Oil 5-Day Direction Forecast — Dashboard

LightGBM classifier with Chain-of-Thought GPT sentiment and Fear & Greed Index features, trained on 2020–2026 WTI data. Walk-forward backtest with non-overlapping trades and transaction costs.

## Result Summary

- **Accuracy:** 57.24% across 1,527 predictions 
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

## Methodology

See the **Methodology** tab in the live dashboard for full details on the model architecture, sentiment pipeline, and validation procedure.
