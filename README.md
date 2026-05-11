# WTI Crude Oil 5-Day Direction Forecast — Decision Support Dashboard

LightGBM with Chain-of-Thought GPT sentiment + Fear & Greed Index features.

## What this dashboard does

Decision-support tool, not just performance reporter. Answers two questions:

1. **What should I do today?** → Today's signal card with BUY/SELL/HOLD, confidence, edge, suggested position size, target, and stop loss.
2. **Can I trust the model?** → Recent trades table + full risk metrics.

## Headline result

- Accuracy: 57.24% (random walk: 50.00% — edge: +7.24 pp)
- Total return: +61.12% across 168 trades
- Sharpe: 0.535
- Win rate: 53.57%

## Local development

```bash
pip install -r requirements.txt
streamlit run dashboard.py
```

## Files

```
dashboard.py
requirements.txt
outputs_optuna_ensemble/
  metadata.json
  predictions.csv
```
