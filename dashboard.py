"""
==============================================================================
  dashboard.py — WTI Forecasting Dashboard (Optuna + Ensemble)
  -----------------------------------------------------------------------------
  Reads pre-computed CSV/JSON outputs from wti_optuna_ensemble.py and renders
  them as an interactive Streamlit dashboard.

  Run locally:    streamlit run dashboard.py
  Deploy:         push to GitHub, deploy via share.streamlit.io

  This dashboard uses NO paid APIs at runtime. All slow/expensive computation
  happens in the pipeline. The dashboard just reads outputs_optuna_ensemble/.
==============================================================================
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────
# Page config
# ──────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WTI Crude Oil Forecasting — Research Dashboard",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

OUT_DIR = Path("outputs_optuna_ensemble")


# ──────────────────────────────────────────────────────────────────────
# Data loading
# ──────────────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_data():
    try:
        with open(OUT_DIR / "metadata.json") as f:
            meta = json.load(f)
        preds = pd.read_csv(OUT_DIR / "predictions.csv", parse_dates=["date"])
        feat_imp = pd.read_csv(OUT_DIR / "feature_importance.csv")
        return meta, preds, feat_imp
    except FileNotFoundError:
        return None, None, None


# ──────────────────────────────────────────────────────────────────────
# Header
# ──────────────────────────────────────────────────────────────────────
st.title("🛢️ WTI Crude Oil Forecasting — Research Dashboard")
st.caption(
    "An honest replication and extension of Dai et al. (2026) using "
    "Optuna hyperparameter tuning and 5-model ensemble averaging. "
    "**Not financial advice.**"
)

data = load_data()
if data[0] is None:
    st.error(
        "⚠️ No model outputs found in `outputs_optuna_ensemble/`.\n\n"
        "Run `python3 wti_optuna_ensemble.py` first to generate the data."
    )
    st.stop()

meta, preds, feat_imp = data


# ──────────────────────────────────────────────────────────────────────
# Top-line metrics
# ──────────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)

c1.metric(
    "Accuracy (out-of-sample)",
    f"{meta['accuracy'] * 100:.2f}%",
    delta=f"{meta['edge_vs_baseline_pp']:+.2f} pp vs always-up baseline" if "edge_vs_baseline_pp" in meta else None,
    help="Walk-forward backtest accuracy. Always-up baseline is shown for comparison.",
)

c2.metric(
    "AUC",
    f"{meta['auc']:.3f}",
    delta=f"{meta['auc'] - 0.5:+.3f} vs random",
    help="Area under ROC curve. 0.5 = random, 1.0 = perfect.",
)

c3.metric(
    "Sharpe ratio",
    f"{meta['sharpe']:.2f}",
    help="Annualized risk-adjusted return. >0.5 = tradeable, >1.0 = good.",
)

c4.metric(
    "Trades placed",
    f"{meta['n_trades']}",
    delta=f"{meta['n_trades'] / max(meta.get('n_predictions', 1), 1) * 100:.1f}% of predictions",
    help="Only trades when ensemble confidence ≥ 52%.",
)

st.divider()


# ──────────────────────────────────────────────────────────────────────
# Tabs
# ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Performance",
    "🔬 Methodology",
    "📈 Feature Importance",
    "📋 Honest Findings",
])


# ══════════════════════════════════════════════════════════════════════
# TAB 1 — PERFORMANCE
# ══════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Equity Curve — Strategy vs Buy & Hold")

    # Build equity curve
    df = preds.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["strategy_eq"] = (1 + df["ret"].where(df["traded"], 0.0)).cumprod()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["date"], y=df["strategy_eq"],
        mode="lines", name="Optuna + Ensemble strategy",
        line=dict(color="#1f77b4", width=2),
    ))

    # Reference: B&H total return for the period
    bh_final = 1 + meta.get("buy_hold_return", 0)
    fig.add_trace(go.Scatter(
        x=[df["date"].min(), df["date"].max()],
        y=[1.0, bh_final],
        mode="lines", name=f"Buy & Hold ({meta.get('buy_hold_return', 0)*100:+.0f}%)",
        line=dict(color="#7f7f7f", width=2, dash="dash"),
    ))

    fig.update_layout(
        height=400,
        hovermode="x unified",
        xaxis_title="Date",
        yaxis_title="Cumulative return (1.0 = starting capital)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "The strategy line shows cumulative returns from trades only. "
        "The buy & hold line is the reference for the same period. "
        "If the strategy line is below buy & hold, holding outperformed."
    )

    st.divider()

    # Rolling accuracy
    st.subheader("Rolling 90-Day Accuracy")
    pr = df[df["traded"]].copy()
    if len(pr) > 30:
        pr["correct"] = (pr["pred"] == pr["actual"]).astype(int)
        pr = pr.set_index("date").sort_index()
        roll_acc = pr["correct"].rolling("90D", min_periods=10).mean()

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(
            x=roll_acc.index, y=roll_acc.values,
            mode="lines", name="90-day rolling accuracy",
            line=dict(color="#2ca02c", width=2),
        ))
        fig2.add_hline(
            y=0.5, line_dash="dot", line_color="gray",
            annotation_text="Random (50%)", annotation_position="left",
        )
        fig2.add_hline(
            y=meta["always_up_baseline"], line_dash="dash", line_color="red",
            annotation_text=f"Always-up baseline ({meta['always_up_baseline']*100:.1f}%)",
            annotation_position="right",
        )
        fig2.update_layout(
            height=350, yaxis_range=[0.3, 0.8], hovermode="x unified",
            xaxis_title="Date", yaxis_title="Accuracy (last 90 days of trades)",
        )
        st.plotly_chart(fig2, use_container_width=True)

        st.caption(
            "Tracks model accuracy through time. If the green line consistently sits below "
            "the red line (always-up baseline), the model is underperforming the trivial strategy."
        )

    st.divider()

    # Confusion summary
    st.subheader("Prediction Distribution")
    if pr.shape[0] > 0:
        c1, c2 = st.columns(2)
        with c1:
            fig3 = go.Figure()
            fig3.add_trace(go.Histogram(
                x=df["prob_up"], nbinsx=40,
                marker=dict(color="#9467bd"),
            ))
            fig3.add_vline(x=0.5, line_dash="dash", line_color="gray")
            fig3.update_layout(
                height=300, xaxis_title="P(WTI up in 5 days)",
                yaxis_title="Number of predictions",
                showlegend=False, title="Confidence distribution",
            )
            st.plotly_chart(fig3, use_container_width=True)

        with c2:
            recall_data = pd.DataFrame({
                "Class": ["UP", "DOWN"],
                "Recall": [meta.get("up_recall", 0), meta.get("down_recall", 0)],
            })
            fig4 = go.Figure(data=[go.Bar(
                x=recall_data["Class"], y=recall_data["Recall"],
                marker=dict(color=["#2ca02c", "#d62728"]),
                text=[f"{v:.1%}" for v in recall_data["Recall"]],
                textposition="auto",
            )])
            fig4.update_layout(
                height=300, yaxis_range=[0, 1],
                yaxis_title="Recall",
                title="Recall by class",
            )
            st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 2 — METHODOLOGY
# ══════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("How the Model Works")

    st.markdown(f"""
    ### Inputs ({meta.get('n_features', 'N/A')} features)

    **Price technicals (6):** RSI(14), MACD(12,26), 20-day volatility,
    5-day momentum, 20-day momentum, Bollinger Band position.

    **GPT-4o-mini sentiment (5):** Five dimensions extracted per article via
    AlphaVantage News API and structured GPT prompts:
    - Relevance, Polarity, Intensity, Uncertainty, Forwardness
    - Aggregated daily with relevance-weighted exponential decay (half-life 3 days)

    **OVX (3):** CBOE Crude Oil Volatility Index (raw, 5-day smoothed, 5-day change)

    **DXY (3):** US Dollar Index (raw, 5-day smoothed, 5-day change)

    ### Target

    Binary classification of WTI 5-day directional move:
    `1 if return > +0.3%, 0 if return < -0.3%, ignore otherwise`

    ### Method

    **LightGBM** binary classifier with two methodological refinements over
    Dai et al. (2026):

    1. **Optuna hyperparameter tuning** — {meta.get('n_optuna_trials', 50)} trials with
       Tree-structured Parzen Estimator, 5-fold time-series cross-validation.
       Tunes 8 parameters: learning rate, num leaves, max depth, min child samples,
       subsample, colsample_bytree, reg_alpha, reg_lambda.

    2. **5-model ensemble** — Trains 5 LightGBM models with different random seeds
       (42-46) using the same Optuna-tuned hyperparameters. Predictions averaged
       to reduce single-model variance.

    ### Walk-Forward Backtesting

    Strict temporal validation:
    - 50% initial training window
    - 30-day step size
    - 5-day purge gap (no future data leaks into training)
    - 20% validation set within training for early stopping
    - 0.2% round-trip transaction fee assumption
    - Trades only when ensemble confidence ≥ 52%
    """)

    if meta.get("best_optuna_params"):
        st.divider()
        st.subheader("Optuna-tuned hyperparameters")
        params_df = pd.DataFrame([
            {"Parameter": k, "Value": f"{v:.6f}" if isinstance(v, float) else str(v)}
            for k, v in meta["best_optuna_params"].items()
        ])
        st.dataframe(params_df, hide_index=True, use_container_width=True)

    st.divider()
    st.subheader("Comparison to published benchmark")
    benchmark_df = pd.DataFrame([
        {
            "Method": "Dai et al. (2026) — published",
            "Accuracy": f"{meta.get('dai_et_al_benchmark_accuracy', 0.5608) * 100:.2f}%",
            "AUC": f"{meta.get('dai_et_al_benchmark_auc', 0.65):.3f}",
            "Notes": "GPT-4o + FinBERT, 5-fold CV",
        },
        {
            "Method": "This work (Optuna + Ensemble)",
            "Accuracy": f"{meta['accuracy'] * 100:.2f}%",
            "AUC": f"{meta['auc']:.3f}",
            "Notes": f"Walk-forward, {meta.get('n_ensemble_models', 5)}-model ensemble",
        },
        {
            "Method": "Always-up baseline",
            "Accuracy": f"{meta['always_up_baseline'] * 100:.2f}%",
            "AUC": "—",
            "Notes": "Predict UP every day",
        },
    ])
    st.dataframe(benchmark_df, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 3 — FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Which Features Matter Most?")

    # Color-code by feature category
    fi = feat_imp.copy()

    def categorize(f):
        if f.startswith("ovx_"):
            return "OVX"
        elif f.startswith("dxy_"):
            return "DXY"
        elif f.startswith("sent_"):
            return "GPT Sentiment"
        elif f.startswith("fg_"):
            return "Fear & Greed"
        else:
            return "Price"

    fi["category"] = fi["feature"].apply(categorize)
    fi = fi.sort_values("importance", ascending=True)

    color_map = {
        "Price": "#1f77b4",
        "GPT Sentiment": "#ff7f0e",
        "OVX": "#2ca02c",
        "DXY": "#d62728",
        "Fear & Greed": "#9467bd",
    }

    fig = go.Figure()
    for cat in fi["category"].unique():
        sub = fi[fi["category"] == cat]
        fig.add_trace(go.Bar(
            x=sub["importance"], y=sub["feature"],
            orientation="h", name=cat,
            marker=dict(color=color_map.get(cat, "#888888")),
        ))

    fig.update_layout(
        height=max(400, 25 * len(fi)),
        xaxis_title="Average importance across folds",
        yaxis_title=None,
        margin=dict(l=140),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.caption(
        "Higher importance means the feature contributed more to splits "
        "across the LightGBM trees. Note: high importance does not always "
        "translate to generalization — features can rank highly but cause overfitting."
    )

    st.divider()
    st.subheader("Top features by category")
    summary = (
        fi.groupby("category")["importance"]
        .agg(["sum", "max", "count"])
        .reset_index()
        .rename(columns={
            "sum": "Total importance",
            "max": "Top feature score",
            "count": "# features",
        })
        .sort_values("Total importance", ascending=False)
    )
    summary["Total importance"] = summary["Total importance"].round(2)
    summary["Top feature score"] = summary["Top feature score"].round(2)
    st.dataframe(summary, hide_index=True, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════
# TAB 4 — HONEST FINDINGS
# ══════════════════════════════════════════════════════════════════════
with tab4:
    st.subheader("What This Project Honestly Found")

    st.markdown(f"""
    ### Summary

    Across {meta.get('n_features', 17)}-feature configurations combining price
    technicals, GPT-scored news sentiment, and macro indicators (OVX + DXY),
    the model achieves **{meta['accuracy']*100:.2f}% accuracy** and
    **AUC {meta['auc']:.3f}** on walk-forward backtesting.

    The always-up baseline is **{meta['always_up_baseline']*100:.2f}%**.
    Buy & hold over the same period returned **{meta['buy_hold_return']*100:+.1f}%**.

    The model's edge over the baseline is **{meta.get('edge_vs_baseline_pp', 0):+.2f} percentage points**
    — within the noise band (cross-fold std ≈ ±2pp).

    ### What this means

    - The methodology (Optuna + ensemble) closely matches the published frontier
      (Dai et al. 2026: 56.08% accuracy, AUC 0.65)
    - The edge is statistically modest. Cross-fold variance dominates.
    - After 0.2% round-trip fees, the strategy underperforms buy & hold by a wide margin.
    - This is consistent with broader literature: retail-accessible sentiment data
      provides limited tradeable edge for liquid commodities.

    ### Why this matters

    Most "AI predicts oil prices" claims fail honest replication. This project
    deliberately tests rigorous methodology — walk-forward validation, transaction
    cost accounting, ensemble averaging, hyperparameter tuning — and reports the
    honest result rather than cherry-picked best fold.

    The negative finding is itself the contribution.

    ### Limitations

    - **6 years of data** (2020–2026) is short for robust ML conclusions
    - **GPT-4o-mini sentiment** is mid-tier; FinBERT + Llama (Dai et al.)
      may add small marginal improvements
    - **Walk-forward variance is high** — single-fold results can swing ±5pp
    - **Transaction costs assumed at 0.2%** — real retail trading costs may be higher
    - **Survivor bias risk**: the 2020-2026 period had a strong oil rally, biasing
      "always-up" baseline upward
    - **Not real trading**: backtest returns assume executable close-to-close fills
    """)

    st.divider()
    st.subheader("Run metadata")
    st.json({
        "Configuration": meta.get("config", "Optuna + ensemble"),
        "Data window": f"{meta.get('data_start', '?')} → {meta.get('data_end', '?')}",
        "Features": meta.get("n_features"),
        "Optuna trials": meta.get("n_optuna_trials"),
        "Ensemble models": meta.get("n_ensemble_models"),
        "Trades / Predictions": f"{meta.get('n_trades')} / {meta.get('n_predictions')}",
        "Confidence threshold": meta.get("confidence_threshold", 0.52),
        "Round-trip fee": meta.get("round_trip_fee", 0.002),
    })


# ──────────────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Built with Python, LightGBM, Optuna, and Streamlit. "
    "Data: FRED (WTI, OVX), Yahoo Finance (DXY), AlphaVantage News API. "
    "**This is a research project. Not financial advice.**"
)
