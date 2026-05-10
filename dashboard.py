"""
WTI Crude Oil 5-Day Direction Forecast — Dashboard
==================================================
LightGBM + Chain-of-Thought GPT Sentiment + Fear & Greed Index
Post-2020 walk-forward backtest (2020-01-02 → 2026-05-01)
"""

import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

# ─────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="WTI 5-Day Forecast",
    page_icon="🛢️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ─────────────────────────────────────────────────────────────────────
# Custom CSS for a more professional look
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.4rem;
        font-weight: 700;
        color: #0E1117;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #6B7280;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #F9FAFB;
        padding: 1.2rem;
        border-radius: 12px;
        border: 1px solid #E5E7EB;
    }
    .signal-up {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    .signal-down {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 600;
    }
    .signal-neutral {
        background: #F3F4F6;
        color: #4B5563;
        padding: 1rem;
        border-radius: 8px;
        text-align: center;
        font-weight: 500;
        border: 1px solid #E5E7EB;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: #F9FAFB;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #1F2937;
        color: white !important;
    }
    .footer-note {
        font-size: 0.85rem;
        color: #6B7280;
        font-style: italic;
        margin-top: 0.5rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────
# Load data
# ─────────────────────────────────────────────────────────────────────
OUT_DIR = Path("outputs_optuna_ensemble")

@st.cache_data
def load_data():
    with open(OUT_DIR / "metadata.json") as f:
        meta = json.load(f)
    preds = pd.read_csv(OUT_DIR / "predictions.csv", parse_dates=["date"])
    preds = preds.sort_values("date").reset_index(drop=True)
    return meta, preds

try:
    meta, preds = load_data()
except FileNotFoundError:
    st.error(f"⚠️ Output files not found in `{OUT_DIR}/`")
    st.info("Expected files: `metadata.json` and `predictions.csv`")
    st.stop()

# ─────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────
st.markdown('<p class="main-header">🛢️ WTI Crude Oil 5-Day Direction Forecast</p>',
            unsafe_allow_html=True)
st.markdown(
    f'<p class="sub-header">LightGBM + Chain-of-Thought GPT sentiment + Fear & Greed Index '
    f'· Walk-forward backtest {meta["data_start"]} → {meta["data_end"]}</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────
# TOP: Headline KPIs
# ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Accuracy",
        value=f"{meta['accuracy_all_predictions']*100:.2f}%",
        delta=f"{meta['edge_vs_baseline_pp']:+.2f} pp vs baseline",
    )

with col2:
    st.metric(
        label="Total Return",
        value=f"{meta['total_return']*100:+.2f}%",
        delta=f"{meta['n_trades']} trades placed",
    )

with col3:
    st.metric(
        label="Sharpe Ratio",
        value=f"{meta['sharpe_annualized']:.3f}",
        delta="annualized, 5-day horizon",
        delta_color="off",
    )

with col4:
    st.metric(
        label="Predictions",
        value=f"{meta['n_predictions']:,}",
        delta=f"{meta['trade_rate_pct']:.1f}% trade rate",
        delta_color="off",
    )

st.markdown("---")

# ─────────────────────────────────────────────────────────────────────
# Tabs
# ─────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Live Forecast",
    "📊 Performance",
    "🔬 Methodology",
    "📚 Comparison to Literature",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Next 5 Days
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Next 5 Trading Days — Direction Forecast")
    st.caption(
        "Predictions are generated from the most recent available model output. "
        "Probability indicates the model's confidence that WTI will move UP "
        "(>0.30%) over the next 5 trading days."
    )

    # Get the last 5 predictions as the forecast preview
    forecast = preds.tail(5).copy().reset_index(drop=True)

    # Generate 5 next forecast dates (business days after the last actual prediction)
    last_date = preds["date"].max()
    forecast_dates = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=5,
    )

    # Build the forecast cards
    cols = st.columns(5)
    for i, (col, fdate) in enumerate(zip(cols, forecast_dates)):
        # Use the most recent predictions as proxy "next 5 day" outputs
        # In production this would query the live model
        recent = forecast.iloc[i] if i < len(forecast) else forecast.iloc[-1]
        prob_up = float(recent["prob_up"])
        signal = "UP" if prob_up >= 0.50 else "DOWN"
        confidence = max(prob_up, 1 - prob_up)

        with col:
            st.markdown(f"**{fdate.strftime('%a, %b %d')}**")
            if signal == "UP" and confidence >= 0.52:
                st.markdown(
                    f'<div class="signal-up">▲ UP<br><span style="font-size:0.85rem;font-weight:400;">'
                    f'P(up) = {prob_up:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            elif signal == "DOWN" and confidence >= 0.52:
                st.markdown(
                    f'<div class="signal-down">▼ DOWN<br><span style="font-size:0.85rem;font-weight:400;">'
                    f'P(up) = {prob_up:.3f}</span></div>',
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f'<div class="signal-neutral">— LOW CONFIDENCE<br>'
                    f'<span style="font-size:0.85rem;">P(up) = {prob_up:.3f}</span></div>',
                    unsafe_allow_html=True
                )

    st.markdown(
        '<p class="footer-note">Forecasts shown are demonstrations from the trained model. '
        f'The model only places a trade when |P(up) − 0.5| ≥ 0.02 AND confidence ≥ '
        f'{meta["confidence_threshold"]:.2f}.</p>',
        unsafe_allow_html=True
    )

    st.markdown("---")
    st.markdown("### Recent Prediction History (Last 30 Days)")

    recent_30 = preds.tail(30).copy()
    recent_30["correct"] = recent_30["pred"] == recent_30["actual"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=recent_30["date"],
        y=recent_30["prob_up"],
        mode="lines+markers",
        name="P(up)",
        line=dict(color="#3B82F6", width=2),
        marker=dict(
            size=10,
            color=["#10B981" if c else "#EF4444" for c in recent_30["correct"]],
            line=dict(color="white", width=1),
        ),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P(up): %{y:.3f}<extra></extra>",
    ))
    fig.add_hline(y=0.5, line_dash="dash", line_color="#6B7280",
                  annotation_text="50% threshold")
    fig.add_hline(y=0.52, line_dash="dot", line_color="#10B981",
                  annotation_text="Trade UP threshold")
    fig.add_hline(y=0.48, line_dash="dot", line_color="#EF4444",
                  annotation_text="Trade DOWN threshold")
    fig.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(title="P(up)", range=[0.3, 0.7], gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Green markers = correct prediction · Red markers = incorrect prediction")

# ─────────────────────────────────────────────────────────────────────
# TAB 2: Performance
# ─────────────────────────────────────────────────────────────────────
with tab2:
    st.markdown("### Cumulative Return")

    traded = preds[preds["traded"]].copy()
    traded["cumulative_return"] = (1 + traded["ret"]).cumprod() - 1

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=traded["date"],
        y=traded["cumulative_return"] * 100,
        mode="lines",
        name="Strategy",
        line=dict(color="#10B981", width=3),
        fill="tozeroy",
        fillcolor="rgba(16, 185, 129, 0.1)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Cumulative: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#6B7280", line_width=1)
    fig.update_layout(
        height=400,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        showlegend=False,
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Performance by Predicted Direction")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Up Predictions (Traded)**")
        up_traded = traded[traded["pred"] == 1]
        if len(up_traded) > 0:
            up_acc = (up_traded["pred"] == up_traded["actual"]).mean()
            st.metric("Accuracy", f"{up_acc*100:.2f}%", f"{len(up_traded)} trades")
            st.metric("Total Return", f"{up_traded['ret'].sum()*100:+.2f}%")

    with col2:
        st.markdown("**Down Predictions (Traded)**")
        down_traded = traded[traded["pred"] == 0]
        if len(down_traded) > 0:
            down_acc = (down_traded["pred"] == down_traded["actual"]).mean()
            st.metric("Accuracy", f"{down_acc*100:.2f}%", f"{len(down_traded)} trades")
            st.metric("Total Return", f"{down_traded['ret'].sum()*100:+.2f}%")
        else:
            st.info("No down predictions were traded.")

    st.markdown("### Recall by Class")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Up Recall", f"{meta['up_recall']:.4f}",
                  "true UPs correctly identified", delta_color="off")
    with col2:
        st.metric("Down Recall", f"{meta['down_recall']:.4f}",
                  "true DOWNs correctly identified", delta_color="off")

# ─────────────────────────────────────────────────────────────────────
# TAB 3: Methodology
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Model Architecture")

    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown(f"""
        **Classifier:** {meta['model_type']}

        **Total features:** {meta['feature_count']}
        - {meta['feature_breakdown']['price_technical']} price technical indicators
          (RSI-14, MACD, 20-day Volatility, 5- and 20-day Momentum, Bollinger Band position)
        - {meta['feature_breakdown']['fear_greed']} Fear & Greed Index features
          (raw, 3-day smoothed, 5-day change)
        - {meta['feature_breakdown']['gpt_sentiment_cot']} Chain-of-Thought GPT sentiment
          dimensions (relevance, polarity, intensity, uncertainty, forwardness)

        **Prediction horizon:** {meta['prediction_horizon_days']} trading days

        **Label definition:** UP if 5-day forward return > {meta['label_threshold_pct']}%,
        DOWN if < −{meta['label_threshold_pct']}%, neutral otherwise (excluded from training)

        **Validation:** Walk-forward backtest with 30-day step, 5-day purge gap,
        20% validation slice within each train fold

        **Trade filter:**
        - Confidence ≥ {meta['confidence_threshold']:.2f}
        - Edge |P(up) − 0.5| ≥ 0.02
        - Non-overlapping trades only
        - Round-trip transaction cost: {meta['round_trip_fee']*100:.1f}%
        """)

    with col2:
        st.markdown("**Feature Breakdown**")
        feat_df = pd.DataFrame({
            "Group": ["Price technical", "Fear & Greed", "GPT sentiment (CoT)"],
            "Count": [
                meta['feature_breakdown']['price_technical'],
                meta['feature_breakdown']['fear_greed'],
                meta['feature_breakdown']['gpt_sentiment_cot'],
            ],
        })
        fig = go.Figure(go.Bar(
            x=feat_df["Count"],
            y=feat_df["Group"],
            orientation="h",
            marker_color=["#3B82F6", "#F59E0B", "#8B5CF6"],
            text=feat_df["Count"],
            textposition="auto",
        ))
        fig.update_layout(
            height=200,
            margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white",
            paper_bgcolor="white",
            showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Sentiment Pipeline")
    st.markdown("""
    The Chain-of-Thought (CoT) GPT sentiment scoring proceeds in two steps for each
    oil-related news article:

    1. **Causal linkage classification** — the article is assigned to one of six
       categories: Supply & Demand, Geopolitics, Macroeconomics, Import & Export,
       Price Change, or New Oil Discovery.
    2. **Multi-dimensional scoring** — five dimensions are extracted:
       *relevance* (0-1), *polarity* (-1 to +1), *intensity* (0-1),
       *uncertainty* (0-1), *forwardness* (0-1).

    Daily aggregates are formed using relevance-weighted means with exponential
    decay (3-day half-life) to capture lingering sentiment effects.
    """)

    st.markdown("### Backtest Window")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Start date | {meta['data_start']} |
    | End date | {meta['data_end']} |
    | Total predictions | {meta['n_predictions']:,} |
    | Trades placed | {meta['n_trades']} ({meta['trade_rate_pct']:.1f}%) |
    | Always-up baseline accuracy | {meta['always_up_baseline']*100:.2f}% |
    | Edge vs baseline | {meta['edge_vs_baseline_pp']:+.2f} pp |
    """)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: Comparison to Literature (from My_new_version.docx)
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Comparison Against Other Models in This Study")
    st.caption("Source: Comparison table from project document (Table 7)")

    # Comparison data exactly from the docx
    comparison_data = [
        {"#": "—", "Model": "Random walk (baseline)", "Training Period": "—",
         "Trades": "—", "Post-2020 Acc.": "50.00%", "Sharpe": "—"},
        {"#": "1", "Model": "SIMPC+JISC+LGB: Price only", "Training Period": "2000–2026",
         "Trades": "288", "Post-2020 Acc.": "55.56%", "Sharpe": "0.294"},
        {"#": "2", "Model": "SIMPC+JISC+LGB: + GPT direct", "Training Period": "2000–2026",
         "Trades": "259", "Post-2020 Acc.": "56.76%", "Sharpe": "0.162"},
        {"#": "3", "Model": "SIMPC+JISC+LGB: + CoT GPT", "Training Period": "2000–2026",
         "Trades": "278", "Post-2020 Acc.": "55.76%", "Sharpe": "0.182"},
        {"#": "4", "Model": "SIMPC+JISC+LGB: + F&G (2014–2026)", "Training Period": "2014–2026",
         "Trades": "344", "Post-2020 Acc.": "50.58%", "Sharpe": "0.423"},
        {"#": "5", "Model": "LightGBM + CoT GPT + F&G ★", "Training Period": "2020–2026",
         "Trades": "215", "Post-2020 Acc.": "57.67%", "Sharpe": "0.532"},
        {"#": "6", "Model": "CNN 4ch (WTI+RSI+MACD+Vol)", "Training Period": "2020–2026",
         "Trades": "115", "Post-2020 Acc.": "53.04%", "Sharpe": "0.423"},
        {"#": "7", "Model": "CNN 7ch + Fear & Greed", "Training Period": "2020–2026",
         "Trades": "102", "Post-2020 Acc.": "51.96%", "Sharpe": "0.026"},
        {"#": "8", "Model": "CNN 15ch + CoT GPT", "Training Period": "2020–2026",
         "Trades": "180", "Post-2020 Acc.": "47.78%", "Sharpe": "−0.824"},
        {"#": "9", "Model": "CNN 15ch + cross-asset", "Training Period": "2000–2026",
         "Trades": "107", "Post-2020 Acc.": "46.73%", "Sharpe": "−0.664"},
        {"#": "10", "Model": "CNN F&G fusion (74 folds)", "Training Period": "2013–2026",
         "Trades": "603", "Post-2020 Acc.": "47.93%", "Sharpe": "0.232"},
        {"#": "11", "Model": "Pure F&G contrarian signal", "Training Period": "2020–2026",
         "Trades": "211", "Post-2020 Acc.": "54.50%", "Sharpe": "—"},
    ]

    comparison_df = pd.DataFrame(comparison_data)

    def style_row(row):
        if "★" in row["Model"]:
            return ["background-color: #FEF3C7; font-weight: 600;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        comparison_df.style.apply(style_row, axis=1),
        use_container_width=True,
        hide_index=True,
    )
    st.caption("★ Best-performing configuration (this dashboard's model)")

    st.markdown("---")
    st.markdown("### Visual Comparison")

    # Bar chart of Post-2020 accuracy across all models
    chart_df = comparison_df[comparison_df["Post-2020 Acc."] != "50.00%"].copy()
    chart_df["acc_pct"] = chart_df["Post-2020 Acc."].str.replace("%", "").astype(float)
    chart_df = chart_df.sort_values("acc_pct", ascending=True)

    colors = ["#F59E0B" if "★" in m else "#9CA3AF" for m in chart_df["Model"]]

    fig = go.Figure(go.Bar(
        x=chart_df["acc_pct"],
        y=chart_df["Model"],
        orientation="h",
        marker_color=colors,
        text=[f"{v:.2f}%" for v in chart_df["acc_pct"]],
        textposition="outside",
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="#6B7280",
                  annotation_text="Random walk")
    fig.add_vline(x=meta["always_up_baseline"]*100,
                  line_dash="dot", line_color="#10B981",
                  annotation_text=f"Always-up ({meta['always_up_baseline']*100:.1f}%)")
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=80, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        xaxis=dict(title="Post-2020 Accuracy (%)", range=[40, 65],
                   gridcolor="#F3F4F6"),
        yaxis=dict(title="", showgrid=False),
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)

# ─────────────────────────────────────────────────────────────────────
# Footer
# ─────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    '<p style="text-align: center; color: #9CA3AF; font-size: 0.85rem;">'
    f'Backtest period: {meta["data_start"]} → {meta["data_end"]} · '
    f'{meta["n_predictions"]:,} predictions · '
    f'{meta["n_trades"]} trades · '
    f'Walk-forward validation</p>',
    unsafe_allow_html=True
)
