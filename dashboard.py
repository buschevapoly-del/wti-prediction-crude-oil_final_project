"""
WTI Crude Oil 5-Day Direction Forecast — Dashboard
==================================================
LightGBM + Chain-of-Thought GPT Sentiment + Fear & Greed Index
"""

import json
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

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
# CSS — bigger header
# ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 4.8rem;
        font-weight: 800;
        color: #0E1117;
        margin-bottom: 0.4rem;
        line-height: 1.05;
        letter-spacing: -0.025em;
    }
    .sub-header {
        font-size: 1.15rem;
        color: #6B7280;
        margin-bottom: 2.2rem;
        font-weight: 400;
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
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #0E1117;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
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

@st.cache_data(ttl=3600)  # cache 1 hour
def fetch_wti_prices(start_date, end_date):
    """Fetch WTI prices via yfinance. Returns DataFrame with date and Close."""
    try:
        import yfinance as yf
        ticker = yf.Ticker("CL=F")
        hist = ticker.history(start=start_date, end=end_date, auto_adjust=False)
        if hist.empty:
            return None
        df = hist[["Close"]].reset_index()
        df.columns = ["date", "Close"]
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        return df
    except Exception:
        return None

try:
    meta, preds = load_data()
except FileNotFoundError:
    st.error(f"⚠️ Output files not found in `{OUT_DIR}/`")
    st.info("Expected files: `metadata.json` and `predictions.csv`")
    st.stop()

# Random walk 50% baseline used everywhere
RANDOM_WALK_BASELINE = 0.50
edge_pp = (meta['accuracy_all_predictions'] - RANDOM_WALK_BASELINE) * 100
edge_traded_pp = (meta['accuracy_traded'] - RANDOM_WALK_BASELINE) * 100

# ─────────────────────────────────────────────────────────────────────
# HEADER (bigger title)
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="main-header">🛢️ WTI Crude Oil 5-Day Direction Forecast</p>',
    unsafe_allow_html=True
)
st.markdown(
    f'<p class="sub-header">LightGBM with Chain-of-Thought GPT sentiment + Fear & Greed Index '
    f'· Walk-forward backtest {meta["data_start"]} → {meta["data_end"]}</p>',
    unsafe_allow_html=True
)

# ─────────────────────────────────────────────────────────────────────
# TOP: Headline KPIs (against random walk 50%)
# ─────────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Accuracy",
        value=f"{meta['accuracy_all_predictions']*100:.2f}%",
        delta=f"{edge_pp:+.2f} pp vs random walk",
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
# MAIN CHART: WTI Price with prediction outcomes (LAST 3 MONTHS)
# ─────────────────────────────────────────────────────────────────────
st.markdown(
    '<p class="section-header">📈 WTI Price with Prediction Outcomes — Last 3 Months</p>',
    unsafe_allow_html=True
)
st.caption(
    "Each dot represents one model prediction overlaid on the actual WTI price. "
    "🟢 Green = correctly predicted direction · 🔴 Red = incorrect prediction."
)

# Filter predictions to last 3 months from latest prediction date
last_pred_date = preds["date"].max()
three_months_ago = last_pred_date - pd.DateOffset(months=3)
recent_preds = preds[preds["date"] >= three_months_ago].copy()

# Fetch WTI prices for the same window (with small buffer at the end through "today")
price_start = three_months_ago - timedelta(days=5)
price_end = max(last_pred_date, pd.Timestamp.now()) + timedelta(days=2)

with st.spinner("Loading WTI price data..."):
    wti_prices = fetch_wti_prices(price_start, price_end)

if wti_prices is None or wti_prices.empty:
    st.warning(
        "⚠️ Could not fetch WTI price data from yfinance right now. "
        "Showing prediction outcomes without the price line."
    )

    # Fallback: predictions vs probability without price line
    plot_df = recent_preds.copy()
    plot_df["correct"] = plot_df["pred"] == plot_df["actual"]

    fig = go.Figure()
    correct = plot_df[plot_df["correct"]]
    incorrect = plot_df[~plot_df["correct"]]
    fig.add_trace(go.Scatter(
        x=correct["date"], y=correct["prob_up"],
        mode="markers", name="Correct",
        marker=dict(size=10, color="#10B981", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P(up): %{y:.3f}<br>✓ Correct<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=incorrect["date"], y=incorrect["prob_up"],
        mode="markers", name="Incorrect",
        marker=dict(size=10, color="#EF4444", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>P(up): %{y:.3f}<br>✗ Incorrect<extra></extra>",
    ))
    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        yaxis=dict(title="P(up)", gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

else:
    # Merge predictions with prices
    plot_df = recent_preds.copy()
    plot_df["date_only"] = plot_df["date"].dt.normalize()
    wti_prices["date_only"] = wti_prices["date"].dt.normalize()
    plot_df = plot_df.merge(
        wti_prices[["date_only", "Close"]],
        on="date_only",
        how="left"
    )
    plot_df["correct"] = plot_df["pred"] == plot_df["actual"]
    plot_df = plot_df.dropna(subset=["Close"])

    n_correct = int(plot_df["correct"].sum())
    n_incorrect = int((~plot_df["correct"]).sum())
    total_recent = len(plot_df)
    pct_correct = n_correct / total_recent * 100 if total_recent > 0 else 0

    # Build the chart
    fig = go.Figure()

    # WTI price line (background)
    fig.add_trace(go.Scatter(
        x=wti_prices["date"],
        y=wti_prices["Close"],
        mode="lines",
        name="WTI Price",
        line=dict(color="#374151", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>WTI: $%{y:.2f}<extra></extra>",
    ))

    # Correct predictions (green dots)
    correct_df = plot_df[plot_df["correct"]]
    fig.add_trace(go.Scatter(
        x=correct_df["date"],
        y=correct_df["Close"],
        mode="markers",
        name=f"Correct ({n_correct})",
        marker=dict(
            size=10,
            color="#10B981",
            line=dict(color="white", width=1.5),
            symbol="circle",
        ),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "WTI: $%{y:.2f}<br>"
            "<b style='color:#10B981'>✓ Correct prediction</b><extra></extra>"
        ),
    ))

    # Incorrect predictions (red dots)
    incorrect_df = plot_df[~plot_df["correct"]]
    fig.add_trace(go.Scatter(
        x=incorrect_df["date"],
        y=incorrect_df["Close"],
        mode="markers",
        name=f"Incorrect ({n_incorrect})",
        marker=dict(
            size=10,
            color="#EF4444",
            line=dict(color="white", width=1.5),
            symbol="circle",
        ),
        hovertemplate=(
            "<b>%{x|%Y-%m-%d}</b><br>"
            "WTI: $%{y:.2f}<br>"
            "<b style='color:#EF4444'>✗ Incorrect prediction</b><extra></extra>"
        ),
    ))

    fig.update_layout(
        height=520,
        margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white",
        paper_bgcolor="white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            bgcolor="rgba(255,255,255,0.9)",
            font=dict(size=12),
        ),
        yaxis=dict(
            title="WTI Price (USD/barrel)",
            gridcolor="#F3F4F6",
        ),
        xaxis=dict(
            title="",
            gridcolor="#F3F4F6",
        ),
        hovermode="closest",
    )

    st.plotly_chart(fig, use_container_width=True)

    # 3 metric cards summarizing the 3-month view
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct (last 3 months)", f"{n_correct}", "")
    with col2:
        st.metric("Incorrect (last 3 months)", f"{n_incorrect}", "")
    with col3:
        edge_3m = pct_correct - 50
        st.metric(
            "3-month accuracy",
            f"{pct_correct:.2f}%",
            f"{edge_3m:+.2f} pp vs random walk"
        )

    st.caption(
        f"Window: {three_months_ago.strftime('%Y-%m-%d')} → "
        f"{last_pred_date.strftime('%Y-%m-%d')} · "
        f"Hover over any dot for details."
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

    forecast = preds.tail(5).copy().reset_index(drop=True)
    last_date = preds["date"].max()
    forecast_dates = pd.bdate_range(
        start=last_date + timedelta(days=1),
        periods=5,
    )

    cols = st.columns(5)
    for i, (col, fdate) in enumerate(zip(cols, forecast_dates)):
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
        '<p class="footer-note">The model only places a trade when '
        f'|P(up) − 0.5| ≥ 0.02 AND confidence ≥ '
        f'{meta["confidence_threshold"]:.2f}.</p>',
        unsafe_allow_html=True
    )

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
    | Random walk baseline | 50.00% |
    | Edge vs random walk | {edge_pp:+.2f} pp |
    """)

# ─────────────────────────────────────────────────────────────────────
# TAB 4: Comparison to Literature (from My_new_version.docx, baseline = 50%)
# ─────────────────────────────────────────────────────────────────────
with tab4:
    st.markdown("### Comparison Against Other Models in This Study")
    st.caption("Source: Comparison table from project document (Table 7)")

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
    # Single baseline reference line: random walk at 50%
    fig.add_vline(
        x=50, line_dash="dash", line_color="#6B7280",
        annotation_text="Random walk (50%)",
        annotation_position="top",
    )
    fig.update_layout(
        height=500,
        margin=dict(l=20, r=80, t=40, b=20),
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
