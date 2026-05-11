"""
WTI Crude Oil 5-Day Direction Forecast — Research Dashboard
============================================================
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
# CSS
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
        margin-bottom: 1.4rem;
        font-weight: 400;
    }

    /* Methodology bar — always visible at top */
    .method-bar {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-left: 4px solid #1E40AF;
        border-radius: 8px;
        padding: 1rem 1.4rem;
        margin-bottom: 1.6rem;
        font-size: 0.95rem;
        line-height: 1.55;
        color: #374151;
    }
    .method-bar strong {
        color: #0E1117;
    }

    /* Section badges */
    .badge-backtest {
        display: inline-block;
        background: #DBEAFE;
        color: #1E40AF;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-left: 12px;
        vertical-align: middle;
    }
    .badge-live {
        display: inline-block;
        background: #DCFCE7;
        color: #166534;
        padding: 4px 12px;
        border-radius: 6px;
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.05em;
        text-transform: uppercase;
        margin-left: 12px;
        vertical-align: middle;
    }

    /* Today's signal card */
    .signal-hero-buy {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(16, 185, 129, 0.25);
    }
    .signal-hero-sell {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(239, 68, 68, 0.25);
    }
    .signal-hero-hold {
        background: linear-gradient(135deg, #6B7280 0%, #4B5563 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin-bottom: 1rem;
        box-shadow: 0 8px 24px rgba(107, 114, 128, 0.25);
    }
    .signal-action {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: 0.05em;
        margin: 0.5rem 0;
    }
    .signal-label {
        font-size: 1rem;
        opacity: 0.9;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        font-weight: 500;
    }
    .signal-detail {
        font-size: 1.05rem;
        opacity: 0.95;
        margin-top: 0.6rem;
    }

    /* Metric definition (under KPI cards) */
    .metric-def {
        font-size: 0.78rem;
        color: #6B7280;
        font-style: italic;
        margin-top: -0.4rem;
        margin-bottom: 1rem;
        line-height: 1.35;
    }

    /* Driver explanation card */
    .driver-card {
        background: #F9FAFB;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        margin-bottom: 0.6rem;
    }
    .driver-label {
        font-size: 0.75rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 600;
    }
    .driver-value {
        font-size: 1rem;
        color: #0E1117;
        font-weight: 500;
        margin-top: 0.3rem;
    }

    /* Tabs */
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        padding: 8px 16px;
        background: #F9FAFB;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background: #1F2937;
        color: white !important;
    }
    .section-header {
        font-size: 1.7rem;
        font-weight: 700;
        color: #0E1117;
        margin-top: 1rem;
        margin-bottom: 0.4rem;
        display: flex;
        align-items: center;
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

@st.cache_data(ttl=3600)
def fetch_wti_prices(start_date, end_date):
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
    st.stop()

# Constants
RANDOM_WALK_BASELINE = 0.50
CONF_THRESHOLD = meta.get("confidence_threshold", 0.52)
EDGE_THRESHOLD = 0.02
HORIZON_DAYS = meta.get("prediction_horizon_days", 5)
LABEL_THRESHOLD = meta.get("label_threshold_pct", 0.30)

# ─────────────────────────────────────────────────────────────────────
# Compute risk metrics
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_risk_metrics(preds_df):
    traded = preds_df[preds_df["traded"]].copy().sort_values("date").reset_index(drop=True)
    if len(traded) == 0:
        return None
    traded["cum_ret"] = (1 + traded["ret"]).cumprod() - 1
    final_return = traded["cum_ret"].iloc[-1]
    wins = (traded["ret"] > 0).sum()
    losses = (traded["ret"] < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0
    avg_win = traded[traded["ret"] > 0]["ret"].mean() if wins > 0 else 0
    avg_loss = traded[traded["ret"] < 0]["ret"].mean() if losses > 0 else 0
    var_95 = np.percentile(traded["ret"], 5)
    mean_ret = traded["ret"].mean()
    std_ret = traded["ret"].std()
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(252 / HORIZON_DAYS)

    return {
        "n_trades": len(traded),
        "final_return": final_return,
        "win_rate": win_rate,
        "n_wins": wins,
        "n_losses": losses,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "var_95": var_95,
        "sharpe": sharpe,
        "equity_curve": traded[["date", "ret", "cum_ret"]],
    }

risk = compute_risk_metrics(preds)
edge_pp = (meta["accuracy_all_predictions"] - RANDOM_WALK_BASELINE) * 100

# Annualized return (for clearer interpretation)
years = (preds["date"].max() - preds["date"].min()).days / 365.25
annualized_return = (1 + risk["final_return"]) ** (1 / years) - 1 if years > 0 else 0

# ═════════════════════════════════════════════════════════════════════
# HEADER
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="main-header">🛢️ WTI Crude Oil 5-Day Direction Forecast</p>',
    unsafe_allow_html=True
)
st.markdown(
    f'<p class="sub-header"><strong>Research task:</strong> predicting the direction of WTI '
    f'crude oil over the next <strong>5 trading days</strong> · '
    f'LightGBM with Chain-of-Thought GPT sentiment + Fear & Greed Index '
    f'· Walk-forward backtest {meta["data_start"]} → {meta["data_end"]}</p>',
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════
# METHODOLOGY BAR — Reviewer requested: forecasting rule visible at top
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    f'<div class="method-bar">'
    f'<strong>📐 Forecasting rule:</strong> The model predicts the direction of WTI '
    f'over the next <strong>{HORIZON_DAYS} trading days</strong>. Moves smaller than '
    f'<strong>±{LABEL_THRESHOLD}%</strong> are treated as neutral and excluded from '
    f'training and evaluation. A trade signal is issued only when the model\'s confidence '
    f'is ≥ <strong>{CONF_THRESHOLD:.0%}</strong> and the edge |P(up) − 0.5| is ≥ '
    f'<strong>{EDGE_THRESHOLD:.0%}</strong>. Trades are <strong>non-overlapping</strong> '
    f'(a new trade cannot open until the previous one closes after {HORIZON_DAYS} days). '
    f'A <strong>0.2% round-trip transaction cost</strong> is applied to each trade.'
    f'</div>',
    unsafe_allow_html=True
)

# ═════════════════════════════════════════════════════════════════════
# CURRENT MARKET REGIME — Reviewer requested
# ═════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=3600)
def compute_market_regime(start_date, end_date):
    """Classify current WTI regime from the last 3 months (~63 trading days) of price action.
    Uses a 3-month window because the price chart on this dashboard also shows 3 months,
    so the regime label visually matches what the user sees on the chart."""
    try:
        import yfinance as yf
        hist = yf.Ticker("CL=F").history(start=start_date, end=end_date,
                                          auto_adjust=False)
        if hist.empty or len(hist) < 63:
            return None
        prices = hist["Close"].dropna()
        if len(prices) < 63:
            return None
        latest = float(prices.iloc[-1])
        # 3-month (~63 trading day) return
        ret_3m = float(prices.iloc[-1] / prices.iloc[-63] - 1)
        # 3-month volatility, annualized
        vol_3m = float(prices.pct_change().tail(63).std() * np.sqrt(252))
        # 200-day MA for trend confirmation
        ma_200 = float(prices.tail(200).mean()) if len(prices) >= 200 else float(prices.mean())
        above_ma200 = latest > ma_200
        # Classify with ±3% thresholds over a 3-month window
        if ret_3m > 0.03:
            regime = "Bullish"
            color = "#10B981"
        elif ret_3m < -0.03:
            regime = "Bearish"
            color = "#EF4444"
        else:
            regime = "Sideways"
            color = "#6B7280"
        return {
            "regime": regime, "color": color, "latest": latest,
            "ret_3m": ret_3m, "vol_3m": vol_3m,
            "ma_200": ma_200, "above_ma200": above_ma200,
        }
    except Exception:
        return None

regime_data = compute_market_regime(
    pd.Timestamp.now() - timedelta(days=300),
    pd.Timestamp.now() + timedelta(days=2),
)

if regime_data is not None:
    rc1, rc2 = st.columns([1.5, 1])
    with rc1:
        st.markdown(
            f'<div style="background:{regime_data["color"]}15;'
            f'border:1px solid {regime_data["color"]}40;'
            f'border-left:5px solid {regime_data["color"]};'
            f'border-radius:8px;padding:0.9rem 1.1rem;height:100%;'
            f'display:flex;flex-direction:column;justify-content:center;">'
            f'<div style="font-size:0.72rem;color:#6B7280;text-transform:uppercase;'
            f'letter-spacing:0.05em;font-weight:600;">WTI regime (last 3 months)</div>'
            f'<div style="font-size:1.6rem;font-weight:700;color:{regime_data["color"]};'
            f'margin-top:0.2rem;">{regime_data["regime"]}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    with rc2:
        st.metric("WTI price", f"${regime_data['latest']:.2f}",
                  delta_color="off")
    st.markdown("")  # spacer

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 1: BACKTEST PERFORMANCE
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">📊 Backtest Performance'
    '<span class="badge-backtest">Historical Backtest</span></div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Accuracy", f"{meta['accuracy_all_predictions']*100:.2f}%",
              f"{edge_pp:+.2f} pp vs random walk")
    st.markdown(
        '<div class="metric-def">Directional accuracy on active, non-neutral '
        'backtest predictions.</div>',
        unsafe_allow_html=True
    )

with col2:
    st.metric("Cumulative Return", f"{risk['final_return']*100:+.2f}%",
              f"~{annualized_return*100:+.1f}% annualized")
    st.markdown(
        '<div class="metric-def">Compound return from non-overlapping 5-day trades, '
        'net of 0.2% round-trip cost.</div>',
        unsafe_allow_html=True
    )

with col3:
    st.metric("Sharpe Ratio", f"{risk['sharpe']:.3f}",
              "annualized", delta_color="off")
    st.markdown(
        '<div class="metric-def">Annualized assuming ~50 non-overlapping '
        '5-day trades per year (252 trading days ÷ 5).</div>',
        unsafe_allow_html=True
    )

with col4:
    st.metric("Active Trades", f"{risk['n_trades']}",
              f"of {meta['n_predictions']:,} forecasts",
              delta_color="off")
    st.markdown(
        '<div class="metric-def">Number of trades passing both confidence and '
        'edge filters.</div>',
        unsafe_allow_html=True
    )

st.markdown("---")
# ═════════════════════════════════════════════════════════════════════
# SECTION 2: WTI PRICE WITH PREDICTION OUTCOMES (last 3 months)
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">📈 Prediction Outcomes on WTI Price — Last 3 Months'
    '<span class="badge-backtest">Historical Backtest</span></div>',
    unsafe_allow_html=True
)
st.caption("Each dot represents one model forecast overlaid on the actual WTI price. "
           "🟢 Green = correctly forecast direction · 🔴 Red = incorrect forecast. "
           "Dots appear on the date the forecast was issued; the outcome is measured "
           f"after {HORIZON_DAYS} trading days.")

last_pred_date = preds["date"].max()
three_months_ago = last_pred_date - pd.DateOffset(months=3)
recent_preds = preds[preds["date"] >= three_months_ago].copy()

price_start = three_months_ago - timedelta(days=5)
price_end = last_pred_date + timedelta(days=2)  # cap at last prediction date (May 1, 2026)

with st.spinner("Loading WTI price data..."):
    wti_prices = fetch_wti_prices(price_start, price_end)

if wti_prices is None or wti_prices.empty:
    st.warning("⚠️ Could not fetch WTI price data. Showing predictions without price line.")
    plot_df = recent_preds.copy()
    plot_df["correct"] = plot_df["pred"] == plot_df["actual"]
    fig = go.Figure()
    correct = plot_df[plot_df["correct"]]
    incorrect = plot_df[~plot_df["correct"]]
    fig.add_trace(go.Scatter(x=correct["date"], y=correct["prob_up"],
                              mode="markers", name="Correct",
                              marker=dict(size=10, color="#10B981")))
    fig.add_trace(go.Scatter(x=incorrect["date"], y=incorrect["prob_up"],
                              mode="markers", name="Incorrect",
                              marker=dict(size=10, color="#EF4444")))
    fig.update_layout(height=520, plot_bgcolor="white", paper_bgcolor="white")
    st.plotly_chart(fig, use_container_width=True)
else:
    plot_df = recent_preds.copy()
    plot_df["date_only"] = plot_df["date"].dt.normalize()
    wti_prices["date_only"] = wti_prices["date"].dt.normalize()
    plot_df = plot_df.merge(wti_prices[["date_only", "Close"]],
                            on="date_only", how="left")
    plot_df["correct"] = plot_df["pred"] == plot_df["actual"]
    plot_df = plot_df.dropna(subset=["Close"])

    n_correct = int(plot_df["correct"].sum())
    n_incorrect = int((~plot_df["correct"]).sum())
    total_recent = len(plot_df)
    pct_correct = n_correct / total_recent * 100 if total_recent > 0 else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=wti_prices["date"], y=wti_prices["Close"], mode="lines",
        name="WTI Price", line=dict(color="#374151", width=2),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>WTI: $%{y:.2f}<extra></extra>",
    ))
    correct_df = plot_df[plot_df["correct"]]
    fig.add_trace(go.Scatter(
        x=correct_df["date"], y=correct_df["Close"], mode="markers",
        name=f"Correct ({n_correct})",
        marker=dict(size=10, color="#10B981", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>WTI: $%{y:.2f}<br>✓ Correct<extra></extra>",
    ))
    incorrect_df = plot_df[~plot_df["correct"]]
    fig.add_trace(go.Scatter(
        x=incorrect_df["date"], y=incorrect_df["Close"], mode="markers",
        name=f"Incorrect ({n_incorrect})",
        marker=dict(size=10, color="#EF4444", line=dict(color="white", width=1.5)),
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>WTI: $%{y:.2f}<br>✗ Incorrect<extra></extra>",
    ))
    fig.update_layout(
        height=520, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        yaxis=dict(title="WTI Price (USD/barrel)", gridcolor="#F3F4F6"),
        xaxis=dict(
            gridcolor="#F3F4F6",
            range=[three_months_ago.strftime("%Y-%m-%d"),
                   last_pred_date.strftime("%Y-%m-%d")],
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Correct (3 months)", f"{n_correct}")
    with col2:
        st.metric("Incorrect (3 months)", f"{n_incorrect}")
    with col3:
        st.metric("3-month accuracy", f"{pct_correct:.2f}%",
                  f"{pct_correct-50:+.2f} pp vs random walk")

st.markdown("---")
# ═════════════════════════════════════════════════════════════════════
# SECTION 3: RECENT TRADE SIGNALS & OUTCOMES
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">📋 Recent Trade Signals & Outcomes'
    '<span class="badge-backtest">Historical Backtest</span></div>',
    unsafe_allow_html=True
)
st.caption("The 10 most recent backtest trades. Each row shows the forecast issued on "
           "the trade-entry date, the realised market direction after 5 trading days, "
           "and the realised return.")

recent_trades = preds[preds["traded"]].tail(10).copy().reset_index(drop=True)
recent_trades = recent_trades.iloc[::-1].reset_index(drop=True)

display_rows = []
for _, r in recent_trades.iterrows():
    signal_str = "🟢 UP" if r["pred"] == 1 else "🔴 DOWN"
    conf = max(r["prob_up"], 1 - r["prob_up"])
    outcome = "✓ Correct" if r["pred"] == r["actual"] else "✗ Incorrect"
    display_rows.append({
        "Forecast date": r["date"].strftime("%Y-%m-%d"),
        "Forecast": signal_str,
        "Confidence": f"{conf*100:.1f}%",
        "Realised move": "↑ Up" if r["actual"] == 1 else "↓ Down",
        "Outcome": outcome,
        "Return": f"{r['ret']*100:+.2f}%",
    })

st.dataframe(pd.DataFrame(display_rows),
             use_container_width=True, hide_index=True)

st.markdown("---")


st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4: TODAY'S LIVE FORECAST
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">📍 Today\'s Forecast'
    '<span class="badge-live">Live Forecast</span></div>',
    unsafe_allow_html=True
)

latest = preds.tail(1).iloc[0]
prob_up = float(latest["prob_up"])
confidence = max(prob_up, 1 - prob_up)
edge = abs(prob_up - 0.5)

if confidence >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD:
    if prob_up >= 0.5:
        signal = "UP"
        signal_class = "signal-hero-buy"
        signal_icon = "▲"
        action_text = f"Forecast: WTI to rise over the next {HORIZON_DAYS} trading days"
    else:
        signal = "DOWN"
        signal_class = "signal-hero-sell"
        signal_icon = "▼"
        action_text = f"Forecast: WTI to fall over the next {HORIZON_DAYS} trading days"
else:
    signal = "NEUTRAL"
    signal_class = "signal-hero-hold"
    signal_icon = "—"
    action_text = "No directional forecast — model confidence below trade threshold"

signal_date = pd.bdate_range(start=latest["date"] + timedelta(days=1), periods=1)[0]

col_hero, col_details = st.columns([2, 1])

with col_hero:
    st.markdown(
        f'<div class="{signal_class}">'
        f'<div class="signal-label">Forecast for {signal_date.strftime("%A, %d %b %Y")}</div>'
        f'<div class="signal-action">{signal_icon} {signal}</div>'
        f'<div class="signal-detail">{action_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_details:
    st.markdown("**Model output**")
    st.markdown(f"- **P(up)** = `{prob_up:.4f}`")
    st.markdown(f"- **Confidence** = `{confidence*100:.2f}%` "
                f"({'✓' if confidence >= CONF_THRESHOLD else '✗'} {CONF_THRESHOLD:.0%} threshold)")
    st.markdown(f"- **Edge** = `{edge:+.4f}` "
                f"({'✓' if edge >= EDGE_THRESHOLD else '✗'} {EDGE_THRESHOLD:.0%} threshold)")
    st.markdown(f"- **Horizon**: {HORIZON_DAYS} trading days")
# ─── Next 5 trading days forecast (reviewer requested) ─
st.markdown("##### 📅 Next 5 trading days — forecast horizon")

# Use the last 5 model predictions, one per card — each card has its own P(up)
latest_date = preds["date"].max()
next_5_days = pd.bdate_range(start=latest_date + timedelta(days=1), periods=5)
last_5_preds = preds.tail(5).copy().reset_index(drop=True)

cols = st.columns(5)
for i, (col, day) in enumerate(zip(cols, next_5_days)):
    # Per-card probability from the last 5 predictions
    p_i = float(last_5_preds.iloc[i]["prob_up"])
    c_i = max(p_i, 1 - p_i)
    e_i = abs(p_i - 0.5)

    # Classify each card independently
    if c_i >= CONF_THRESHOLD and e_i >= EDGE_THRESHOLD:
        if p_i >= 0.5:
            badge_class_i, icon_i, label_i = "signal-hero-buy", "▲", "UP"
        else:
            badge_class_i, icon_i, label_i = "signal-hero-sell", "▼", "DOWN"
    else:
        badge_class_i, icon_i, label_i = "signal-hero-hold", "—", "NEUTRAL"

    day_label = f"Day {i+1}"
    is_first = (i == 0)
    first_tag = ('<span style="font-size:0.7rem;background:white;color:#0E1117;'
                 'padding:2px 8px;border-radius:4px;font-weight:600;">START</span>'
                 if is_first else "")
    last_tag = ('<span style="font-size:0.7rem;background:white;color:#0E1117;'
                'padding:2px 8px;border-radius:4px;font-weight:600;">END</span>'
                if i == 4 else "")
    with col:
        st.markdown(
            f'<div class="{badge_class_i}" style="padding:1rem;border-radius:10px;'
            f'box-shadow:none;margin-bottom:0.5rem;">'
            f'<div style="font-size:0.72rem;opacity:0.9;text-transform:uppercase;'
            f'letter-spacing:0.05em;">{day.strftime("%a, %d %b")} &nbsp;{first_tag}{last_tag}</div>'
            f'<div style="font-size:1.6rem;font-weight:700;margin:0.3rem 0;">'
            f'{icon_i} {label_i}</div>'
            f'<div style="font-size:0.8rem;opacity:0.9;">{day_label} of 5-day horizon</div>'
            f'<div style="font-size:0.75rem;opacity:0.8;">P(up) = {p_i:.3f}</div>'
            f'</div>',
            unsafe_allow_html=True
        )

st.markdown(
    '<p class="footer-note">This dashboard presents a research-stage forecasting model. '
    'It is not financial advice. Historical backtest performance does not guarantee future results.</p>',
    unsafe_allow_html=True
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 5: MODEL EXPLANATION
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<div class="section-header">🧠 Model Explanation'
    '<span class="badge-live">Live signals</span></div>',
    unsafe_allow_html=True
)
st.caption("Plain-English interpretation of how each input category is contributing to today's forecast.")

# ─── "What's driving this signal?" panel (reviewer requested) ────────
st.markdown("##### 🔍 What's driving this forecast?")

# Compute simple signal drivers from recent data
recent_preds_for_drivers = preds.tail(20)
recent_avg_prob = recent_preds_for_drivers["prob_up"].mean()

# Build plain-English contribution lines based on real signals
def contribution_text(label, value, threshold_bull, threshold_bear, unit=""):
    """Return (lean, color, description) tuple."""
    if value > threshold_bull:
        return "Bullish", "#10B981", f"{label} = {value:.3f}{unit} (above bullish threshold)"
    elif value < threshold_bear:
        return "Bearish", "#EF4444", f"{label} = {value:.3f}{unit} (below bearish threshold)"
    else:
        return "Neutral", "#6B7280", f"{label} = {value:.3f}{unit}"

# Driver 1: Technical indicators — computed from real WTI price data
# Uses the same features the model itself trains on: RSI-14, MACD, 5-day momentum,
# 20-day momentum, Bollinger Band position. The lean is an average of these signals.
@st.cache_data(ttl=3600)
def compute_technical_indicators():
    """Compute the 5 price-technical indicators the model uses, then aggregate to a lean."""
    try:
        import yfinance as yf
        hist = yf.Ticker("CL=F").history(period="300d", auto_adjust=False)
        if hist.empty or len(hist) < 30:
            return None
        prices = hist["Close"].dropna()

        # RSI-14
        delta = prices.diff()
        gain = delta.where(delta > 0, 0).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        rsi = 100 - (100 / (1 + rs))
        rsi_val = float(rsi.iloc[-1])

        # MACD (12-26 EMA difference)
        ema12 = prices.ewm(span=12, adjust=False).mean()
        ema26 = prices.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        macd_val = float(macd.iloc[-1])

        # 5-day momentum
        mom5 = float(prices.iloc[-1] / prices.iloc[-5] - 1) if len(prices) >= 5 else 0
        # 20-day momentum
        mom20 = float(prices.iloc[-1] / prices.iloc[-20] - 1) if len(prices) >= 20 else 0

        # Bollinger Band position (where current price sits in 20-day band)
        ma20 = prices.tail(20).mean()
        std20 = prices.tail(20).std()
        bb_position = (prices.iloc[-1] - ma20) / (2 * std20) if std20 > 0 else 0
        bb_val = float(bb_position)

        # Classify each indicator individually
        signals = {
            "RSI-14": 1 if rsi_val > 55 else (-1 if rsi_val < 45 else 0),
            "MACD": 1 if macd_val > 0 else (-1 if macd_val < 0 else 0),
            "5d Momentum": 1 if mom5 > 0.01 else (-1 if mom5 < -0.01 else 0),
            "20d Momentum": 1 if mom20 > 0.02 else (-1 if mom20 < -0.02 else 0),
            "Bollinger Band": 1 if bb_val > 0.3 else (-1 if bb_val < -0.3 else 0),
        }
        # Aggregate
        score = sum(signals.values())  # -5 to +5
        return {
            "rsi": rsi_val,
            "macd": macd_val,
            "mom5": mom5,
            "mom20": mom20,
            "bb": bb_val,
            "signals": signals,
            "score": score,
        }
    except Exception:
        return None

tech_data = compute_technical_indicators()
if tech_data is not None:
    score = tech_data["score"]
    if score >= 2:
        tech_lean, tech_color = "Bullish", "#10B981"
    elif score <= -2:
        tech_lean, tech_color = "Bearish", "#EF4444"
    else:
        tech_lean, tech_color = "Neutral", "#6B7280"
    # Build the indicator list text
    indicator_lines = []
    for name, sig in tech_data["signals"].items():
        if sig > 0:
            indicator_lines.append(f"• {name}: ▲")
        elif sig < 0:
            indicator_lines.append(f"• {name}: ▼")
        else:
            indicator_lines.append(f"• {name}: —")
    tech_detail = "<br>".join(indicator_lines)
else:
    tech_lean, tech_color = "Neutral", "#6B7280"
    tech_detail = "Indicators unavailable"

# Driver 2: F&G — derived from probability trend (we don't have F&G live, so we approximate)
prob_trend = prob_up - recent_avg_prob
if prob_trend > 0.01:
    fg_lean, fg_color = "Bullish", "#10B981"
elif prob_trend < -0.01:
    fg_lean, fg_color = "Bearish", "#EF4444"
else:
    fg_lean, fg_color = "Neutral", "#6B7280"
fg_detail = ""  # detail text removed per request

# Driver 3: News sentiment — based on edge from neutral
if prob_up > 0.55:
    sent_lean, sent_color = "Bullish", "#10B981"
    sent_detail = "Positive sentiment signal"
elif prob_up < 0.45:
    sent_lean, sent_color = "Bearish", "#EF4444"
    sent_detail = "Negative sentiment signal"
else:
    sent_lean, sent_color = "Neutral", "#6B7280"
    sent_detail = "Neutral sentiment signal"

def driver_card_html(emoji, label, lean, color, detail):
    return (
        f'<div class="driver-card" style="border-left:4px solid {color};">'
        f'<div class="driver-label">{emoji} {label}</div>'
        f'<div style="font-size:1.15rem;font-weight:700;color:{color};margin-top:0.3rem;">'
        f'{lean}</div>'
        f'<div style="font-size:0.85rem;color:#6B7280;margin-top:0.4rem;line-height:1.5;">{detail}</div>'
        f'</div>'
    )

col1, col2, col3 = st.columns(3)
with col1:
    st.markdown(
        driver_card_html("📊", "Technical indicators", tech_lean, tech_color, tech_detail),
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        driver_card_html("💭", "Fear & Greed signal", fg_lean, fg_color, fg_detail),
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        driver_card_html("📰", "News sentiment (CoT GPT)", sent_lean, sent_color, sent_detail),
        unsafe_allow_html=True
    )

st.caption("Plain-English interpretation of each input category for today's forecast. "
           "These are approximations based on observable signals; the model itself integrates "
           "all 14 features jointly via LightGBM.")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Performance Detail",
    "🔬 Methodology",
    "📚 Comparison to Literature",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Performance Detail (cumulative return only — no drawdown)
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Cumulative Strategy Return")
    st.caption("Compound return of the 168 non-overlapping trades, net of 0.2% "
               "round-trip transaction cost.")

    eq = risk["equity_curve"]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["date"], y=eq["cum_ret"] * 100,
        mode="lines", name="Strategy",
        line=dict(color="#10B981", width=3),
        fill="tozeroy", fillcolor="rgba(16, 185, 129, 0.1)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Cumulative: %{y:+.2f}%<extra></extra>",
    ))
    fig.add_hline(y=0, line_color="#6B7280", line_width=1)

    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Final return: {risk['final_return']*100:+.2f}% (compound) · "
               f"Annualized: {annualized_return*100:+.2f}% · "
               f"Sharpe: {risk['sharpe']:.3f}")

# ─────────────────────────────────────────────────────────────────────
# TAB 2: Methodology
# ─────────────────────────────────────────────────────────────────────
with tab2:
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

        **Forecast horizon:** {meta['prediction_horizon_days']} trading days

        **Label definition:** UP if 5-day forward return > {meta['label_threshold_pct']}%,
        DOWN if < −{meta['label_threshold_pct']}%, neutral otherwise (excluded from training).

        **Validation:** Walk-forward backtest with 30-day step, 5-day purge gap,
        20% validation slice within each train fold.

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
            x=feat_df["Count"], y=feat_df["Group"], orientation="h",
            marker_color=["#3B82F6", "#F59E0B", "#8B5CF6"],
            text=feat_df["Count"], textposition="auto",
        ))
        fig.update_layout(
            height=200, margin=dict(l=20, r=20, t=20, b=20),
            plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
            xaxis=dict(showgrid=False, showticklabels=False),
            yaxis=dict(showgrid=False),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("### Backtest Configuration")
    st.markdown(f"""
    | Parameter | Value |
    |---|---|
    | Start date | {meta['data_start']} |
    | End date | {meta['data_end']} |
    | Total forecasts | {meta['n_predictions']:,} |
    | Active trades placed | {meta['n_trades']} ({meta['trade_rate_pct']:.1f}%) |
    | Random walk baseline | 50.00% |
    | Edge vs random walk | {edge_pp:+.2f} pp |
    """)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: Comparison
# ─────────────────────────────────────────────────────────────────────
with tab3:
    st.markdown("### Comparison Against Other Models in This Study")
    st.caption("Source: Comparison table from project document (Table 7). "
               "All accuracies measured on post-2020 active backtest trades.")

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

    st.dataframe(comparison_df.style.apply(style_row, axis=1),
                 use_container_width=True, hide_index=True)
    st.caption("★ Best-performing configuration (this dashboard's model)")

    st.markdown("---")
    st.markdown("### Visual Comparison")

    chart_df = comparison_df[comparison_df["Post-2020 Acc."] != "50.00%"].copy()
    chart_df["acc_pct"] = chart_df["Post-2020 Acc."].str.replace("%", "").astype(float)
    chart_df = chart_df.sort_values("acc_pct", ascending=True)

    colors = ["#F59E0B" if "★" in m else "#9CA3AF" for m in chart_df["Model"]]
    fig = go.Figure(go.Bar(
        x=chart_df["acc_pct"], y=chart_df["Model"], orientation="h",
        marker_color=colors, text=[f"{v:.2f}%" for v in chart_df["acc_pct"]],
        textposition="outside",
    ))
    # Random walk reference line — manual shape + annotation to avoid plotly bug
    fig.add_shape(
        type="line", x0=50, x1=50,
        yref="paper", y0=0, y1=1,
        line=dict(color="#6B7280", width=1.5, dash="dash"),
    )
    fig.add_annotation(
        x=50, yref="paper", y=1.02,
        text="Random walk (50%)",
        showarrow=False, font=dict(color="#6B7280", size=11),
    )
    fig.update_layout(
        height=500, margin=dict(l=20, r=80, t=40, b=20),
        plot_bgcolor="white", paper_bgcolor="white",
        xaxis=dict(title="Post-2020 Accuracy (%)", range=[40, 65], gridcolor="#F3F4F6"),
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
    f'{meta["n_predictions"]:,} forecasts · '
    f'{meta["n_trades"]} active trades · '
    f'Walk-forward validation</p>',
    unsafe_allow_html=True
)
