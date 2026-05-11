"""
WTI Crude Oil 5-Day Direction Forecast — Decision Support Dashboard
====================================================================
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
        margin-bottom: 2.2rem;
        font-weight: 400;
    }

    /* Today's signal card — the centerpiece */
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

    /* Risk metric cards */
    .risk-card {
        background: #FAFAFA;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    .risk-label {
        font-size: 0.8rem;
        color: #6B7280;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 500;
        margin-bottom: 0.3rem;
    }
    .risk-value {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0E1117;
    }
    .risk-value-red { color: #DC2626; }
    .risk-value-green { color: #059669; }

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
        font-size: 1.6rem;
        font-weight: 700;
        color: #0E1117;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
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

# ─────────────────────────────────────────────────────────────────────
# Compute risk metrics from predictions
# ─────────────────────────────────────────────────────────────────────
@st.cache_data
def compute_risk_metrics(preds_df):
    traded = preds_df[preds_df["traded"]].copy().sort_values("date").reset_index(drop=True)
    if len(traded) == 0:
        return None

    # Equity curve
    traded["cum_ret"] = (1 + traded["ret"]).cumprod() - 1
    final_return = traded["cum_ret"].iloc[-1]

    # Max drawdown
    equity = (1 + traded["ret"]).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max
    max_dd = drawdown.min()
    dd_idx = drawdown.idxmin()
    dd_date = traded.loc[dd_idx, "date"]

    # Win / loss stats
    wins = (traded["ret"] > 0).sum()
    losses = (traded["ret"] < 0).sum()
    win_rate = wins / (wins + losses) if (wins + losses) > 0 else 0

    largest_win = traded["ret"].max()
    largest_loss = traded["ret"].min()
    avg_win = traded[traded["ret"] > 0]["ret"].mean() if wins > 0 else 0
    avg_loss = traded[traded["ret"] < 0]["ret"].mean() if losses > 0 else 0

    # VaR 95
    var_95 = np.percentile(traded["ret"], 5)

    # Sharpe
    mean_ret = traded["ret"].mean()
    std_ret = traded["ret"].std()
    sharpe = (mean_ret / (std_ret + 1e-12)) * np.sqrt(252 / HORIZON_DAYS)

    return {
        "n_trades": len(traded),
        "final_return": final_return,
        "max_drawdown": max_dd,
        "dd_date": dd_date,
        "win_rate": win_rate,
        "n_wins": wins,
        "n_losses": losses,
        "largest_win": largest_win,
        "largest_loss": largest_loss,
        "avg_win": avg_win,
        "avg_loss": avg_loss,
        "var_95": var_95,
        "sharpe": sharpe,
        "equity_curve": traded[["date", "ret", "cum_ret"]],
    }

risk = compute_risk_metrics(preds)
edge_pp = (meta["accuracy_all_predictions"] - RANDOM_WALK_BASELINE) * 100

# ─────────────────────────────────────────────────────────────────────
# HEADER
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

# ═════════════════════════════════════════════════════════════════════
# SECTION 1: TODAY'S SIGNAL (DECISION SUPPORT)
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">📍 Today\'s Trading Signal</p>',
    unsafe_allow_html=True
)

# Generate today's signal from the most recent prediction
latest = preds.tail(1).iloc[0]
prob_up = float(latest["prob_up"])
confidence = max(prob_up, 1 - prob_up)
edge = abs(prob_up - 0.5)

# Determine signal type
if confidence >= CONF_THRESHOLD and edge >= EDGE_THRESHOLD:
    if prob_up >= 0.5:
        signal = "BUY"
        signal_class = "signal-hero-buy"
        signal_icon = "▲"
        action_text = f"Long WTI futures, hold for {HORIZON_DAYS} trading days"
    else:
        signal = "SELL"
        signal_class = "signal-hero-sell"
        signal_icon = "▼"
        action_text = f"Short WTI futures, hold for {HORIZON_DAYS} trading days"
else:
    signal = "HOLD"
    signal_class = "signal-hero-hold"
    signal_icon = "—"
    action_text = "No trade — model confidence below threshold"

# Suggested target/stop based on historical average win/loss
suggested_target = abs(risk["avg_win"]) * 100
suggested_stop = abs(risk["avg_loss"]) * 100

# Date for the signal (next business day after the latest data point)
signal_date = pd.bdate_range(start=latest["date"] + timedelta(days=1), periods=1)[0]

# Hero card
col_hero, col_details = st.columns([2, 1])

with col_hero:
    st.markdown(
        f'<div class="{signal_class}">'
        f'<div class="signal-label">Recommendation for {signal_date.strftime("%A, %d %b %Y")}</div>'
        f'<div class="signal-action">{signal_icon} {signal}</div>'
        f'<div class="signal-detail">{action_text}</div>'
        f'</div>',
        unsafe_allow_html=True
    )

with col_details:
    st.markdown("**Model output**")
    st.markdown(f"- **P(up)** = `{prob_up:.4f}`")
    st.markdown(f"- **Confidence** = `{confidence*100:.2f}%`  "
                f"({'✓ above' if confidence >= CONF_THRESHOLD else '✗ below'} "
                f"{CONF_THRESHOLD:.0%} threshold)")
    st.markdown(f"- **Edge** = `{edge:+.4f}`  "
                f"({'✓ above' if edge >= EDGE_THRESHOLD else '✗ below'} "
                f"{EDGE_THRESHOLD:.0%} threshold)")
    st.markdown(f"- **Horizon**: {HORIZON_DAYS} trading days")

# Position sizing calculator (if a trade is recommended)
if signal != "HOLD":
    st.markdown("##### 💼 Position Sizing Calculator")
    col1, col2, col3 = st.columns([1.5, 1, 1])
    with col1:
        portfolio_size = st.number_input(
            "Portfolio size (USD)",
            min_value=1000, max_value=10000000,
            value=10000, step=1000, format="%d",
            help="Total capital available for trading"
        )
    with col2:
        risk_pct = st.select_slider(
            "Risk per trade",
            options=[1.0, 1.5, 2.0, 2.5, 3.0, 5.0],
            value=2.0,
            format_func=lambda x: f"{x:.1f}%",
            help="Maximum portfolio % at risk per trade (Kelly-conservative: 1-2%)"
        )
    with col3:
        position_size = portfolio_size * (risk_pct / 100) / (abs(risk["avg_loss"]) + 1e-6)
        st.metric(
            "Suggested position",
            f"${position_size:,.0f}",
            f"~{(position_size/portfolio_size)*100:.1f}% of portfolio",
            delta_color="off",
        )

    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Suggested target", f"+{suggested_target:.2f}%",
                  "based on historical avg win", delta_color="off")
    with col2:
        st.metric("Suggested stop loss", f"−{suggested_stop:.2f}%",
                  "based on historical avg loss", delta_color="off")
    with col3:
        rr_ratio = suggested_target / suggested_stop if suggested_stop > 0 else 0
        st.metric("Risk/Reward ratio", f"1 : {rr_ratio:.2f}",
                  "target ÷ stop", delta_color="off")

    st.markdown(
        '<p class="footer-note">⚠️ Suggestions are based on historical model performance. '
        'Past performance does not guarantee future results. Always apply your own risk management.</p>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 2: HEADLINE KPIs (existing — kept)
# ═════════════════════════════════════════════════════════════════════
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Accuracy", f"{meta['accuracy_all_predictions']*100:.2f}%",
              f"{edge_pp:+.2f} pp vs random walk")
with col2:
    st.metric("Total Return", f"{risk['final_return']*100:+.2f}%",
              f"{risk['n_trades']} trades")
with col3:
    st.metric("Sharpe Ratio", f"{risk['sharpe']:.3f}",
              "annualized", delta_color="off")
with col4:
    st.metric("Win Rate", f"{risk['win_rate']*100:.2f}%",
              f"{risk['n_wins']} wins / {risk['n_losses']} losses",
              delta_color="off")

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 3: RISK METRICS (NEW — for decision support)
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">⚠️ Risk Metrics</p>',
    unsafe_allow_html=True
)
st.caption("Historical risk characteristics from the full backtest. "
           "Use these to size positions and set stops.")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Max Drawdown</div>'
        f'<div class="risk-value risk-value-red">{risk["max_drawdown"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'observed {risk["dd_date"].strftime("%b %Y")}</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">VaR 95%</div>'
        f'<div class="risk-value risk-value-red">{risk["var_95"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'5th percentile of trade returns</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col3:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Largest Loss</div>'
        f'<div class="risk-value risk-value-red">{risk["largest_loss"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'worst single trade observed</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col4:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Largest Win</div>'
        f'<div class="risk-value risk-value-green">+{risk["largest_win"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'best single trade observed</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("")
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Avg Win</div>'
        f'<div class="risk-value risk-value-green">+{risk["avg_win"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'across {risk["n_wins"]} winning trades</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col2:
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Avg Loss</div>'
        f'<div class="risk-value risk-value-red">{risk["avg_loss"]*100:.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'across {risk["n_losses"]} losing trades</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col3:
    win_loss_ratio = abs(risk["avg_win"] / risk["avg_loss"]) if risk["avg_loss"] != 0 else 0
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Win/Loss Ratio</div>'
        f'<div class="risk-value">{win_loss_ratio:.2f}</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'avg win ÷ avg loss</div>'
        f'</div>',
        unsafe_allow_html=True
    )
with col4:
    expectancy = risk["win_rate"] * risk["avg_win"] + (1 - risk["win_rate"]) * risk["avg_loss"]
    st.markdown(
        f'<div class="risk-card">'
        f'<div class="risk-label">Expectancy/Trade</div>'
        f'<div class="risk-value">{expectancy*100:+.2f}%</div>'
        f'<div style="font-size:0.8rem; color:#6B7280; margin-top:0.3rem;">'
        f'expected value per trade</div>'
        f'</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# SECTION 4: WTI PRICE WITH PREDICTION OUTCOMES (last 3 months — kept)
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">📈 WTI Price with Prediction Outcomes — Last 3 Months</p>',
    unsafe_allow_html=True
)
st.caption("Each dot represents one model prediction overlaid on the actual WTI price. "
           "🟢 Green = correct · 🔴 Red = incorrect.")

last_pred_date = preds["date"].max()
three_months_ago = last_pred_date - pd.DateOffset(months=3)
recent_preds = preds[preds["date"] >= three_months_ago].copy()

price_start = three_months_ago - timedelta(days=5)
price_end = max(last_pred_date, pd.Timestamp.now()) + timedelta(days=2)

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
        xaxis=dict(gridcolor="#F3F4F6"),
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
# SECTION 5: RECENT TRADES TABLE (NEW — for decision support)
# ═════════════════════════════════════════════════════════════════════
st.markdown(
    '<p class="section-header">📋 Recent Trade Signals & Outcomes</p>',
    unsafe_allow_html=True
)
st.caption("The most recent 10 actual trade signals from the model. "
           "Use this to assess the model's recent track record.")

recent_trades = preds[preds["traded"]].tail(10).copy().reset_index(drop=True)
recent_trades = recent_trades.iloc[::-1].reset_index(drop=True)  # newest first

display_rows = []
for _, r in recent_trades.iterrows():
    signal_str = "🟢 BUY" if r["pred"] == 1 else "🔴 SELL"
    confidence = max(r["prob_up"], 1 - r["prob_up"])
    outcome = "✓ Profit" if r["ret"] > 0 else "✗ Loss"
    display_rows.append({
        "Date": r["date"].strftime("%Y-%m-%d"),
        "Signal": signal_str,
        "Confidence": f"{confidence*100:.1f}%",
        "Actual move": "↑ Up" if r["actual"] == 1 else "↓ Down",
        "Outcome": outcome,
        "Return": f"{r['ret']*100:+.2f}%",
    })

st.dataframe(
    pd.DataFrame(display_rows),
    use_container_width=True,
    hide_index=True,
)

st.markdown("---")

# ═════════════════════════════════════════════════════════════════════
# Tabs
# ═════════════════════════════════════════════════════════════════════
tab1, tab2, tab3 = st.tabs([
    "📊 Performance",
    "🔬 Methodology",
    "📚 Comparison to Literature",
])

# ─────────────────────────────────────────────────────────────────────
# TAB 1: Performance
# ─────────────────────────────────────────────────────────────────────
with tab1:
    st.markdown("### Cumulative Return")

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

    # Mark max drawdown
    fig.add_vline(x=risk["dd_date"], line_dash="dot", line_color="#EF4444",
                  annotation_text=f"Max DD ({risk['max_drawdown']*100:.1f}%)",
                  annotation_position="top")

    fig.update_layout(
        height=420, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Cumulative Return (%)", gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Final return: {risk['final_return']*100:+.2f}% · "
               f"Max drawdown: {risk['max_drawdown']*100:.2f}% · "
               f"Sharpe: {risk['sharpe']:.3f}")

    # Drawdown chart
    st.markdown("### Drawdown Over Time")
    equity = (1 + eq["ret"]).cumprod()
    running_max = equity.cummax()
    drawdown = (equity - running_max) / running_max

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=eq["date"], y=drawdown * 100,
        mode="lines", line=dict(color="#DC2626", width=2),
        fill="tozeroy", fillcolor="rgba(220, 38, 38, 0.15)",
        hovertemplate="<b>%{x|%Y-%m-%d}</b><br>Drawdown: %{y:.2f}%<extra></extra>",
    ))
    fig.update_layout(
        height=300, margin=dict(l=20, r=20, t=20, b=20),
        plot_bgcolor="white", paper_bgcolor="white", showlegend=False,
        yaxis=dict(title="Drawdown (%)", gridcolor="#F3F4F6"),
        xaxis=dict(gridcolor="#F3F4F6"),
    )
    st.plotly_chart(fig, use_container_width=True)

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
    | Total predictions | {meta['n_predictions']:,} |
    | Trades placed | {meta['n_trades']} ({meta['trade_rate_pct']:.1f}%) |
    | Random walk baseline | 50.00% |
    | Edge vs random walk | {edge_pp:+.2f} pp |
    """)

# ─────────────────────────────────────────────────────────────────────
# TAB 3: Comparison
# ─────────────────────────────────────────────────────────────────────
with tab3:
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
    fig.add_vline(x=50, line_dash="dash", line_color="#6B7280",
                  annotation_text="Random walk (50%)", annotation_position="top")
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
    f'{meta["n_predictions"]:,} predictions · '
    f'{meta["n_trades"]} trades · '
    f'Walk-forward validation</p>',
    unsafe_allow_html=True
)
