"""
==============================================================================
  wti_optuna_ensemble.py — Improving on Dai et al. (2026)
  -----------------------------------------------------------------------------
  Builds on the Dai et al. methodology with two principled improvements:

    1. Optuna hyperparameter tuning (Tree-structured Parzen Estimator)
       - 50 trials, optimizing AUC on time-series CV (5 splits)
       - Same metric Dai et al. used, just done more rigorously
       - Tunes: learning_rate, num_leaves, max_depth, min_child_samples,
                subsample, colsample_bytree, reg_alpha, reg_lambda

    2. 5-model ensemble averaging
       - Train 5 LightGBM models with seeds 42-46
       - Average their predicted probabilities
       - Reduces variance from any single model's idiosyncratic patterns

  Feature set (17 total):
    · 6 price technicals
    · 5 GPT sentiment dimensions (FROM CACHE — no new GPT calls)
    · 3 OVX features
    · 3 DXY features

  Setup:
    pip3 install lightgbm pandas numpy scikit-learn optuna requests yfinance
    # Cache files needed (from prior runs):
    #   cache/wti_news_cache.csv
    #   cache/wti_scores_cache.csv
    python3 wti_optuna_ensemble.py

  Cost: $0 (uses cached scores, free data sources)
  Runtime: ~30-60 minutes (Optuna takes most of it)
==============================================================================
"""
from __future__ import annotations

import json
import math
import warnings
from io import StringIO
from pathlib import Path

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import requests
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

# ──────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────
HORIZON              = 5
LABEL_THRESHOLD      = 0.003
RESTRICT_TO_N_YEARS  = 6
SENTIMENT_HALFLIFE   = 3

# Walk-forward (same as your prior runs for direct comparison)
INIT_TRAIN_RATIO     = 0.50
STEP_SIZE            = 30
PURGE_GAP            = 5
VAL_FRACTION         = 0.20
ROUND_TRIP_FEE       = 0.002
CONFIDENCE_THRESHOLD = 0.52
MIN_EDGE             = 0.02

# Optuna config
OPTUNA_N_TRIALS      = 50          # Search budget
OPTUNA_CV_SPLITS     = 5           # Time-series CV folds
OPTUNA_TIMEOUT_SEC   = 1800        # Max 30 min for Optuna

# Ensemble config
ENSEMBLE_SEEDS       = [42, 43, 44, 45, 46]

PRICE_FEATURES = ["feat_RSI", "feat_MACD", "feat_Vol20",
                  "feat_Mom5", "feat_Mom20", "feat_BB"]
SENT_DIMS      = ["relevance", "polarity", "intensity",
                  "uncertainty", "forwardness"]
OVX_FEATURES   = ["ovx_raw", "ovx_smooth5", "ovx_change5"]
DXY_FEATURES   = ["dxy_raw", "dxy_smooth5", "dxy_change5"]

CACHE_LOCATIONS = [
    Path("cache"),
    Path("/content"),
    Path.home() / "cache",
    Path("."),
]

OUT_DIR = Path("outputs_optuna_ensemble")
OUT_DIR.mkdir(exist_ok=True)


def find_cache_file(filename: str) -> Path:
    for loc in CACHE_LOCATIONS:
        candidate = loc / filename
        if candidate.exists():
            return candidate
    return None


# ──────────────────────────────────────────────────────────────────────
# Data fetchers (same as wti_pipeline_ovx_dxy.py)
# ──────────────────────────────────────────────────────────────────────
def fetch_fred(series_id: str, label: str) -> pd.Series:
    urls = [
        f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_id}",
        f"https://fred.stlouisfed.org/series/{series_id}/downloaddata/{series_id}.csv",
    ]
    for url in urls:
        try:
            r = requests.get(url, timeout=20,
                             headers={"User-Agent": "Mozilla/5.0"})
            if r.status_code != 200:
                continue
            df = pd.read_csv(StringIO(r.text))
            date_col = next((c for c in df.columns
                             if c.lower() in ("date", "observation_date")),
                            df.columns[0])
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            df = df.dropna(subset=[date_col]).set_index(date_col)
            val_col = next((c for c in df.columns
                            if c != date_col and c.lower() != "date"), None)
            if val_col is None and len(df.columns) > 0:
                val_col = df.columns[0]
            s = pd.to_numeric(df[val_col], errors="coerce").dropna()
            s.name = series_id
            print(f"  {label}: {len(s):,} rows from FRED")
            return s
        except Exception:
            continue
    return None


def fetch_via_yfinance(ticker: str, label: str) -> pd.Series:
    try:
        import yfinance as yf
        h = yf.Ticker(ticker).history(period="max")[["Close"]]
        if h.empty:
            return None
        h.index = pd.to_datetime(h.index).tz_localize(None)
        s = h["Close"]
        s.name = ticker
        print(f"  {label}: {len(s):,} rows from yfinance")
        return s
    except Exception as e:
        print(f"  {label}: yfinance failed: {e}")
        return None


def fetch_wti() -> pd.DataFrame:
    print("[1/6] Fetching WTI prices...")
    s = fetch_fred("DCOILWTICO", "WTI")
    if s is None or len(s) < 100:
        s = fetch_via_yfinance("CL=F", "WTI")
    if s is None:
        raise RuntimeError("Could not fetch WTI from any source")
    return pd.DataFrame({"WTI": s})


def add_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    p = out["WTI"]
    delta = p.diff()
    gain = delta.clip(lower=0).rolling(14, min_periods=14).mean()
    loss = (-delta.clip(upper=0)).rolling(14, min_periods=14).mean()
    out["feat_RSI"]   = 100 - (100 / (1 + gain / (loss + 1e-8)))
    ema12 = p.ewm(span=12, adjust=False, min_periods=12).mean()
    ema26 = p.ewm(span=26, adjust=False, min_periods=26).mean()
    out["feat_MACD"]  = ema12 - ema26
    out["feat_Vol20"] = p.rolling(20, min_periods=20).std()
    out["feat_Mom5"]  = p.pct_change(5)
    out["feat_Mom20"] = p.pct_change(20)
    ma20 = p.rolling(20, min_periods=20).mean()
    std20 = p.rolling(20, min_periods=20).std()
    out["feat_BB"] = (p - ma20) / (2 * std20 + 1e-8)
    return out


def fetch_ovx() -> pd.Series:
    print("[2/6] Fetching OVX (oil volatility)...")
    s = fetch_fred("OVXCLS", "OVX")
    if s is None:
        s = fetch_via_yfinance("^OVX", "OVX")
    return s


def fetch_dxy() -> pd.Series:
    print("[3/6] Fetching DXY (dollar index)...")
    s = fetch_via_yfinance("DX-Y.NYB", "DXY")
    if s is None or len(s) < 500:
        s = fetch_fred("DTWEXBGS", "DXY (broad dollar)")
    return s


def add_external_features(df: pd.DataFrame, series: pd.Series,
                           prefix: str) -> pd.DataFrame:
    out = df.copy()
    if series is None:
        out[f"{prefix}_raw"] = np.nan
        out[f"{prefix}_smooth5"] = np.nan
        out[f"{prefix}_change5"] = np.nan
        return out
    if series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    aligned = series.reindex(df.index).ffill()
    out[f"{prefix}_raw"] = aligned.values
    out[f"{prefix}_smooth5"] = aligned.rolling(5, min_periods=1).mean().values
    out[f"{prefix}_change5"] = aligned.diff(5).fillna(0.0).values
    return out


# ──────────────────────────────────────────────────────────────────────
# Sentiment from cache
# ──────────────────────────────────────────────────────────────────────
def load_cached_news_and_scores() -> pd.DataFrame:
    print("[4/6] Loading cached news + GPT scores (no API calls)...")
    news_path = find_cache_file("wti_news_cache.csv")
    score_path = find_cache_file("wti_scores_cache.csv")

    if news_path is None or score_path is None:
        print("  ⚠️  Cache files not found")
        return pd.DataFrame()

    print(f"  Found news cache:   {news_path}")
    print(f"  Found scores cache: {score_path}")

    news = pd.read_csv(news_path, parse_dates=["date"])
    scores = pd.read_csv(score_path)

    missing = [d for d in SENT_DIMS if d not in scores.columns]
    if missing:
        print(f"  ⚠️  Score cache missing columns: {missing}")
        return pd.DataFrame()

    merge_cols = ["headline"] + SENT_DIMS
    merged = news.merge(scores[merge_cols], on="headline", how="left")
    merged = merged.dropna(subset=SENT_DIMS)
    print(f"  Final news+scores: {len(merged):,} articles")
    return merged


def build_sentiment_series(news_df: pd.DataFrame,
                            dates: pd.DatetimeIndex) -> pd.DataFrame:
    cols = [f"sent_{d}" for d in SENT_DIMS]
    if news_df.empty:
        return pd.DataFrame(0.0, index=dates, columns=cols)

    nd = news_df.copy()
    nd["date"] = pd.to_datetime(nd["date"]).dt.normalize()

    decay = np.log(2) / SENTIMENT_HALFLIFE
    out = pd.DataFrame(0.0, index=dates, columns=cols)

    daily = {}
    for d in SENT_DIMS:
        if d == "relevance":
            daily[d] = nd.groupby("date")[d].mean().reindex(dates).fillna(0.0)
        else:
            def w(grp, dd=d):
                wt = grp["relevance"].values
                s = grp[dd].values
                return float(np.dot(wt, s) / wt.sum()) if wt.sum() > 1e-8 else 0.0
            daily[d] = nd.groupby("date").apply(w).reindex(dates).fillna(0.0)

    for d in SENT_DIMS:
        raw = daily[d].values
        smth = np.zeros(len(dates))
        for i in range(len(dates)):
            wts = np.exp(-decay * np.arange(i + 1)[::-1])
            wts /= wts.sum()
            smth[i] = np.dot(wts, raw[:i + 1])
        out[f"sent_{d}"] = smth

    coverage = (daily["relevance"] > 0).sum()
    print(f"  Sentiment coverage: {coverage:,}/{len(dates):,} days")
    return out


# ──────────────────────────────────────────────────────────────────────
# Labels
# ──────────────────────────────────────────────────────────────────────
def compute_labels(wti: pd.Series) -> np.ndarray:
    fwd = (wti.shift(-HORIZON) - wti) / wti
    return np.where(fwd.values > LABEL_THRESHOLD, 1,
           np.where(fwd.values < -LABEL_THRESHOLD, 0, -1)).astype(int)


# ──────────────────────────────────────────────────────────────────────
# Optuna objective — runs on training data only via TimeSeriesSplit
# ──────────────────────────────────────────────────────────────────────
def make_optuna_objective(X_train: np.ndarray, y_train: np.ndarray):
    """Returns an objective function for Optuna. Uses TimeSeriesSplit to
       avoid leaking future into past during tuning."""
    def objective(trial):
        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "verbosity": -1,
            "n_estimators": 300,
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 8, 64),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 50),
            "subsample": trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "random_state": 42,
            "n_jobs": -1,
        }

        tscv = TimeSeriesSplit(n_splits=OPTUNA_CV_SPLITS)
        aucs = []
        for train_idx, val_idx in tscv.split(X_train):
            if len(np.unique(y_train[train_idx])) < 2:
                continue
            n0 = max((y_train[train_idx] == 0).sum(), 1)
            n1 = max((y_train[train_idx] == 1).sum(), 1)
            clf = lgb.LGBMClassifier(**params, scale_pos_weight=n0 / n1)
            clf.fit(X_train[train_idx], y_train[train_idx],
                    eval_set=[(X_train[val_idx], y_train[val_idx])],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(-1)])
            prob = clf.predict_proba(X_train[val_idx])[:, 1]
            try:
                auc = roc_auc_score(y_train[val_idx], prob)
                aucs.append(auc)
            except ValueError:
                pass

        return float(np.mean(aucs)) if aucs else 0.5

    return objective


def tune_hyperparameters(X_train: np.ndarray, y_train: np.ndarray) -> dict:
    """Runs Optuna and returns the best hyperparameters."""
    print(f"\n[Optuna] Tuning {OPTUNA_N_TRIALS} trials with {OPTUNA_CV_SPLITS}-fold time-series CV...")
    sampler = optuna.samplers.TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    objective = make_optuna_objective(X_train, y_train)
    study.optimize(objective, n_trials=OPTUNA_N_TRIALS,
                   timeout=OPTUNA_TIMEOUT_SEC, show_progress_bar=True)
    print(f"  Best AUC (CV): {study.best_value:.4f}")
    print(f"  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")
    return study.best_params


# ──────────────────────────────────────────────────────────────────────
# Walk-forward with Optuna + Ensemble
# ──────────────────────────────────────────────────────────────────────
def walk_forward_ensemble(df: pd.DataFrame, feat_cols: list) -> dict:
    df = df.copy()
    df[PRICE_FEATURES] = df[PRICE_FEATURES].ffill()
    df = df.dropna(subset=PRICE_FEATURES)

    labels = compute_labels(df["WTI"])
    valid_idx = np.where(labels != -1)[0]
    valid_dates = df.index[valid_idx]
    valid_labels = labels[valid_idx]
    n = len(valid_idx)

    init_train = max(100, int(n * INIT_TRAIN_RATIO))
    X_all = df[feat_cols].values.astype(np.float32)
    wti = df["WTI"].values.astype(np.float32)

    # ── First: tune hyperparameters on initial training set ─────────
    print(f"\n[5/6] Hyperparameter tuning phase")
    initial_train_rows = valid_idx[:init_train]
    X_init_train = X_all[initial_train_rows]
    y_init_train = valid_labels[:init_train]

    # Scale and impute for tuning
    scaler_init = StandardScaler()
    X_init_scaled = scaler_init.fit_transform(X_init_train)
    imp_init = SimpleImputer(strategy="mean")
    X_init_scaled = imp_init.fit_transform(X_init_scaled)

    best_params = tune_hyperparameters(X_init_scaled, y_init_train)
    # Add fixed parameters
    best_params.update({
        "objective": "binary",
        "metric": "binary_logloss",
        "verbosity": -1,
        "n_estimators": 300,
        "n_jobs": -1,
    })

    # ── Walk-forward with ensemble ──────────────────────────────────
    print(f"\n[6/6] Walk-forward backtest with {len(ENSEMBLE_SEEDS)}-model ensemble")
    print(f"  (each fold trains {len(ENSEMBLE_SEEDS)} models, averages predictions)")

    all_pred, all_prob, all_true, all_mask, all_ret, all_dates_list = [], [], [], [], [], []
    feat_imp_accum = np.zeros(len(feat_cols))
    folds = 0
    cursor = init_train

    while cursor < n:
        test_end = min(cursor + STEP_SIZE, n)
        train_end = max(0, cursor - PURGE_GAP)
        idx_train_all = np.arange(0, train_end)
        idx_test = np.arange(cursor, test_end)
        if len(idx_train_all) < 80 or len(idx_test) == 0:
            cursor += STEP_SIZE
            continue
        split = max(50, int(len(idx_train_all) * (1 - VAL_FRACTION)))
        val_start = min(split + PURGE_GAP, len(idx_train_all))
        idx_train = idx_train_all[:split]
        idx_val = idx_train_all[val_start:]
        if len(idx_val) < 10:
            cursor += STEP_SIZE
            continue

        rows_train = valid_idx[idx_train]
        y_train = valid_labels[idx_train]
        rows_val = valid_idx[idx_val]
        y_val = valid_labels[idx_val]
        rows_test = valid_idx[idx_test]
        y_test = valid_labels[idx_test]

        scaler = StandardScaler()
        X_tr = scaler.fit_transform(X_all[rows_train])
        X_val = scaler.transform(X_all[rows_val])
        X_te = scaler.transform(X_all[rows_test])
        imp = SimpleImputer(strategy="mean")
        X_tr = imp.fit_transform(X_tr)
        X_val = imp.transform(X_val)
        X_te = imp.transform(X_te)

        n0 = max((y_train == 0).sum(), 1)
        n1 = max((y_train == 1).sum(), 1)
        spw = n0 / n1

        # Train ensemble: same hyperparams, different seeds
        ensemble_probs = []
        for seed in ENSEMBLE_SEEDS:
            params = dict(best_params)
            params["random_state"] = seed
            clf = lgb.LGBMClassifier(**params, scale_pos_weight=spw)
            clf.fit(X_tr, y_train, eval_set=[(X_val, y_val)],
                    callbacks=[lgb.early_stopping(30, verbose=False),
                               lgb.log_evaluation(-1)])
            prob_up = clf.predict_proba(X_te)[:, 1]
            ensemble_probs.append(prob_up)
            # Accumulate feature importance from first seed only
            if seed == ENSEMBLE_SEEDS[0]:
                feat_imp_accum += np.array(clf.feature_importances_, dtype=float)

        # Average predictions across ensemble
        prob_up = np.mean(ensemble_probs, axis=0)
        pred = (prob_up >= 0.5).astype(int)
        conf = np.maximum(prob_up, 1 - prob_up)
        edge = np.abs(prob_up - 0.5)
        mask = (conf >= CONFIDENCE_THRESHOLD) & (edge >= MIN_EDGE)

        # Non-overlap filter
        open_after = -1
        for i, row in enumerate(rows_test):
            if not mask[i]:
                continue
            if row < open_after:
                mask[i] = False
                continue
            open_after = row + HORIZON

        ret = np.zeros(len(rows_test), dtype=np.float32)
        for i, row in enumerate(rows_test):
            if not mask[i]:
                continue
            if row + HORIZON >= len(wti):
                mask[i] = False
                continue
            p0, p1 = float(wti[row]), float(wti[row + HORIZON])
            r = (p1 - p0) / (p0 + 1e-12)
            ret[i] = (r if pred[i] == 1 else -r) - ROUND_TRIP_FEE

        folds += 1

        all_pred.extend(pred.tolist())
        all_prob.extend(prob_up.tolist())
        all_true.extend(y_test.tolist())
        all_mask.extend(mask.tolist())
        all_ret.extend(ret.tolist())
        all_dates_list.extend(valid_dates[idx_test].tolist())

        cursor += STEP_SIZE
        if folds % 5 == 0:
            print(f"    Fold {folds} done...")

    y_arr = np.array(all_true, int)
    p_arr = np.array(all_pred, int)
    prob_arr = np.array(all_prob, float)
    m_arr = np.array(all_mask, bool)
    r_arr = np.array(all_ret, np.float32)

    if m_arr.any():
        acc = accuracy_score(y_arr[m_arr], p_arr[m_arr])
        mf1 = f1_score(y_arr[m_arr], p_arr[m_arr], average="macro", zero_division=0)
        dr = ((p_arr[m_arr] == 0) & (y_arr[m_arr] == 0)).sum() / max((y_arr[m_arr] == 0).sum(), 1)
        ur = ((p_arr[m_arr] == 1) & (y_arr[m_arr] == 1)).sum() / max((y_arr[m_arr] == 1).sum(), 1)
        try:
            overall_auc = roc_auc_score(y_arr[m_arr], prob_arr[m_arr])
        except ValueError:
            overall_auc = 0.5
        tr = r_arr[m_arr]
        total = float(tr.sum())
        sharpe = float((tr.mean() / (tr.std() + 1e-12)) * math.sqrt(252 / HORIZON))
    else:
        acc = mf1 = dr = ur = total = sharpe = 0.0
        overall_auc = 0.5

    feat_imp = (pd.DataFrame({
        "feature": feat_cols,
        "importance": feat_imp_accum / max(folds, 1),
    }).sort_values("importance", ascending=False))

    return dict(
        accuracy=acc, f1=mf1, total_return=total, sharpe=sharpe,
        auc=overall_auc, n_trades=int(m_arr.sum()),
        n_predictions=len(m_arr), feat_imp=feat_imp,
        down_recall=dr, up_recall=ur, best_params=best_params,
        all_pred=p_arr, all_prob=prob_arr, all_true=y_arr,
        all_mask=m_arr, all_ret=r_arr,
        all_dates=pd.DatetimeIndex(all_dates_list),
    )


# ──────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────
def main():
    df = fetch_wti()
    if RESTRICT_TO_N_YEARS:
        cutoff = df.index.max() - pd.DateOffset(years=RESTRICT_TO_N_YEARS)
        df = df[df.index >= cutoff].copy()
        print(f"  Restricted: {df.index.min().date()} -> {df.index.max().date()}, "
              f"{len(df):,} rows")

    df = add_price_features(df)

    ovx = fetch_ovx()
    df = add_external_features(df, ovx, "ovx")
    if ovx is not None:
        cov = (~df["ovx_raw"].isna()).sum()
        print(f"  OVX coverage: {cov:,}/{len(df):,} days")

    dxy = fetch_dxy()
    df = add_external_features(df, dxy, "dxy")
    if dxy is not None:
        cov = (~df["dxy_raw"].isna()).sum()
        print(f"  DXY coverage: {cov:,}/{len(df):,} days")

    news_scored = load_cached_news_and_scores()
    if news_scored.empty:
        print("\n⚠️  No cached sentiment available.")
        sent_cols = []
    else:
        sent = build_sentiment_series(news_scored, df.index)
        for c in sent.columns:
            df[c] = sent[c].reindex(df.index).fillna(0.0)
        sent_cols = list(sent.columns)

    feat_cols = PRICE_FEATURES + sent_cols + OVX_FEATURES + DXY_FEATURES
    print(f"\n  Final feature set: {len(feat_cols)} features")
    print(f"    Price: {len(PRICE_FEATURES)}, Sentiment: {len(sent_cols)}, "
          f"OVX: {len(OVX_FEATURES)}, DXY: {len(DXY_FEATURES)}")

    # Baselines
    labels_all = compute_labels(df["WTI"])
    labels_valid = labels_all[labels_all != -1]
    always_up_acc = (labels_valid == 1).mean()
    bh_total_ret = (df["WTI"].iloc[-1] / df["WTI"].iloc[0]) - 1
    print(f"\n  Always-up baseline : {always_up_acc*100:.2f}%")
    print(f"  Buy & hold return  : {bh_total_ret*100:+.1f}%")

    # Run model
    res = walk_forward_ensemble(df, feat_cols)

    # ── Print results ──────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("  RESULTS — Optuna-tuned ensemble (improving on Dai et al. 2026)")
    print("=" * 80)
    print(f"  Accuracy        : {res['accuracy']*100:.2f}%")
    print(f"  AUC             : {res['auc']:.4f}")
    print(f"  Macro F1        : {res['f1']:.3f}")
    print(f"  Sharpe          : {res['sharpe']:.2f}")
    print(f"  Total return    : {res['total_return']*100:+.1f}%")
    print(f"  Up recall       : {res['up_recall']:.3f}")
    print(f"  Down recall     : {res['down_recall']:.3f}")
    print(f"  Trades          : {res['n_trades']}/{res['n_predictions']} "
          f"({100*res['n_trades']/max(res['n_predictions'],1):.1f}%)")
    print(f"  Edge vs always-up: {(res['accuracy'] - always_up_acc)*100:+.2f} pp")

    print(f"\n  Optuna best params:")
    for k, v in res["best_params"].items():
        if k not in ("objective", "metric", "verbosity", "n_estimators", "n_jobs"):
            if isinstance(v, float):
                print(f"    {k}: {v:.6f}")
            else:
                print(f"    {k}: {v}")

    print("\n  Feature importance ranking:")
    for i, (_, row) in enumerate(res["feat_imp"].iterrows(), 1):
        f = row["feature"]
        if f.startswith("ovx_"):    marker = " <-- OVX"
        elif f.startswith("dxy_"):  marker = " <-- DXY"
        elif f.startswith("sent_"): marker = " <-- sentiment"
        else:                       marker = ""
        print(f"    {i:>2}. {f:<22} {row['importance']:>8.2f}{marker}")

    print("\n  Comparison to all prior runs:")
    print(f"    {'Configuration':<48} {'Accuracy':>10} {'AUC':>8} {'Sharpe':>8}")
    print(f"    {'-'*48} {'-'*10} {'-'*8} {'-'*8}")
    print(f"    {'Run 1: pipeline (price+F&G+GPT)':<48} {'52.86%':>10} {'—':>8} {'0.38':>8}")
    print(f"    {'Run 2: Colab replica (sent broken)':<48} {'57.67%':>10} {'—':>8} {'0.53':>8}")
    print(f"    {'Run 3: price+F&G only':<48} {'50.65%':>10} {'—':>8} {'-0.28':>8}")
    print(f"    {'OVX+DXY only (no sentiment)':<48} {'55.10%':>10} {'—':>8} {'-0.11':>8}")
    print(f"    {'price+sentiment+OVX+DXY':<48} {'52.35%':>10} {'—':>8} {'-0.51':>8}")
    print(f"    {'Dai et al. (2026) paper':<48} {'56.08%':>10} {'0.65':>8} {'—':>8}")
    print(f"    {'THIS RUN: Optuna + 5-model ensemble':<48} {res['accuracy']*100:>9.2f}% {res['auc']:>8.4f} {res['sharpe']:>8.2f}")
    print(f"    {'Always-up baseline':<48} {always_up_acc*100:>9.2f}% {'—':>8} {'—':>8}")

    # Save outputs
    print(f"\n  Saving outputs to {OUT_DIR}/...")
    pred_df = pd.DataFrame({
        "date": res["all_dates"],
        "actual": res["all_true"],
        "pred": res["all_pred"],
        "prob_up": res["all_prob"],
        "traded": res["all_mask"],
        "ret": res["all_ret"],
    })
    pred_df.to_csv(OUT_DIR / "predictions.csv", index=False)
    res["feat_imp"].to_csv(OUT_DIR / "feature_importance.csv", index=False)

    metadata = {
        "config": "Optuna-tuned LightGBM, 5-model ensemble, price+sentiment+OVX+DXY",
        "n_optuna_trials": OPTUNA_N_TRIALS,
        "n_ensemble_models": len(ENSEMBLE_SEEDS),
        "best_optuna_params": {
            k: v for k, v in res["best_params"].items()
            if k not in ("objective", "metric", "verbosity", "n_estimators", "n_jobs")
        },
        "accuracy": float(res["accuracy"]),
        "auc": float(res["auc"]),
        "sharpe": float(res["sharpe"]),
        "total_return": float(res["total_return"]),
        "always_up_baseline": float(always_up_acc),
        "buy_hold_return": float(bh_total_ret),
        "n_features": len(feat_cols),
        "n_trades": res["n_trades"],
        "data_start": str(df.index.min().date()),
        "data_end": str(df.index.max().date()),
        "dai_et_al_benchmark_accuracy": 0.5608,
        "dai_et_al_benchmark_auc": 0.65,
    }
    with open(OUT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"  ✅ Done. Results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
