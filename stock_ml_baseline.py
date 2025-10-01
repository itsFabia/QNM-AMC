# stock_ml_baseline.py
import warnings; warnings.filterwarnings("ignore")

# ------------------------------------------------
# Imports
# ------------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

# Eigene Module
from top_performers import compute_top_outperformers
from top_performers_SMI import SMI_TICKERS, SMI_BENCH  # ^SSMI

# ------------------------------------------------
# Globale Einstellungen
# ------------------------------------------------
LOOKBACK_YEARS = 10
TOP_N = 5
DATA_START = "2010-01-01"   # großzügig; Lookback schneidet das Fenster
PRED_THRESH = 0.52
FEE_BPS = 2
MIN_HOLD_DAYS = 10
N_SPLITS = 5
ROLL_BETA = 60  # für Beta/Excess-Return

# ------------------------------------------------
# 1) Top-5 Outperformer vs. SMI (10 Jahre)
# ------------------------------------------------
top5, smi_tr = compute_top_outperformers(
    index_members=SMI_TICKERS,
    benchmark_ticker=SMI_BENCH,
    start=DATA_START,
    lookback_years=LOOKBACK_YEARS,
    top_n=TOP_N,
)
print("\nTop 5 Outperformer vs. SMI (10 Jahre):")
print(top5)

# ------------------------------------------------
# Hilfsfunktionen: Spalten-Picking & TR-Loader (robust)
# ------------------------------------------------
def _pick(colnames, candidates):
    cols = [str(c).lower().replace(" ", "_") for c in colnames]
    # exakte Treffer
    for cand in candidates:
        if not cand: continue
        c = cand.lower()
        if c in cols:
            return cols[cols.index(c)]
    # endet-mit (für Suffixe wie close_nesn.sw)
    for cand in candidates:
        if not cand: continue
        c = cand.lower()
        for col in cols:
            if col.endswith(c):
                return col
    return None

def _flatten_cols(columns):
    # MultiIndex -> Strings, ansonsten Stringify
    try:
        import pandas as pd  # local scope
        if isinstance(columns, pd.MultiIndex):
            return [
                "_".join([str(x) for x in col if x is not None and str(x) != ""])
                for col in columns.values
            ]
        return [str(c) for c in columns]
    except Exception:
        return [str(c) for c in columns]

def _load_tr_series(ticker: str, start: str) -> pd.Series:
    d = yf.download(ticker, start=start, auto_adjust=True, progress=False)
    if d.empty:
        raise ValueError(f"Keine Daten für {ticker}")
    d.columns = _flatten_cols(d.columns)
    d.columns = [c.lower().replace(" ", "_") for c in d.columns]
    t = ticker.lower()
    col = _pick(d.columns, ["adj_close", "adjclose", "close",
                            f"adj_close_{t}", f"adjclose_{t}", f"close_{t}"])
    if col is None:
        raise KeyError(f"{ticker}: keine Adj/Close-Spalte gefunden in {list(d.columns)}")
    px = d[col]
    ret = px.pct_change().fillna(0.0)
    return (1 + ret).cumprod()

# ------------------------------------------------
# Plot: SMI (TR) + Top-5 (normiert)
# ------------------------------------------------
plt.figure(figsize=(11,6))
smi_tr.plot(label="SMI (TR)")
for t in top5.index:
    try:
        tr = _load_tr_series(t, DATA_START)
        merged = pd.concat([smi_tr.rename("SMI"), tr.rename(t)], axis=1).dropna()
        if len(merged) < 2:
            print(f"Zu wenig überlappende Daten für {t} – überspringe.")
            continue
        (merged[t] / merged[t].iloc[0]).plot(label=t)
    except Exception as e:
        print(f"Plot-Skip {t}: {e}")
plt.title("Top-5 vs. SMI – Total Return (normiert)")
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

# ------------------------------------------------
# 2) ML-Baseline: Outperformance ggü. Index (für Top-1-Ticker)
# ------------------------------------------------
TICKER = top5.index[0]
print(f"\nML-Baseline auf Outperformance vs. Index für: {TICKER}")

def _normalize_ohlc(df, ticker=None):
    """Normiert Yahoo-Download auf: adj_close, close, open, high, low, volume."""
    df.columns = _flatten_cols(df.columns)
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]
    t = (ticker or "").lower()
    out = {}

    # adj_close / close
    adjc = _pick(df.columns, ["adj_close", "adjclose", f"adj_close_{t}", f"adjclose_{t}"])
    clos = _pick(df.columns, ["close", f"close_{t}"])
    if adjc is not None:
        out["adj_close"] = df[adjc]
    if clos is not None:
        out["close"] = df[clos]

    # open / high / low / volume
    for k in ["open", "high", "low", "volume"]:
        col = _pick(df.columns, [k, f"{k}_{t}"])
        if col is not None:
            out[k] = df[col]

    if not out.get("adj_close") and not out.get("close"):
        raise KeyError(f"{ticker or 'Asset'}: keine Adj/Close-Spalte gefunden in {list(df.columns)}")

    return pd.DataFrame(out)

# Preise laden (Asset + Benchmark)
asset_raw = yf.download(TICKER, start=DATA_START, auto_adjust=True, progress=False)
bench_raw = yf.download(SMI_BENCH, start=DATA_START, auto_adjust=True, progress=False)
if asset_raw.empty: raise ValueError(f"Keine Daten für {TICKER}")
if bench_raw.empty: raise ValueError(f"Keine Daten für {SMI_BENCH}")

a = _normalize_ohlc(asset_raw, TICKER)
b = _normalize_ohlc(bench_raw, SMI_BENCH)

# Close-Serien (Adj bevorzugt)
a_close = (a["adj_close"] if "adj_close" in a.columns else a["close"]).rename("a_close")
b_close = (b["adj_close"] if "adj_close" in b.columns else b["close"]).rename("b_close")

# Gemeinsames DF
df = pd.concat([a_close, b_close], axis=1).dropna()
df["ret"] = df["a_close"].pct_change()
df["bm_ret"] = df["b_close"].pct_change()

# Rolling-Beta & Excess-Return (beta-gehedged)
cov = df["ret"].rolling(ROLL_BETA).cov(df["bm_ret"])
var = df["bm_ret"].rolling(ROLL_BETA).var() + 1e-12
df["beta"] = (cov / var).clip(-5, 5)
df["excess_ret"] = df["ret"] - df["beta"] * df["bm_ret"]

# Ziel: Outperformance morgen
df["target"] = (df["excess_ret"].shift(-1) > 0).astype(int)

# ------------------------------------------------
# Features (markt-relativ) – nutzt normalisierte a/b
# ------------------------------------------------
def make_features_market_rel(d, windows=(3,5,10,20)):
    out = d.copy()
    for w in windows:
        out[f"ret_{w}d"]      = out["a_close"].pct_change(w)
        out[f"bm_ret_{w}d"]   = out["b_close"].pct_change(w)
        out[f"ret_ex_{w}d"]   = out[f"ret_{w}d"] - out[f"bm_ret_{w}d"]
        out[f"vol_{w}d"]      = out["ret"].rolling(w).std()
        out[f"bm_vol_{w}d"]   = out["bm_ret"].rolling(w).std()
        out[f"zex_{w}d"]      = (out["excess_ret"] - out["excess_ret"].rolling(w).mean()) / (out["excess_ret"].rolling(w).std() + 1e-9)
        out[f"beta_{w}d"]     = out["beta"].rolling(w).mean()
        if "high" in a.columns and "low" in a.columns:
            out[f"hi_lo_{w}d"] = (a["high"].rolling(w).max() - a["low"].rolling(w).min()) / out["a_close"]
        if "volume" in a.columns:
            out[f"volchg_{w}d"] = a["volume"].pct_change(w)
    out["dow"] = out.index.dayofweek
    out = pd.get_dummies(out, columns=["dow"], drop_first=True)
    out = out.dropna()
    return out

df_feat = make_features_market_rel(df)

drop_cols = ["target","a_close","b_close","ret","bm_ret","beta","excess_ret"]
feature_cols = [c for c in df_feat.columns if c not in drop_cols]
X = df_feat[feature_cols]
y = df_feat["target"]
dates = df_feat.index

# ------------------------------------------------
# Modelle & Cross-Validation
# ------------------------------------------------
logit = Pipeline([
    ("scaler", StandardScaler()),
    ("clf", LogisticRegression(max_iter=1000))
])

xgb = XGBClassifier(
    n_estimators=400,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_lambda=1.0,
    eval_metric="logloss",
    tree_method="hist",
    random_state=42,
)

def evaluate_model(model, X, y, n_splits=5):
    tscv = TimeSeriesSplit(n_splits=n_splits)
    proba_all = np.full(len(y), np.nan)
    rows = []
    for tr, te in tscv.split(X):
        Xtr, Xte = X.iloc[tr], X.iloc[te]
        ytr, yte = y.iloc[tr], y.iloc[te]
        model.fit(Xtr, ytr)
        p = model.predict_proba(Xte)[:,1]
        proba_all[te] = p
        yhat = (p>=0.5).astype(int)
        rows.append({"acc": float(accuracy_score(yte, yhat)),
                     "auc": float(roc_auc_score(yte, p))})
    return pd.DataFrame(rows), pd.Series(proba_all, index=y.index)

log_cv, log_proba = evaluate_model(logit, X, y, n_splits=N_SPLITS)
xgb_cv, xgb_proba = evaluate_model(xgb, X, y, n_splits=N_SPLITS)

print("\nCV-Logit:", log_cv.mean().round(4).to_dict())
print("CV-XGB  :", xgb_cv.mean().round(4).to_dict())

# ------------------------------------------------
# Backtest (Pair-Trade: long Asset, short Beta*Index -> excess_ret)
# Korrigierte Fees (einseitig pro Wechsel) + Mindesthaltefrist
# ------------------------------------------------
def backtest_with_min_hold(
    proba, dates, ret_series,
    thresh=0.52, fee_bps=2, slippage_bps=0,
    min_hold_days=10
):
    s = pd.Series(proba, index=dates).ffill()
    raw_sig = (s >= thresh).astype(int)

    # State machine: Entscheidung T, Ausführung T+1
    pos = []
    current = 0
    hold = 0
    for sig in raw_sig:
        if hold > 0:
            pos.append(current); hold -= 1
        else:
            if sig != current:
                current = sig
                hold = max(min_hold_days - 1, 0)
            pos.append(current)

    pos = pd.Series(pos, index=raw_sig.index).shift(1).fillna(0).astype(int)

    trades = pos.diff().fillna(pos.iloc[0]).abs()
    fee = fee_bps / 10_000.0
    slip = slippage_bps / 10_000.0
    daily_costs = trades * (fee + slip)

    strat_ret = pos * ret_series - daily_costs
    equity = (1 + strat_ret).cumprod()
    bench  = (1 + ret_series).cumprod()

    n_trades = int(trades.sum())
    hold_lengths = pos.groupby((pos != pos.shift()).cumsum()).transform('size')[pos == 1]
    avg_hold = float(hold_lengths.mean()) if not hold_lengths.empty else 0.0
    turnover = float(trades.sum() / max((pos.abs().sum()), 1))
    extras = {"Trades": n_trades, "AvgHoldDays": avg_hold, "Turnover": turnover}
    return equity, bench, strat_ret, pos, extras

def metrics(equity, ret_series, rf=0.0):
    ann = 252
    cagr = equity.iloc[-1]**(ann/len(ret_series)) - 1
    vol = ret_series.std() * np.sqrt(ann)
    sharpe = ((ret_series.mean() - rf/ann) * ann) / (vol + 1e-12)
    dd_curve = equity / equity.cummax() - 1
    maxdd = dd_curve.min()
    calmar = cagr / abs(maxdd) if maxdd < 0 else np.nan
    downside = ret_series[ret_series < 0]
    dvol = downside.std() * np.sqrt(ann) if len(downside) else np.nan
    dsharpe = ((ret_series.mean())*ann) / (dvol + 1e-12) if pd.notna(dvol) else np.nan
    hit = (ret_series > 0).mean()
    return {"CAGR": cagr, "Vol": vol, "Sharpe": sharpe, "DownsideSharpe": dsharpe,
            "MaxDD": maxdd, "Calmar": calmar, "Hit-Rate": hit}

eq, bench_ex, strat, pos, extras = backtest_with_min_hold(
    xgb_proba, dates, df_feat["excess_ret"].loc[dates],
    thresh=PRED_THRESH, fee_bps=FEE_BPS, slippage_bps=0, min_hold_days=MIN_HOLD_DAYS
)

m_strat = {k: round(v,4) for k,v in metrics(eq, strat).items()}
m_bench = {k: round(v,4) for k,v in metrics(bench_ex, df_feat["excess_ret"].loc[dates]).items()}
print("\nExtras:", {k: (round(v,4) if isinstance(v, float) else v) for k,v in extras.items()})
print("Strategy (Pair-Trade, excess_ret):", m_strat)
print("Benchmark (passiv excess_ret):   ", m_bench)

# Plot Equity
plt.figure(figsize=(10,6))
eq.plot(label="Strategy (Pair-Trade)")
bench_ex.plot(label="Benchmark (excess_ret Buy&Hold)")
plt.legend()
plt.title(f"Equity Curve – {TICKER} vs. SMI (Outperformance-Strategie)")
plt.tight_layout()
plt.show()

# Feature Importances (XGB)
xgb.fit(X, y)
fi = pd.Series(xgb.feature_importances_, index=X.columns).sort_values(ascending=False).head(15)
print("\nTop-Features:\n", fi.round(4))
