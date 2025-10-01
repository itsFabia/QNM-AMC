# top_performers.py
import pandas as pd
import yfinance as yf
from typing import List, Tuple

def _pick(colnames, candidates):
    """Ersten passenden Spaltennamen wählen (inkl. 'endet mit'-Fallback)."""
    cols = [str(c).lower().replace(" ", "_") for c in colnames]
    # exakte Treffer
    for cand in candidates:
        if not cand:
            continue
        c = cand.lower()
        if c in cols:
            return cols[cols.index(c)]
    # endet-mit
    for cand in candidates:
        if not cand:
            continue
        c = cand.lower()
        for col in cols:
            if col.endswith(c):
                return col
    raise KeyError(f"Keine der Kandidaten gefunden: {candidates} in {cols}")

def _norm_cols(df: pd.DataFrame, ticker: str | None = None) -> pd.Series:
    """Adj-Close/Close robust finden (MultiIndex, Ticker-Suffixe, adjclose-Variante)."""
    # MultiIndex ggf. flatten
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["_".join([str(c) for c in col if c]) for col in df.columns.values]
    else:
        df.columns = [str(c) for c in df.columns]
    # vereinheitlichen
    df.columns = [c.lower().replace(" ", "_") for c in df.columns]

    t = (ticker or "").lower()
    candidates = [
        "adj_close", "adjclose", "close",          # Standards
        f"adj_close_{t}" if t else None,           # mit Ticker-Suffix
        f"adjclose_{t}" if t else None,
        f"close_{t}" if t else None,
    ]
    col = _pick(df.columns, candidates)
    return df[col]

def load_prices(tickers: List[str], start: str) -> pd.DataFrame:
    """Adj-Close/Close für mehrere Ticker laden (auto_adjust=True)."""
    data = {}
    for t in tickers:
        d = yf.download(t, start=start, auto_adjust=True, progress=False)
        if d.empty:
            print(f"Warnung: keine Daten für {t}")
            continue
        try:
            data[t] = _norm_cols(d, ticker=t)
        except KeyError as e:
            print(f"Spaltenproblem bei {t}: {list(d.columns)} -> {e}")
            raise
    if not data:
        raise ValueError("Keine Kursdaten geladen.")
    return pd.DataFrame(data).dropna(how="all")

def total_return_series(adj_close: pd.Series) -> pd.Series:
    """Kumulierte Total-Return-Reihe aus Adj-Close (inkl. Dividenden)."""
    ret = adj_close.pct_change().fillna(0.0)
    return (1 + ret).cumprod()

def compute_top_outperformers(
    index_members: List[str],
    benchmark_ticker: str,
    start: str = "2010-01-01",
    lookback_years: int = 10,
    top_n: int = 5,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Ranking der Top-N Outperformer ggü. Benchmark (Total-Return) + Benchmark-TR-Serie.
    """
    # Preise
    px_assets = load_prices(index_members, start=start)
    px_bench = load_prices([benchmark_ticker], start=start).iloc[:, 0]  # Serie

    # Total-Return
    tr_bench = total_return_series(px_bench)
    tr_assets = px_assets.apply(total_return_series)

    # Lookback-Fenster
    end_date = tr_bench.index.max()
    start_lb = end_date - pd.DateOffset(years=lookback_years)
    tr_bench_lb = tr_bench[tr_bench.index >= start_lb].dropna()
    tr_assets_lb = tr_assets[tr_assets.index >= start_lb].dropna(how="all")

    # Faktoren
    bench_factor = tr_bench_lb.iloc[-1] / tr_bench_lb.iloc[0]
    asset_factor = tr_assets_lb.iloc[-1] / tr_assets_lb.iloc[0]
    rel_factor = asset_factor / bench_factor

    out = pd.DataFrame({
        "TR_Factor_Asset": asset_factor,
        "TR_Factor_Bench": bench_factor,
        "Rel_Factor": rel_factor,
        "Outperformance_pct": (rel_factor - 1.0) * 100.0,
        "Start": tr_assets_lb.index[0] if len(tr_assets_lb) else pd.NaT,
        "End": tr_assets_lb.index[-1] if len(tr_assets_lb) else pd.NaT,
    }).sort_values("Rel_Factor", ascending=False)

    return out.head(top_n), tr_bench
