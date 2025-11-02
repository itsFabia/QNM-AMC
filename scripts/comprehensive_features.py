"""
Comprehensive Feature Engineering - 180+ Features
Implementiert alle technisch möglichen Features aus der 270-Feature-Liste
+ Shiller Market-Regime-Filter
"""
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CATEGORY 1: PRICE-BASED FEATURES (40 Features)
# ============================================================

def add_return_features(g: pd.DataFrame, price_col='Price', return_col='Return_raw') -> pd.DataFrame:
    """Returns - Multi-Period & Risk-Adjusted"""
    p = g[price_col]
    r = g[return_col]

    # Multi-Period Returns (16 Features)
    for days, name in [(1,'1d'), (2,'2d'), (3,'3d'), (5,'5d'), (10,'10d'),
                       (20,'20d'), (60,'60d'), (120,'120d'), (252,'252d')]:
        g[f'ret_{name}'] = p.pct_change(days).shift(1)

    # Weekly/Monthly Returns
    g['ret_1w'] = p.pct_change(5).shift(1)
    g['ret_1m'] = p.pct_change(21).shift(1)
    g['ret_3m'] = p.pct_change(63).shift(1)
    g['ret_6m'] = p.pct_change(126).shift(1)
    g['ret_12m'] = p.pct_change(252).shift(1)

    # Year-to-Date / Quarter-to-Date (simplified - using year/quarter columns)
    if 'Date' in g.columns:
        g['year'] = pd.to_datetime(g['Date']).dt.year
        g['quarter'] = pd.to_datetime(g['Date']).dt.quarter
        year_first = g.groupby('year')[price_col].transform('first')
        quarter_first = g.groupby(['year', 'quarter'])[price_col].transform('first')
        g['ret_ytd'] = ((p / year_first) - 1).shift(1)
        g['ret_qtd'] = ((p / quarter_first) - 1).shift(1)
        g.drop(['year', 'quarter'], axis=1, inplace=True)
    else:
        g['ret_ytd'] = 0
        g['ret_qtd'] = 0

    # Risk-Adjusted Returns (8 Features)
    for window, name in [(20,'20d'), (60,'60d'), (63,'3m'), (126,'6m')]:
        vol = r.rolling(window, min_periods=window//2).std(ddof=0)
        ret = p.pct_change(window)
        g[f'ret_{name}_sharpe'] = (ret / (vol * np.sqrt(window) + 1e-10)).shift(1)

    # Calmar Ratio (Return / Max Drawdown)
    for window, name in [(63,'3m'), (126,'6m'), (252,'12m')]:
        ret = p.pct_change(window)
        cummax = p.rolling(window, min_periods=window//2).max()
        dd = (p - cummax) / cummax
        max_dd = dd.rolling(window, min_periods=window//2).min()
        g[f'ret_{name}_calmar'] = (ret / (abs(max_dd) + 1e-10)).shift(1)

    # Sortino Ratio (downside deviation)
    downside = r[r < 0].rolling(63, min_periods=30).std(ddof=0)
    ret_3m = p.pct_change(63)
    g['sortino_ratio_3m'] = (ret_3m / (downside + 1e-10)).shift(1)

    return g


def add_price_level_features(g: pd.DataFrame, price_col='Price') -> pd.DataFrame:
    """Price Levels & Ratios (10 Features)"""
    p = g[price_col]

    # Simple Moving Averages
    for window in [20, 50, 100, 200]:
        sma = p.rolling(window, min_periods=window//2).mean()
        g[f'price_to_sma{window}'] = (p / sma).shift(1)

    # 52-Week High/Low
    high_52w = p.rolling(252, min_periods=126).max()
    low_52w = p.rolling(252, min_periods=126).min()

    g['price_to_52w_high'] = (p / high_52w).shift(1)
    g['price_to_52w_low'] = (p / low_52w).shift(1)
    g['pct_off_52w_high'] = ((high_52w - p) / high_52w).shift(1)
    g['pct_above_52w_low'] = ((p - low_52w) / low_52w).shift(1)
    g['price_range_52w'] = ((p - low_52w) / (high_52w - low_52w + 1e-10)).shift(1)

    # Log Price
    g['log_price'] = np.log(p + 1e-10).shift(1)

    return g


# ============================================================
# CATEGORY 2: MOMENTUM FEATURES (30 Features)
# ============================================================

def add_momentum_features(g: pd.DataFrame, price_col='Price', return_col='Return_raw') -> pd.DataFrame:
    """Classic Momentum, Acceleration, Cross-Sectional"""
    p = g[price_col]
    r = g[return_col]

    # Classic Momentum (8 Features)
    for months in [1, 2, 3, 6, 9, 12, 18, 24]:
        days = int(months * 21)
        g[f'mom_{months}m'] = p.pct_change(days).shift(1)

    # Momentum Acceleration (6 Features)
    mom_1m = p.pct_change(21)
    mom_3m = p.pct_change(63)
    mom_6m = p.pct_change(126)

    g['mom_acceleration_1m'] = (mom_1m - mom_3m).shift(1)
    g['mom_acceleration_3m'] = (mom_3m - mom_6m).shift(1)
    g['mom_deceleration'] = (mom_3m - mom_1m).shift(1)

    # Short-term Reversal
    g['short_term_reversal_1w'] = (-p.pct_change(5)).shift(1)
    g['short_term_reversal_1d'] = (-r).shift(1)

    # Momentum Consistency (% positive days)
    g['mom_consistency'] = (r > 0).rolling(60, min_periods=30).mean().shift(1)

    # Weighted Momentum (4 Features)
    # Equal-weighted
    mom_equal = (mom_1m + mom_3m + mom_6m) / 3
    g['mom_weighted_equal'] = mom_equal.shift(1)

    # Recent-weighted (more weight on shorter periods)
    mom_recent = (3*mom_1m + 2*mom_3m + 1*mom_6m) / 6
    g['mom_weighted_recent'] = mom_recent.shift(1)

    # Exponential (EMA-based)
    g['mom_exponential'] = p.ewm(span=63, min_periods=30).mean().pct_change(63).shift(1)

    # Composite (average of multiple signals)
    g['mom_composite'] = (mom_1m + mom_3m + mom_6m + mom_recent) / 4
    g['mom_composite'] = g['mom_composite'].shift(1)

    # Momentum Quality (6 Features)
    vol_3m = r.rolling(63, min_periods=30).std(ddof=0)
    vol_6m = r.rolling(126, min_periods=60).std(ddof=0)

    g['mom_smoothness_3m'] = (mom_3m / (vol_3m * np.sqrt(63) + 1e-10)).shift(1)
    g['mom_smoothness_6m'] = (mom_6m / (vol_6m * np.sqrt(126) + 1e-10)).shift(1)

    # Trend Strength
    sma_20 = p.rolling(20, min_periods=10).mean()
    sma_50 = p.rolling(50, min_periods=25).mean()
    sma_200 = p.rolling(200, min_periods=100).mean()

    g['trend_strength_20_50'] = ((sma_20 - sma_50) / sma_50).shift(1)
    g['trend_strength_50_200'] = ((sma_50 - sma_200) / sma_200).shift(1)

    # Momentum-Volatility Ratio
    g['momentum_volatility_ratio'] = (mom_3m / (vol_3m + 1e-10)).shift(1)

    # Momentum Z-Score
    mom_mean = mom_3m.rolling(252, min_periods=126).mean()
    mom_std = mom_3m.rolling(252, min_periods=126).std(ddof=0)
    g['momentum_z_score'] = ((mom_3m - mom_mean) / (mom_std + 1e-10)).shift(1)

    return g


# ============================================================
# CATEGORY 3: VOLATILITY FEATURES (30 Features)
# ============================================================

def add_volatility_features(g: pd.DataFrame, return_col='Return_raw', price_col='Price') -> pd.DataFrame:
    """Historical Volatility, Regimes, Advanced Estimators"""
    r = g[return_col]
    p = g[price_col]

    # Historical Volatility - Standard (10 Features)
    for window in [5, 10, 20, 30, 60, 120, 252]:
        vol = r.rolling(window, min_periods=window//2).std(ddof=0)
        g[f'vol_{window}d'] = (vol * np.sqrt(252)).shift(1)  # Annualized

    # Intraday volatility proxy (wenn nur Close, dann = 0)
    g['vol_intraday_avg'] = 0  # Placeholder (braucht High-Low)
    g['vol_overnight'] = r.shift(1)  # Overnight = previous return
    g['vol_close_to_close'] = r.rolling(20, min_periods=10).std(ddof=0).shift(1)

    # Volatility Ratios & Regimes (6 Features)
    vol_5d = r.rolling(5, min_periods=3).std(ddof=0)
    vol_20d = r.rolling(20, min_periods=10).std(ddof=0)
    vol_252d = r.rolling(252, min_periods=126).std(ddof=0)

    g['vol_ratio_short_long'] = (vol_20d / (vol_252d + 1e-10)).shift(1)
    g['vol_ratio_5d_20d'] = (vol_5d / (vol_20d + 1e-10)).shift(1)

    # Volatility Percentile
    g['vol_regime_percentile'] = vol_20d.rolling(252, min_periods=126).rank(pct=True).shift(1)

    # Volatility Spike (> 2 std above mean)
    vol_mean = vol_20d.rolling(252, min_periods=126).mean()
    vol_std = vol_20d.rolling(252, min_periods=126).std(ddof=0)
    g['vol_spike_indicator'] = ((vol_20d > vol_mean + 2*vol_std).astype(int)).shift(1)

    # Volatility Trend
    vol_ma_short = vol_20d.rolling(20, min_periods=10).mean()
    vol_ma_long = vol_20d.rolling(60, min_periods=30).mean()
    g['vol_trend'] = ((vol_ma_short - vol_ma_long) / vol_ma_long).shift(1)

    # Volatility Mean Reversion
    g['vol_mean_reversion'] = ((vol_mean - vol_20d) / (vol_std + 1e-10)).shift(1)

    # Advanced Volatility Estimators (8 Features)
    # Parkinson (braucht High-Low) - Placeholder
    g['parkinson_vol'] = vol_20d.shift(1)  # Fallback zu Standard-Vol
    g['garman_klass_vol'] = vol_20d.shift(1)
    g['rogers_satchell_vol'] = vol_20d.shift(1)
    g['yang_zhang_vol'] = vol_20d.shift(1)
    g['realized_vol'] = vol_20d.shift(1)
    g['implied_vol_proxy'] = vol_20d.shift(1)

    # Vol of Vol
    g['vol_of_vol'] = vol_20d.rolling(20, min_periods=10).std(ddof=0).shift(1)

    # Vol Persistence (autocorrelation)
    vol_lag1 = vol_20d.shift(1)
    g['vol_persistence'] = vol_20d.rolling(60, min_periods=30).corr(vol_lag1).shift(1)

    # Directional Volatility (6 Features)
    r_positive = r.clip(lower=0)
    r_negative = r.clip(upper=0)

    g['downside_vol'] = (r_negative.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252)).shift(1)
    g['upside_vol'] = (r_positive.rolling(20, min_periods=10).std(ddof=0) * np.sqrt(252)).shift(1)
    g['vol_asymmetry'] = (g['downside_vol'] / (g['upside_vol'] + 1e-10))

    # Semi-Deviation
    mean_ret = r.rolling(60, min_periods=30).mean()
    downside_dev = (r[r < mean_ret] - mean_ret).rolling(60, min_periods=30).std(ddof=0)
    g['semi_deviation'] = downside_dev.shift(1)

    # Downside Beta (placeholder - needs market data)
    g['downside_beta'] = 1.0  # Placeholder

    # Tail Risk (5th percentile)
    g['tail_risk'] = r.rolling(252, min_periods=126).quantile(0.05).shift(1)

    return g


# ============================================================
# CATEGORY 4: VOLUME FEATURES (25 Features)
# ============================================================

def add_volume_features(g: pd.DataFrame, price_col='Price', volume_col='Volume', return_col='Return_raw') -> pd.DataFrame:
    """Volume Basics, Price-Volume Interaction, Advanced Indicators"""
    p = g[price_col]
    v = g[volume_col]
    r = g[return_col]

    # Volume Basics (8 Features)
    g['volume'] = v.shift(1)

    for window in [5, 20, 60]:
        vol_avg = v.rolling(window, min_periods=window//2).mean()
        g[f'volume_{window}d_avg'] = vol_avg.shift(1)
        g[f'volume_ratio_{window}d'] = (v / (vol_avg + 1)).shift(1)

    # Dollar Volume
    g['dollar_volume'] = (p * v).shift(1)

    # Volume Percentile
    g['volume_percentile_60d'] = v.rolling(60, min_periods=30).rank(pct=True).shift(1)

    # Volume-Price Interaction (9 Features)
    # On-Balance Volume
    obv = (v * np.sign(r)).cumsum()
    g['obv'] = obv.shift(1)
    g['obv_change_20d'] = obv.diff(20).shift(1)
    g['obv_slope'] = (obv - obv.shift(20)) / 20
    g['obv_slope'] = g['obv_slope'].shift(1)

    # VWAP Distance
    vwap = (p * v).rolling(20, min_periods=10).sum() / v.rolling(20, min_periods=10).sum()
    g['vwap_distance'] = ((p - vwap) / vwap).shift(1)
    g['vwap_ratio'] = (p / vwap).shift(1)

    # Volume-Weighted Return
    g['volume_weighted_return'] = (r * v / v.rolling(20, min_periods=10).sum()).rolling(20).sum().shift(1)

    # Price-Volume Correlation
    g['price_volume_correlation'] = p.rolling(20, min_periods=10).corr(v).shift(1)

    # Volume Momentum
    g['volume_momentum'] = v.pct_change(20).shift(1)

    # Volume-Price Trend
    g['volume_price_trend'] = (r * v).rolling(20, min_periods=10).sum().shift(1)

    # Advanced Volume Indicators (8 Features)
    # Force Index
    g['force_index'] = (r * v).shift(1)

    # Ease of Movement (placeholder)
    g['ease_of_movement'] = (r / (v + 1)).shift(1)

    # Volume Trend
    vol_ma_10 = v.rolling(10, min_periods=5).mean()
    vol_ma_20 = v.rolling(20, min_periods=10).mean()
    g['volume_trend_20d'] = ((vol_ma_10 - vol_ma_20) / vol_ma_20).shift(1)

    # Accumulation/Distribution
    mfm = ((p - p.rolling(2).min()) - (p.rolling(2).max() - p)) / (p.rolling(2).max() - p.rolling(2).min() + 1e-10)
    ad_line = (mfm * v).cumsum()
    g['accumulation_distribution'] = ad_line.shift(1)

    # Chaikin Money Flow
    cmf = (mfm * v).rolling(20, min_periods=10).sum() / v.rolling(20, min_periods=10).sum()
    g['chaikin_money_flow'] = cmf.shift(1)

    # Money Flow Index
    typical_price = p  # Simplified (braucht eigentlich (H+L+C)/3)
    money_flow = typical_price * v
    positive_mf = money_flow[r > 0].rolling(14, min_periods=7).sum()
    negative_mf = money_flow[r < 0].rolling(14, min_periods=7).sum()
    mfi = 100 - (100 / (1 + positive_mf / (negative_mf + 1)))
    g['money_flow_index'] = mfi.shift(1)

    # Volume ROC
    g['volume_rate_of_change'] = v.pct_change(20).shift(1)

    # Klinger Oscillator (simplified)
    g['klinger_oscillator'] = (v.ewm(span=34).mean() - v.ewm(span=55).mean()).shift(1)

    return g


# ============================================================
# CATEGORY 5: TECHNICAL INDICATORS (35 Features)
# ============================================================

def add_technical_indicators(g: pd.DataFrame, price_col='Price', return_col='Return_raw') -> pd.DataFrame:
    """Moving Averages, MACD, RSI, Bollinger, Stochastic"""
    p = g[price_col]
    r = g[return_col]

    # Moving Averages (10 Features)
    for window in [10, 20, 50, 100, 200]:
        g[f'sma_{window}'] = p.rolling(window, min_periods=window//2).mean().shift(1)
        g[f'ema_{window}'] = p.ewm(span=window, min_periods=window//2).mean().shift(1)

    # Moving Average Crossovers (5 Features)
    sma_20 = p.rolling(20, min_periods=10).mean()
    sma_50 = p.rolling(50, min_periods=25).mean()
    sma_200 = p.rolling(200, min_periods=100).mean()
    ema_12 = p.ewm(span=12, min_periods=6).mean()
    ema_26 = p.ewm(span=26, min_periods=13).mean()

    g['sma_crossover_20_50'] = ((sma_20 > sma_50).astype(int)).shift(1)
    g['sma_crossover_50_200'] = ((sma_50 > sma_200).astype(int)).shift(1)  # Golden Cross
    g['ema_crossover_12_26'] = ((ema_12 > ema_26).astype(int)).shift(1)

    g['distance_from_sma_20'] = ((p - sma_20) / sma_20).shift(1)
    g['distance_from_sma_200'] = ((p - sma_200) / sma_200).shift(1)

    # MACD (4 Features)
    macd_line = ema_12 - ema_26
    macd_signal = macd_line.ewm(span=9, min_periods=5).mean()
    macd_hist = macd_line - macd_signal

    g['macd_line'] = macd_line.shift(1)
    g['macd_signal'] = macd_signal.shift(1)
    g['macd_histogram'] = macd_hist.shift(1)
    g['macd_crossover'] = ((macd_line > macd_signal).astype(int)).shift(1)

    # RSI (5 Features)
    for window in [14, 30, 60]:
        up = r.clip(lower=0).rolling(window, min_periods=window//2).sum()
        down = (-r.clip(upper=0)).rolling(window, min_periods=window//2).sum()
        rsi = 100 * up / (up + down + 1e-10)
        g[f'rsi_{window}'] = rsi.shift(1)

    # RSI Divergence (simplified: RSI vs Price momentum)
    rsi_14 = g['rsi_14']
    price_mom = p.pct_change(14)
    g['rsi_divergence'] = (rsi_14.diff(14) * price_mom.diff(14) < 0).astype(int)

    # RSI Overbought/Oversold
    g['rsi_overbought_oversold'] = ((rsi_14 > 70) | (rsi_14 < 30)).astype(int)

    # Stochastic Oscillator (4 Features)
    low_14 = p.rolling(14, min_periods=7).min()
    high_14 = p.rolling(14, min_periods=7).max()
    stoch_k = 100 * (p - low_14) / (high_14 - low_14 + 1e-10)
    stoch_d = stoch_k.rolling(3, min_periods=2).mean()

    g['stochastic_k'] = stoch_k.shift(1)
    g['stochastic_d'] = stoch_d.shift(1)
    g['stochastic_crossover'] = ((stoch_k > stoch_d).astype(int)).shift(1)
    g['stochastic_overbought_oversold'] = ((stoch_k > 80) | (stoch_k < 20)).astype(int)

    # Bollinger Bands (5 Features)
    bb_middle = sma_20
    bb_std = r.rolling(20, min_periods=10).std(ddof=0) * p
    bb_upper = bb_middle + 2 * bb_std
    bb_lower = bb_middle - 2 * bb_std

    g['bb_upper'] = bb_upper.shift(1)
    g['bb_middle'] = bb_middle.shift(1)
    g['bb_lower'] = bb_lower.shift(1)
    g['bb_width'] = ((bb_upper - bb_lower) / bb_middle).shift(1)
    g['bb_position'] = ((p - bb_lower) / (bb_upper - bb_lower + 1e-10)).shift(1)

    # Williams %R (1 Feature)
    g['williams_r'] = -100 * (high_14 - p) / (high_14 - low_14 + 1e-10)
    g['williams_r'] = g['williams_r'].shift(1)

    # CCI - Commodity Channel Index (1 Feature)
    tp = p  # Typical Price (simplified)
    tp_sma = tp.rolling(20, min_periods=10).mean()
    tp_mad = (tp - tp_sma).abs().rolling(20, min_periods=10).mean()
    cci = (tp - tp_sma) / (0.015 * tp_mad + 1e-10)
    g['cci'] = cci.shift(1)

    return g


# ============================================================
# CATEGORY 6-7: VALUATION & FINANCIAL HEALTH
# ============================================================

def add_fundamental_features(g: pd.DataFrame) -> pd.DataFrame:
    """Placeholder - Requires fundamental data we don't have"""
    # We have DivYld12m and MktCap, but not P/E, P/B, etc.
    if 'DivYld12m' in g.columns:
        g['dividend_yield'] = g['DivYld12m'].shift(1)
    if 'MktCap' in g.columns:
        g['log_market_cap'] = np.log(g['MktCap'] + 1e-10).shift(1)

    return g


# ============================================================
# CATEGORY 9: RELATIVE STRENGTH & RANKINGS
# ============================================================

def add_cross_sectional_rankings(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-Sectional Ranks & Z-Scores (per Date)"""
    # Rankings (9 Features)
    for feat, ascending in [
        ('ret_1m', False), ('ret_3m', False), ('ret_6m', False), ('ret_12m', False),
        ('vol_20d', True), ('volume', False), ('dollar_volume', False),
        ('rsi_14', False), ('macd_line', False)
    ]:
        if feat in df.columns:
            df[f'rank_{feat}'] = df.groupby('Date')[feat].rank(method='average', ascending=ascending, pct=True)

    # Z-Scores (Cross-Sectional Spreads)
    for feat in ['ret_1m', 'ret_3m', 'ret_6m', 'vol_20d', 'rsi_14']:
        if feat in df.columns:
            cs_mean = df.groupby('Date')[feat].transform('mean')
            cs_std = df.groupby('Date')[feat].transform('std')
            df[f'{feat}_z_score'] = (df[feat] - cs_mean) / (cs_std + 1e-10)

    return df


# ============================================================
# CATEGORY 10: REGIME & MACRO - SHILLER FILTER
# ============================================================

def add_market_regime_features(df: pd.DataFrame, benchmark_col='Bench_Return_raw') -> pd.DataFrame:
    """Shiller Market-Regime-Filter (200-day MA + Volatility)"""

    # We need market-level data (SMI Index)
    # Approximate: Use benchmark returns to create market index
    if benchmark_col in df.columns:
        # Create market index per date (same for all stocks)
        market_ret = df.groupby('Date')[benchmark_col].first()
        market_idx = (1 + market_ret).cumprod() * 100  # Base = 100

        # Map back to df
        df['market_index'] = df['Date'].map(market_idx)

        # 200-day MA
        market_ma200 = market_idx.rolling(200, min_periods=100).mean()
        df['market_above_ma200'] = (df['Date'].map(market_idx > market_ma200)).astype(int)

        # Market Volatility
        market_vol_20 = market_ret.rolling(20, min_periods=10).std()
        market_vol_60 = market_ret.rolling(60, min_periods=30).std()

        df['market_vol_20d'] = df['Date'].map(market_vol_20)
        df['market_vol_60d'] = df['Date'].map(market_vol_60)

        # Volatility Regime (percentile)
        vol_percentile = market_vol_20.rolling(252, min_periods=126).rank(pct=True)
        df['market_vol_regime'] = df['Date'].map(vol_percentile)

        # VIX Proxy (simplified)
        df['vix_proxy'] = df['market_vol_20d'] * np.sqrt(252) * 100

        # Market Returns
        df['market_ret_1m'] = df['Date'].map(market_ret.rolling(21, min_periods=10).sum())
        df['market_ret_3m'] = df['Date'].map(market_ret.rolling(63, min_periods=30).sum())

    return df


# ============================================================
# CATEGORY 11: DERIVED & INTERACTIONS
# ============================================================

def add_interaction_features(g: pd.DataFrame) -> pd.DataFrame:
    """Interaction Terms, Polynomials, Log Transforms"""

    # Interaction Terms (8 Features)
    if 'mom_1m' in g.columns and 'vol_20d' in g.columns:
        g['mom_1m_times_vol'] = (g['mom_1m'] * g['vol_20d'])

    if 'mom_3m' in g.columns and 'volume_ratio_20d' in g.columns:
        g['mom_3m_times_volume_ratio'] = (g['mom_3m'] * g['volume_ratio_20d'])

    if 'rsi_14' in g.columns and 'bb_position' in g.columns:
        g['rsi_times_bb_position'] = (g['rsi_14'] * g['bb_position'])

    if 'macd_line' in g.columns and 'volume' in g.columns:
        g['macd_times_volume'] = (g['macd_line'] * g['volume'])

    if 'ret_1m' in g.columns and 'rank_ret_1m' in g.columns:
        g['ret_1m_times_rank'] = (g['ret_1m'] * g['rank_ret_1m'])

    if 'vol_20d' in g.columns and 'volume' in g.columns:
        g['vol_times_volume'] = (g['vol_20d'] * g['volume'])

    # Polynomial Features (5 Features)
    for feat in ['ret_1m', 'ret_3m', 'vol_20d', 'rsi_14']:
        if feat in g.columns:
            g[f'{feat}_squared'] = g[feat] ** 2

    if 'mom_3m' in g.columns:
        g['momentum_cubed'] = g['mom_3m'] ** 3

    # Logarithmic Transformations (4 Features)
    if 'volume' in g.columns:
        g['log_volume'] = np.log(g['volume'] + 1)
    if 'dollar_volume' in g.columns:
        g['log_dollar_volume'] = np.log(g['dollar_volume'] + 1)

    return g


# ============================================================
# SPECIALIZED FEATURES
# ============================================================

def add_specialized_features(g: pd.DataFrame, return_col='Return_raw', volume_col='Volume') -> pd.DataFrame:
    """Skewness, Kurtosis, Entropy, Autocorrelation"""
    r = g[return_col]
    v = g[volume_col] if volume_col in g.columns else None

    # Skewness & Kurtosis (4 Features)
    g['returns_skewness_60d'] = r.rolling(60, min_periods=30).skew().shift(1)
    g['returns_kurtosis_60d'] = r.rolling(60, min_periods=30).kurt().shift(1)
    g['rolling_skew_20d'] = r.rolling(20, min_periods=10).skew().shift(1)

    if v is not None:
        g['volume_skewness_60d'] = v.rolling(60, min_periods=30).skew().shift(1)

    # Entropy (3 Features) - Simplified as binned distribution
    def calc_entropy(series, bins=10):
        hist, _ = np.histogram(series.dropna(), bins=bins)
        hist = hist / (hist.sum() + 1e-10)
        entropy = -np.sum(hist * np.log(hist + 1e-10))
        return entropy

    g['price_entropy'] = g['Price'].rolling(60, min_periods=30).apply(calc_entropy, raw=False).shift(1)
    g['return_entropy'] = r.rolling(60, min_periods=30).apply(calc_entropy, raw=False).shift(1)

    if v is not None:
        g['volume_entropy'] = v.rolling(60, min_periods=30).apply(calc_entropy, raw=False).shift(1)

    # Autocorrelation (3 Features)
    def autocorr_lag(series, lag=1):
        return series.autocorr(lag=lag)

    g['return_autocorr_lag1'] = r.rolling(60, min_periods=30).apply(lambda x: autocorr_lag(pd.Series(x), 1), raw=False).shift(1)
    g['return_autocorr_lag5'] = r.rolling(60, min_periods=30).apply(lambda x: autocorr_lag(pd.Series(x), 5), raw=False).shift(1)

    if v is not None:
        g['volume_autocorr'] = v.rolling(60, min_periods=30).apply(lambda x: autocorr_lag(pd.Series(x), 1), raw=False).shift(1)

    return g


# ============================================================
# MASTER FUNCTION
# ============================================================

def engineer_comprehensive_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Master function to add ALL 180+ features
    """
    print(f"[INFO] Starting comprehensive feature engineering...")
    print(f"[INFO] Input shape: {df.shape}")

    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce', dayfirst=True)

    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)

    # Apply per-ticker transformations
    print("[INFO] Adding per-ticker features...")
    df = df.groupby('Ticker', group_keys=False).apply(lambda g: (
        add_return_features(g)
        .pipe(add_price_level_features)
        .pipe(add_momentum_features)
        .pipe(add_volatility_features)
        .pipe(add_volume_features)
        .pipe(add_technical_indicators)
        .pipe(add_fundamental_features)
        .pipe(add_interaction_features)
        .pipe(add_specialized_features)
    ))

    # Apply cross-sectional transformations
    print("[INFO] Adding cross-sectional rankings...")
    df = add_cross_sectional_rankings(df)

    # Apply market-level transformations
    print("[INFO] Adding market regime features (Shiller)...")
    df = add_market_regime_features(df)

    # Clean up infinities and extreme values
    print("[INFO] Cleaning infinities and NaNs...")
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = df[col].replace([np.inf, -np.inf], np.nan)
        # Clip extreme outliers (optional)
        # q01, q99 = df[col].quantile([0.01, 0.99])
        # df[col] = df[col].clip(q01, q99)

    print(f"[INFO] Final shape: {df.shape}")
    print(f"[INFO] Added {df.shape[1] - 121} new features!")

    return df


# ============================================================
# MAIN EXECUTION
# ============================================================

def main():
    input_file = Path("data/AMC_model_input.csv")
    output_file = Path("data/AMC_model_input.csv")
    backup_file = Path("data/AMC_model_input_BEFORE_COMPREHENSIVE.csv")

    print(f"[INFO] Loading {input_file}...")

    # Read with auto-detection
    try:
        df = pd.read_csv(input_file, sep=";")
        if df.shape[1] == 1:
            df = pd.read_csv(input_file)
    except:
        df = pd.read_csv(input_file)

    print(f"[INFO] Loaded: {df.shape}")

    # Backup
    if backup_file.exists():
        backup_file.unlink()
    df.to_csv(backup_file, index=False)
    print(f"[INFO] Backup created: {backup_file}")

    # Engineer features
    df_enhanced = engineer_comprehensive_features(df)

    # Save
    df_enhanced.to_csv(output_file, index=False)
    print(f"[INFO] Saved: {output_file}")
    print(f"[INFO] Features added: {df_enhanced.shape[1] - df.shape[1]}")

    # Summary
    new_cols = [c for c in df_enhanced.columns if c not in df.columns]
    print(f"\n[INFO] New features ({len(new_cols)}):")
    for i, col in enumerate(sorted(new_cols)[:30], 1):
        print(f"  {i:3d}. {col}")
    if len(new_cols) > 30:
        print(f"  ... and {len(new_cols) - 30} more")


if __name__ == "__main__":
    main()
