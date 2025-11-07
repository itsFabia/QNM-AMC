# Production Model Documentation
**Model Version:** v1.0-top10
**Date:** 2025-11-03
**Status:** ✅ Production-Ready

---

## Executive Summary

**Recommended Model:** Top-10 Features with LightGBM Ranker

**Performance Metrics (Walk-Forward CV, 57 Splits, 2015-2025):**
- **Sharpe Ratio:** +0.533
- **CAGR:** +4.76%
- **Information Coefficient:** 0.08%
- **Hit-Rate:** 47.08%
- **Max Drawdown:** -5.25%
- **Avg Turnover:** 24.46%

**Key Strengths:**
- ✅ Positive Sharpe Ratio (solidly positive)
- ✅ Controlled Drawdown (<6%)
- ✅ No Look-Ahead Bias
- ✅ Robust Feature Selection (199 → 10 features)
- ✅ 10 years historical validation

---

## Model Features

### Top-10 Features (Exact List)

| # | Feature | Category | Description |
|---|---------|----------|-------------|
| 1 | **DivYld12m** | Fundamental | 12-month Dividend Yield |
| 2 | **ret_ytd** | Momentum | Year-to-Date Return |
| 3 | **vol_persistence** | Volatility | Persistence of Volatility Regime |
| 4 | **EURCHF Curncy \| Last Price__chgstd20** | Macro FX | EUR/CHF 20d Standardized Change |
| 5 | **GDBR10 Index \| Last Price__lag1** | Macro Rates | German Bund 10Y Yield (1d lag) |
| 6 | **vol_spike_indicator** | Volatility | Volatility Spike Detection |
| 7 | **MOVE Index \| Last Price__lag1** | Macro Vol | Bond Volatility Index (1d lag) |
| 8 | **market_ret_1m** | Momentum | Market Return 1-Month |
| 9 | **Rank_MA20** | Cross-Sectional | Rank of 20d Moving Average |
| 10 | **GSWISS10 Index \| Last Price__lag1** | Macro Rates | Swiss 10Y Yield (1d lag) |

**Feature Categories Breakdown:**
- **Macro Features (Rates, FX, Vol):** 4/10 (40%)
- **Momentum Features:** 2/10 (20%)
- **Volatility Features:** 2/10 (20%)
- **Cross-Sectional Ranking:** 1/10 (10%)
- **Fundamental:** 1/10 (10%)

**Key Insight:** Macro-economic features (interest rates, FX, bond volatility) are dominant for SMI stock selection. Swiss market is highly influenced by global macro environment.

---

## Model Configuration

### LightGBM Hyperparameters (Current)

```python
params = {
    'objective': 'lambdarank',
    'metric': 'ndcg',
    'ndcg_eval_at': [10],
    'learning_rate': 0.05,
    'num_leaves': 31,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'lambda_l1': 0.1,
    'lambda_l2': 0.1,
    'min_data_in_leaf': 50,
    'num_boost_round': 100,
    'early_stopping_rounds': 10,
    'verbose': -1
}
```

**Note:** These are default hyperparameters. No hyperparameter tuning has been performed to avoid over-optimization.

### Preprocessing Pipeline

1. **Data Cleaning** (`scripts/clean_and_reduce.py`)
   - Minimum samples per feature: 500
   - Forward-fill macro features (5-day rolling mean)
   - Remove features with >50% missing values

2. **Feature Selection** (`scripts/feature_selection.py`)
   - LightGBM importance-based selection
   - Aggregated over first 10 training splits
   - Top-10 features selected

3. **Per-Split Preprocessing** (`scripts/scale_and_save.py`)
   - **Winsorization:** 3-sigma clipping on `__logdiff1` and `__chgstd20` patterns
   - **Correlation Reduction:** Remove features with >0.95 Pearson correlation
   - **Standardization:** RobustScaler (median/IQR) fit on train-only
   - **Result:** 10 features → 15 features after correlation check

4. **Walk-Forward Cross-Validation**
   - Training window: 3 years (rolling)
   - Test period: 1 month
   - Number of splits: 57
   - Timeframe: 2015-01-01 to 2025-11-03

### Portfolio Construction

- **Ranking:** LightGBM predicts excess returns → rank stocks
- **Selection:** Top-10 ranked stocks selected
- **Weighting:** Equal-weight within Top-10
- **Rebalancing:** Monthly (beginning of month)
- **Transaction Costs:** 10 bps (0.1%) per trade

---

## Expected Performance (Out-of-Sample)

### Realistic Expectations

Based on 10-year backtest and accounting for potential degradation:

| Metric | Backtest | Conservative Estimate | Reason |
|--------|----------|----------------------|--------|
| **Sharpe Ratio** | +0.53 | **+0.30 to +0.50** | Some regime sensitivity |
| **CAGR** | +4.76% | **+3% to +5%** | Market-dependent |
| **Max Drawdown** | -5.25% | **-8% to -12%** | Backtest may underestimate |
| **Hit-Rate** | 47.08% | **45% to 50%** | Near random for monthly |
| **Turnover** | 24.46% | **25% to 30%** | Slightly higher in live |

**Key Risks:**
- ⚠️ Performance dependent on macro regime (interest rate environment)
- ⚠️ Small universe (20 stocks) → concentration risk
- ⚠️ Sharpe +0.53 is good but not exceptional
- ⚠️ IC 0.08% is very low → weak signal

---

## Monitoring & Maintenance

### Key Metrics to Monitor (Monthly)

1. **Performance Metrics**
   - Rolling 12M Sharpe Ratio
   - Rolling 6M IC (Information Coefficient)
   - Cumulative P&L vs Benchmark
   - Drawdown from Peak

2. **Feature Drift**
   - Distribution shifts in Top-10 features
   - Missing data rates
   - Correlation matrix stability

3. **Model Behavior**
   - Average predicted excess returns
   - Prediction confidence (std of predictions)
   - Feature importances (monthly recompute)

### Monitoring Thresholds (Red Flags)

| Metric | Threshold | Action |
|--------|-----------|--------|
| Rolling 12M Sharpe | < -0.2 for 3 months | **STOP TRADING** |
| Rolling 6M IC | < -0.5% for 2 months | Investigate |
| Drawdown | > -15% | Reduce position size |
| Missing data | > 20% for any Top-10 feature | Alert |
| Feature correlation | >0.98 for any pair | Review features |

### Retraining Schedule

**Frequency:** Monthly (after new test month)

**Process:**
1. **Data Update:** Add latest month of data
2. **Feature Validation:** Check for missing/anomalous values
3. **Retrain Model:** Use expanded training window
4. **Backtest Latest Split:** Validate on most recent month
5. **Deploy:** If validation IC > -1%, deploy new model

**Note:** Do NOT retune hyperparameters without extensive out-of-sample validation to avoid overfitting.

---

## Position Sizing & Risk Management

### Recommended Position Sizing

**Conservative Approach (Recommended for first 6 months):**
- **Max position size:** 10% per stock (equal-weight Top-10)
- **Kelly Fraction:** 0.5x (half Kelly for safety)
- **Max leverage:** 1.0x (no leverage)
- **Cash buffer:** 10% (hold 10% cash as buffer)

**Calculation:**
```python
portfolio_value = 1,000,000  # CHF
n_stocks = 10
position_size = portfolio_value * 0.90 / n_stocks  # 90k CHF per stock
```

**After 6 Months (if Sharpe > +0.3):**
- Consider increasing to 1.0x Kelly
- Reduce cash buffer to 5%

### Stop-Loss Rules

**Position-Level:**
- Individual stock loss > -20% → Sell immediately
- Sector concentration > 40% → Reduce

**Portfolio-Level:**
- Total portfolio drawdown > -10% → Reduce position size to 50%
- Total portfolio drawdown > -15% → **STOP TRADING**

---

## Fallback Strategy

### Triggers for Fallback

If any of the following occur:
1. Rolling 12M Sharpe < -0.2 for 3 consecutive months
2. Total drawdown > -15%
3. Data feed failure for >3 days
4. Model prediction confidence collapses (std < 0.01%)

### Fallback Action

**Switch to Equal-Weight SMI Portfolio:**
- Buy all 20 SMI stocks in equal weights
- Rebalance quarterly
- Hold until model performance recovers

**Rationale:** Equal-weight SMI is a conservative fallback with low tracking error to benchmark.

---

## Production Deployment Checklist

### Pre-Deployment

- [ ] Verify all Top-10 features available in production data feed
- [ ] Test feature pipeline on latest month
- [ ] Backtest most recent 3 months (out-of-sample)
- [ ] Set up monitoring dashboard (Grafana/Tableau)
- [ ] Document emergency procedures
- [ ] Test rebalancing logic (paper trading)

### Deployment

- [ ] Start with 50% capital allocation (first month)
- [ ] Increase to 75% after 1 month (if Sharpe > 0)
- [ ] Increase to 100% after 3 months (if Sharpe > +0.2)

### Post-Deployment

- [ ] Daily P&L monitoring
- [ ] Weekly feature drift check
- [ ] Monthly performance review
- [ ] Quarterly model retraining

---

## Known Limitations

1. **Small Universe**
   - Only 20 SMI stocks
   - High concentration risk
   - Limited diversification

2. **Weak Signal**
   - IC 0.08% is very low
   - Near-random stock selection
   - High variance in monthly performance

3. **Macro Dependency**
   - 40% features are macro (rates, FX, vol)
   - Performance tied to interest rate regime
   - May underperform in low-volatility environments

4. **No Fundamentals**
   - No P/E, P/B, ROE, etc.
   - Missing company-specific information
   - Relies heavily on market timing

5. **Backtest vs Reality**
   - 10 bps transaction costs may be optimistic
   - Market impact not modeled
   - Slippage in illiquid names

---

## Change Log

| Version | Date | Changes |
|---------|------|---------|
| v1.0-top10 | 2025-11-03 | Initial production model with Top-10 features |

---

## Contact & Escalation

**Model Owner:** [Your Name]
**Email:** [your.email@domain.com]
**Emergency Contact:** [emergency.contact@domain.com]

**Escalation Path:**
1. Performance degradation → Alert Model Owner
2. Data feed failure → Alert IT Operations
3. Stop-Loss triggered → Alert Risk Management
4. Regulatory concerns → Alert Compliance

---

## Appendix: Reproduction Commands

### Full Pipeline (from scratch)

```bash
# 1. Feature Selection
python scripts/feature_selection.py \
  --parquet-dir reports/scaled \
  --input-csv data/AMC_model_input_reduced.csv \
  --output-csv data/AMC_model_input_top10.csv \
  --top-k 10 --n-splits 10 --target Excess_5d_fwd

# 2. Preprocessing & Scaling
python scripts/scale_and_save.py \
  --csv data/AMC_model_input_top10.csv \
  --splits-log reports/splits_log.csv \
  --outdir reports/scaled_top10 \
  --target Excess_5d_fwd --winsorize --winsorize-sigma 3.0 \
  --winsorize-patterns "__logdiff1$,__chgstd20$" \
  --reduce-features --corr-threshold 0.95

# 3. Model Training & Evaluation
python scripts/model_eval_lgbm.py \
  --parquet-dir reports/scaled_top10 \
  --target Excess_5d_fwd \
  --outdir reports/eval_top10 \
  --ranker --topk 10 --cost-bps 10.0
```

### Performance Reports

- **Evaluation Summary:** `reports/eval_top10/eval_summary.csv`
- **Equity Curve:** `reports/eval_top10/equity_curve_all.png`
- **IC by Split:** `reports/eval_top10/ic_by_split.png`
- **Hit Rate by Split:** `reports/eval_top10/hit_rate_by_split.png`

---

**Last Updated:** 2025-11-03
**Next Review:** 2025-12-01
**Status:** ✅ Production-Ready with Conservative Position Sizing
