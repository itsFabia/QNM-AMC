# Neue Feature Engineering Strategie
**Datum:** 2025-11-02
**Status:** Implementiert, Testing ausstehend
**Kontext:** Nach Behebung des Future-Leakage-Bugs (IC von 70% auf 1.6%) neue Features entwickelt

---

## Hintergrund

Nach dem Entfernen des massiven Data Leakage (Ret_5d_fwd und Bench_Ret_5d_fwd waren als Features verfügbar) zeigte das Modell praktisch keine Predictive Power mehr:

**Performance OHNE Leakage:**
- IC Mean: 0.0161 (1.6%)
- Hit-Rate: 48.08% (unter Random 50%)
- Sharpe: -0.04 (negativ)
- CAGR: 0.85%

**Diagnose:** Die existierenden Features (basic Momentum, Volatility, Liquidity) reichen nicht aus, um echtes Alpha zu generieren.

---

## Neue Feature-Kategorien

Ich habe **24 neue Features** implementiert, die von 97 auf 121 Spalten erweitern:

### 1. Multi-Timeframe Momentum (3 Features)

**Motivation:** Verschiedene Zeithorizonte können unterschiedliche Signale liefern.

```python
for period, name in [(21, "1M"), (63, "3M"), (126, "6M")]:
    cum_ret = (1 + r).rolling(period, min_periods=period//2).apply(
        lambda x: x.prod() - 1, raw=True
    )
    g[f"Return_{name}"] = cum_ret.shift(1)
```

**Features:**
- `Return_1M`: 1-Monats kumulative Returns
- `Return_3M`: 3-Monats kumulative Returns
- `Return_6M`: 6-Monats kumulative Returns

**Rationale:** Momentum-Effekte können über verschiedene Zeiträume persistent sein. Research zeigt, dass 3-6 Monats Momentum oft prädiktiv ist.

---

### 2. Mean Reversion Features (4 Features)

**Motivation:** Überreaktionen und Umkehrungen erkennen.

```python
# RSI-like indicator
up = r.clip(lower=0).rolling(20, min_periods=10).sum()
down = (-r.clip(upper=0)).rolling(20, min_periods=10).sum()
rsi = 100 * up / (up + down + 1e-10)
g["RSI20"] = rsi.shift(1)

# Distance to Moving Averages
for w in [10, 20, 60]:
    ma = p.rolling(w, min_periods=w//2).mean()
    distance = ((p - ma) / ma).shift(1)
    g[f"DistMA{w}"] = distance
```

**Features:**
- `RSI20`: Relative Strength Index (20-day)
- `DistMA10/20/60`: Prozentuale Distanz zu MAs

**Rationale:** Mean-Reversion-Effekte, besonders nach starken Moves. RSI identifiziert overbought/oversold Zustände.

---

### 3. Trend Consistency (2 Features)

**Motivation:** Qualität des Trends messen, nicht nur Magnitude.

```python
for w in [10, 20]:
    pct_positive = (r > 0).rolling(w, min_periods=w//2).mean()
    g[f"PctPos{w}d"] = pct_positive.shift(1)
```

**Features:**
- `PctPos10d`: % positive Returns letzte 10 Tage
- `PctPos20d`: % positive Returns letzte 20 Tage

**Rationale:** Ein konsistenter Trend (hoher %) ist möglicherweise stabiler als ein volatiler Trend mit gleicher kumulativer Return.

---

### 4. Momentum Acceleration (1 Feature)

**Motivation:** Zweite Ableitung - beschleunigt oder verlangsamt sich Momentum?

```python
ma10 = r.rolling(10, min_periods=5).mean()
ma20 = r.rolling(20, min_periods=10).mean()
mom_accel = (ma10 - ma20).shift(1)
g["MomAccel"] = mom_accel
```

**Feature:**
- `MomAccel`: Differenz zwischen kurzfristigem und langfristigem Momentum

**Rationale:** Beschleunigendes Momentum kann stärkere Fortsetzung signalisieren.

---

### 5. Volatility-Adjusted Returns (2 Features)

**Motivation:** Risk-adjusted Performance ist oft aussagekräftiger als Raw Returns.

```python
# Volatility-adjusted returns
vol_adj_ret = (r / (r.rolling(20, min_periods=10).std(ddof=0) + 1e-10)).shift(1)
g["VolAdjRet20"] = vol_adj_ret

# Volatility-of-Volatility
vol = r.rolling(20, min_periods=10).std(ddof=0)
vol_of_vol = vol.rolling(20, min_periods=10).std(ddof=0).shift(1)
g["VolOfVol"] = vol_of_vol
```

**Features:**
- `VolAdjRet20`: Returns normalisiert durch Volatilität
- `VolOfVol`: Volatilität der Volatilität (Regime-Indikator)

**Rationale:** Gleiche Returns mit niedrigerer Vol sind attraktiver. VolOfVol identifiziert Regime-Wechsel.

---

### 6. Price-Volume Interactions (2 Features)

**Motivation:** Volume bestätigt oder widerspricht Preis-Moves.

```python
vol_pct_change = g[volume_col].pct_change()
# Price-Volume Correlation
pv_corr = r.rolling(20, min_periods=10).corr(vol_pct_change).shift(1)
g["PriceVol_Corr"] = pv_corr

# Volume Trend
vol_ma10 = g[volume_col].rolling(10, min_periods=5).mean()
vol_ma20 = g[volume_col].rolling(20, min_periods=10).mean()
g["VolTrend"] = (vol_ma10 / vol_ma20).shift(1)
```

**Features:**
- `PriceVol_Corr`: Korrelation zwischen Preis und Volumen
- `VolTrend`: Volume MA10/MA20 Ratio

**Rationale:** Preis-Moves mit hohem Volume sind meist nachhaltiger. Steigendes Volume kann Trend-Fortsetzung signalisieren.

---

### 7. Cross-Sectional Features (10 Features)

**Motivation:** Relative Position im Universum ist oft wichtiger als absolute Werte.

```python
# Rankings
for feat, ascending in [
    ("Return_1M", False), ("Return_3M", False), ("Return_6M", False),
    ("RSI20", False), ("VolAdjRet20", False), ("PctPos20d", False)
]:
    df[f"Rank_{feat}"] = df.groupby("Date")[feat].rank(
        method="average", ascending=ascending, pct=True
    )

# Cross-sectional spreads (Z-scores)
for feat in ["Return_1M", "Return_3M", "VolAdjRet20", "RSI20"]:
    cs_mean = df.groupby("Date")[feat].transform("mean")
    cs_std = df.groupby("Date")[feat].transform("std")
    df[f"{feat}_CSSpread"] = (df[feat] - cs_mean) / (cs_std + 1e-10)
```

**Features (Rankings):**
- `Rank_Return_1M/3M/6M`: Percentile Rankings für Momentum
- `Rank_RSI20`: RSI Ranking
- `Rank_VolAdjRet20`: Risk-adjusted Return Ranking
- `Rank_PctPos20d`: Trend Consistency Ranking

**Features (Z-Scores):**
- `Return_1M/3M_CSSpread`: Abweichung vom Cross-Sectional Mean
- `RSI20_CSSpread`: RSI relative zum Universum
- `VolAdjRet20_CSSpread`: Risk-adjusted Returns Z-Score

**Rationale:** Bei 20 Stocks ist relative Performance entscheidender als absolute. Rankings sind stabiler als Raw Values.

---

## Implementierung

### Dateien

**1. `scripts/add_new_features.py`** (NEU)
- Liest bestehende `AMC_model_input.csv`
- Fügt alle neuen Features hinzu
- Bewahrt Original-Date-Format
- Erstellt Backup

**2. `scripts/build_smi_features.py`** (MODIFIZIERT)
- Erweitert mit allen neuen Feature-Kategorien
- Für zukünftige Re-Generation von Grund auf

**3. Bestehende Pipeline** (UNVERÄNDERT)
- `clean_and_reduce.py`: Data cleaning
- `scale_and_save.py`: Per-split preprocessing
- `model_eval_lgbm.py`: Training & Evaluation

### Ausführung

```bash
# 1. Neue Features hinzufügen
python scripts/add_new_features.py

# 2. Pipeline ausführen
python scripts/clean_and_reduce.py \
  --input data/AMC_model_input.csv \
  --output data/AMC_model_input_reduced.csv \
  --min-samples 500

python scripts/scale_and_save.py \
  --csv data/AMC_model_input_reduced.csv \
  --splits-log reports/splits_log.csv \
  --outdir reports/scaled \
  --target Excess_5d_fwd \
  --winsorize --winsorize-sigma 3.0 \
  --winsorize-patterns "__logdiff1$,__chgstd20$" \
  --reduce-features --corr-threshold 0.95

python scripts/model_eval_lgbm.py \
  --parquet-dir reports/scaled \
  --target Excess_5d_fwd \
  --outdir reports/eval \
  --ranker --topk 10 --cost-bps 10.0
```

---

## Erwartete Verbesserungen

### Hypothesen

1. **Multi-Timeframe Momentum**: Sollte IC um 0.02-0.04 verbessern (auf ~5-6%)
2. **Cross-Sectional Features**: Bei nur 20 Stocks sollten Rankings stark sein (+0.03 IC)
3. **Mean Reversion**: Kann in Konsolidierungsphasen zusätzliches Alpha liefern
4. **Volatility-Adjusted**: Bessere Risk-Reward in Low-Vol Regimes

### Realistische Ziele

- **IC Mean**: Von 1.6% auf 4-7%
- **Hit-Rate**: Von 48% auf 52-55%
- **Sharpe**: Von -0.04 auf 0.5-1.0

### Wenn das nicht reicht...

Falls IC < 3% bleibt, weitere Schritte:

1. **Fundamentals Integration**
   - P/E, P/B Ratios
   - Earnings Surprises
   - Analyst Revisions

2. **Alternative Data**
   - News Sentiment
   - Social Media
   - Options Flow

3. **Macro Integration**
   - Interest Rate Regime
   - Sector Rotation Signals
   - Currency Flows (CHF wichtig für SMI)

4. **Ensemble Approaches**
   - Multiple Targets (1d, 5d, 20d)
   - Multiple Models (Linear, Tree, NN)
   - Meta-Learning

---

## Leakage-Safe Garantien

**Alle Features sind leakage-safe:**
- ✅ `.shift(1)` auf alle Features nach Rolling-Operations
- ✅ Keine `_fwd` Suffixe in Features
- ✅ Cross-sectional Operations nutzen nur contemporaneous data
- ✅ Per-split winsorizing/normalization in `scale_and_save.py`

**Verifizierung:**
```python
# Keine forward-looking Features
leaked_features = [c for c in df.columns if '_fwd' in c and c != target]
assert len(leaked_features) == 0
```

---

## Nächste Schritte

1. ✅ Features implementiert (24 neue Features)
2. ✅ Backup der alten Ergebnisse erstellt
3. ⏳ Pipeline ausführen mit neuen Features
4. ⏳ Performance-Vergleich: Alte vs Neue Features
5. ⏳ Feature Importance Analyse
6. ⏳ Entscheidung über weitere Feature-Kategorien

---

## Performance Tracking

### Baseline (OHNE neue Features)
```
IC Mean:       0.0161 (1.6%)
Hit-Rate:      48.08%
Sharpe:        -0.04
CAGR:          0.85%
Features:      72 (nach Leakage-Fix)
```

### Mit neuen Features (PENDING)
```
IC Mean:       TBD
Hit-Rate:      TBD
Sharpe:        TBD
CAGR:          TBD
Features:      ~96 (nach Feature Reduction)
```

---

## Lessons Learned

1. **Feature Engineering ist iterativ:** Nach Leakage-Fix mussten wir von Grund auf neu anfangen
2. **Quantität ≠ Qualität:** 24 neue Features, aber nur 10-15 werden wahrscheinlich wirklich nützlich sein
3. **Cross-Sectional ist key:** Bei kleinem Universum (20 Stocks) sind relative Measures kritisch
4. **Pipeline-Stabilität:** Wichtig, dass Feature-Engineering reproduzierbar und modular ist

---

## Code-Struktur

```
scripts/
├── add_new_features.py          # NEU: Fügt Features zu bestehenden Daten hinzu
├── build_smi_features.py        # MODIFIZIERT: Komplettes Feature Engineering von Grund auf
├── clean_and_reduce.py          # UNVERÄNDERT: Data cleaning
├── scale_and_save.py            # UNVERÄNDERT: Per-split preprocessing
└── model_eval_lgbm.py           # UNVERÄNDERT: Training & Evaluation

notebooks/thoughts_todos/
├── future_leakage_analysis_and_fix.md    # Leakage-Analyse
└── new_feature_engineering.md             # Dieses Dokument
```

---

**Fazit:** Systematischer Ansatz zur Feature-Erweiterung nach Leakage-Behebung. Fokus auf Cross-Sectional, Multi-Timeframe und Volatility-Adjusted Measures. Pipeline bleibt modular und leakage-safe.
