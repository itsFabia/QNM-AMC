# Sanity Check – AMC_model_input_reduced.csv

## 1. Struktur
- Zeilen: 50,300
- Spalten: 77
- Duplikate (Date+Ticker): 0
- Date-Parsing: ISO (%Y-%m-%d)

## 2. Fehlende Werte
- Spalten mit NaNs: 33
- Durchschnittlicher NaN-Anteil: 0.020

### Top 10 NaN-Spalten
| Spalte | NaN-Anteil |
|---|---|
| DivYld12m | 0.078827 |
| Volume_to_MA20 | 0.070577 |
| Return_Z20 | 0.070577 |
| Return_STD20 | 0.070577 |
| Return_MA20 | 0.070577 |
| Rank_LowVol20 | 0.070577 |
| Rank_MA20 | 0.070577 |
| Rank_Liquidity | 0.070179 |
| Volume_MA20 | 0.070179 |
| Return_STD10 | 0.068588 |

## 3. Ausreisser (>5σ)
- Spalten mit Ausreissern: 8
- Ø Ausreisser-Anteil: 0.001278

### Top 10 Ausreisser-Spalten
| Spalte | Anteil |
|---|---|
| SZCPIYOY Index | Last Price__logdiff1 | 0.022664 |
| ECCPEMUY Index | Last Price__logdiff1 | 0.017893 |
| EURR002W Index | Last Price__chgstd20 | 0.016700 |
| FDTR Index | Last Price__logdiff1 | 0.015507 |
| EURR002W Index | Last Price__logdiff1 | 0.015507 |
| Bench_Ret_5d_fwd | 0.003579 |
| Ret_5d_fwd | 0.002247 |
| Excess_5d_fwd | 0.001789 |
| Price | 0.000000 |
| Volume | 0.000000 |

## 4. Leak-Check (Korrelation mit Target)
| Spalte | Korrelation |
|---|---|
| Excess_5d_fwd | 1.000000 |
| Ret_5d_fwd | 0.805660 |
| Bench_Ret_5d_fwd | 0.039364 |
| SZCPIYOY Index | Last Price__lag1 | 0.032199 |
| MOVE Index | Last Price__chgstd20 | 0.028784 |
| MktCap | 0.025119 |
| FDTR Index | Last Price__chgstd20 | 0.021645 |
| ECCPEMUY Index | Last Price__chgstd20 | -0.021266 |
| DXY Curncy | Last Price__logdiff1 | -0.020910 |
| ECCPEMUY Index | Last Price__lag1 | 0.020515 |

## 5. Heatmap
- Datei: `reports/corr_heatmap_v2.png`