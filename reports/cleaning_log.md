# Cleaning Log – AMC_model_input.csv
**Datum:** 2025-10-19 15:45  
**Zeilen/Spalten (vorher):** 50,300 / 97  
**Zeilen/Spalten (nachher):** 50,300 / 77  

## Struktur & Basis
- Duplikate (Date+Ticker): **50280**
- Date-Parsen: **dd.mm.yyyy** via `dayfirst=True` gesetzt.

## Missing Values
- Fehlende Werte **vorher (gesamt):** 361,780
- Fehlende Werte **nachher (gesamt):** 161,496

### Top-NaN-Spalten vorher (Anteil; grobe Orientierung)
| Spalte | NaN-Anteil |
|---|---|
| Date | 1.000 |
| EURR002W Index | Last Price__chgstd20 | 0.639 |
| EURR002W Index | Last Price__logdiff1 | 0.636 |
| GSWISS10 Index | Last Price__chgstd20 | 0.579 |
| GSWISS10 Index | Last Price__logdiff1 | 0.575 |
| GDBR10 Index | Last Price__chgstd20 | 0.320 |
| GDBR10 Index | Last Price__logdiff1 | 0.314 |
| SZCPIYOY Index | Last Price__chgstd20 | 0.287 |
| SZCPIYOY Index | Last Price__logdiff1 | 0.285 |
| GSWISS20 Index | Last Price__chgstd20 | 0.275 |

## Imputing & Glättung (Makro)
- Makro-Felder (heuristisch erkannt): 56 Spalten.
- Methode: **forward-fill pro Ticker** + **rolling mean (W=5)**.

## Outlier-Handling
- Winsorizing auf **±3 SD** für Feature-Spalten (Targets unverändert).

## Feature-Reduktion
- Korrelationsschwelle: **>0.95** (absolut).
- Entfernte Features: **20**
  - Beispiele: RealizedVol20_ann, DivYld12m_lag1, MktCap_lag1, CPURNSA Index | Last Price__lag5, DXY Curncy | Last Price__lag5, ECCPEMUY Index | Last Price__lag5, EURCHF Curncy | Last Price__lag5, EURR002W Index | Last Price__lag5, FDTR Index | Last Price__lag5, GDBR10 Index | Last Price__lag5, GSWISS10 Index | Last Price__lag5, GSWISS20 Index | Last Price__lag1 … (+8 weitere)

## Optional: Heatmap
- Korrelations-Heatmap gespeichert: `corr_heatmap_after.png`

## Dateien (Outputs)
- Clean (Imputing + Winsorizing): `data/AMC_model_input_clean.csv`
- Reduced (zzgl. Feature-Drop): `data/AMC_model_input_reduced.csv`

## Selbstkritik / Risiken
- Imputing kann Bias erzeugen, wenn Makro-Reihen lange Lücken haben (Regime-Übergänge).
- Korrelations-Drop ist heuristisch: kann nützliche, aber redundante Signale kappen.
- Winsorizing glättet Schocks – gut für Stabilität, aber reduziert Extrem-Alpha.
- Date-Parsen: **ISO %Y-%m-%d** (kein dayfirst).
- Winsorizing: **±3 SD** nur auf Makro-Diff/Vol-Spalten (`__logdiff1`, `__chgstd20`).

## Nächste Schritte
1. Sanity-Check erneut laufen lassen (auf `*_reduced.csv`).
2. Erste Baseline-Modelle (z. B. RandomForest/XGBoost) mit Walk-Forward testen.
3. Erklärbarkeit: Permutation Importance/SHAP prüfen (Bias-Detektor).
4. Versionslog im Projekt aktualisieren (`reports/sanity_check_log.md`).