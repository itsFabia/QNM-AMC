# Cleaning Log – AMC_model_input.csv
**Datum:** 2025-11-02 22:54  
**Zeilen/Spalten (vorher):** 50,300 / 316  
**Zeilen/Spalten (nachher):** 50,300 / 316  

## Struktur & Basis
- Duplikate (Date+Ticker): **0**
- Date-Parsen: **dd.mm.yyyy** via `dayfirst=True` gesetzt.

## Missing Values
- Fehlende Werte **vorher (gesamt):** 1,250,218
- Fehlende Werte **nachher (gesamt):** 1,062,438

### Top-NaN-Spalten vorher (Anteil; grobe Orientierung)
| Spalte | NaN-Anteil |
|---|---|
| money_flow_index | 1.000 |
| EURR002W Index | Last Price__chgstd20 | 0.639 |
| EURR002W Index | Last Price__logdiff1 | 0.636 |
| GSWISS10 Index | Last Price__chgstd20 | 0.579 |
| sortino_ratio_3m | 0.578 |
| GSWISS10 Index | Last Price__logdiff1 | 0.575 |
| semi_deviation | 0.528 |
| GDBR10 Index | Last Price__chgstd20 | 0.320 |
| GDBR10 Index | Last Price__logdiff1 | 0.314 |
| SZCPIYOY Index | Last Price__chgstd20 | 0.287 |

## Imputing & Glättung (Makro)
- Makro-Felder (heuristisch erkannt): 56 Spalten.
- Methode: **forward-fill pro Ticker** + **rolling mean (W=5)**.

## Outlier-Handling & Feature-Reduktion
- **WICHTIG**: Winsorizing und Feature-Reduktion wurden aus diesem Skript entfernt!
- **Grund**: Future Information Leakage vermeiden.
- **Neue Pipeline**: Beide Operationen werden jetzt per-split in `scale_and_save.py` durchgeführt,
  basierend ausschließlich auf Train-Daten.

## Optional: Heatmap
- Korrelations-Heatmap gespeichert: `corr_heatmap_after.png`

## Dateien (Outputs)
- Clean (nur Imputing): `data/AMC_model_input_clean.csv`
- Reduced (identisch zu Clean): `data/AMC_model_input_reduced.csv`
  - **Hinweis**: Beide Dateien sind jetzt identisch (Backward-Compatibility)

## Selbstkritik / Risiken
- Imputing kann Bias erzeugen, wenn Makro-Reihen lange Lücken haben (Regime-Übergänge).
- Forward-Fill ist konservativ, aber kann bei strukturellen Breaks problematisch sein.
- **UPDATE 2025**: Winsorizing/Feature-Reduktion wurden entfernt aus diesem Skript,
  um Future Information Leakage zu verhindern (jetzt per-split in scale_and_save.py).

## Nächste Schritte
1. Sanity-Check erneut laufen lassen (auf `*_reduced.csv`).
2. Erste Baseline-Modelle (z. B. RandomForest/XGBoost) mit Walk-Forward testen.
3. Erklärbarkeit: Permutation Importance/SHAP prüfen (Bias-Detektor).
4. Versionslog im Projekt aktualisieren (`reports/sanity_check_log.md`).