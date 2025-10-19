# Model Notes & To-Do List  
**Datum:** 2025-10-19  
**Autor:** Fabia Holzer  

---

## 1. Beobachtungen aus Sanity Check v2  
- **Struktur:** 50 300 Zeilen, 77 Spalten, keine Duplikate (`Date+Ticker`), Datumsformat korrekt (ISO `%Y-%m-%d`).  
- **NaNs:** Ø 2 %, hauptsächlich in Rolling-Fenstern (`MA20`, `STD20`, `Z20`) – technisch erklärbar.  
- **Ausreißer:** 8 Spalten > 5 σ, v. a. Makroserien mit `logdiff1` oder `chgstd20`. Kein akuter Handlungsbedarf, aber Winsorizing empfohlen.  
- **Leak-Check:** keine auffälligen Korrelationen mit `Excess_5d_fwd`, kein Look-Ahead.  
- **Multikollinearität:** deutliche Abhängigkeiten zwischen `Price`, `MktCap`, `Return_MA*`, `Return_STD*`, `Return_lag*`.  
- **Heatmap:** Clusterbildung bei Return-Familien und Makro-Indizes (`GDBR10`, `GSWISS10`).

---

## 2. Erste Bias-Überlegungen  
- **Survivorship Bias:** Delistete Titel fehlen → mögliche Überschätzung der Outperformance.  
- **Frequency Bias:** Makrodaten (monatlich/vierteljährlich) glätten Volatilität → mögliche Fehleinschätzung von Risiken.  
- **Overfitting-Risiko:** hohe Feature-Redundanz → Feature-Pruning oder PCA prüfen.  
- **Regime Bias:** Trainingszeitraum enthält überwiegend Bullenphasen → Subperioden-Backtests nötig.  
- **Look-Ahead Bias:** aktuell keiner erkennbar, aber nach Makro-Imputation erneut prüfen.

---

## 3. To-Do Liste (laufend)  

1. **Makro-Imputing**  
   - Forward-Fill + Rolling Mean testen.  
   - Keine künstliche Hochfrequenz-Interpolation.  

2. **Outlier-Handling**  
   - Winsorize ± 3 SD für Makrovariablen (`logdiff1`, `chgstd20`).  

3. **Feature Pruning**  
   - Features mit ρ > 0.95 entfernen (`Price`, `MktCap`, `Return_*`).  
   - Prüfen, ob semantisch wichtige Variablen erhalten bleiben.  

4. **Bias-Review**  
   - Indexverteilung je Ticker (`value_counts`) zur Erkennung von Over-/Underrepresentation.  
   - Zeitlich rollende Cross-Validation statt zufälliger Splits verwenden.  

5. **Erklärbarkeit & Monitoring**  
   - SHAP/Permutation Importance vorbereiten → Bias in Feature Importance sichtbar machen.  
   - Modellversionierung einführen (`model_v2_cleaned`).  

6. **Dokumentation**  
   - Sanity Check Log V2 nach Cleaning aktualisieren.  
   - Bias-Notizen fortführen und im Repository dokumentieren.  
