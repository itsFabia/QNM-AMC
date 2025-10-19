# To-Do Liste (laufend)  
**Stand:** 2025-10-19 – Cleaning abgeschlossen, Bias- und Modellphase offen  

---

## 4. **Bias-Review**  
- Indexverteilung je `Ticker` (`value_counts`) zur Erkennung von Over-/Underrepresentation.  
- Zeitlich rollende Cross-Validation statt zufälliger Splits verwenden.  

---

## 5. **Erklärbarkeit & Monitoring**  
- SHAP / Permutation Importance **vorbereiten** → Bias in Feature Importance sichtbar machen.  
- Modellversionierung einführen (`model_v2_cleaned`).  

---

## 6. **Dokumentation**  
- Sanity Check Log V2 **nach Cleaning aktualisieren**.  
- Bias-Notizen fortführen und im Repository dokumentieren.  

---

# ✅ Done  

## **Makro-Imputing**  
- Forward-Fill + Rolling Mean (W=5) implementiert.  
- Keine Backfill-Operation → kein Look-Ahead.  
- Frequenz der Makrodaten unverändert (keine Hochfrequenz-Interpolation).  

---

## **Outlier-Handling**  
- Winsorizing ±3 SD nur für Makrovariablen (`__logdiff1`, `__chgstd20`).  
- Extremwerte erfolgreich geglättet, ohne relevante Features zu verlieren.  

---

## **Feature-Pruning**  
- Stark korrelierte Features (>0.95) entfernt.  
- Targets und IDs ausgeschlossen.  
- Ergebnisse im Cleaning-Log dokumentiert (`reports/cleaning_log.md`).  

---

## **Dokumentation**  
- `cleaning_log.md` erstellt und mit Methoden, Risiken und Ergebnissen ergänzt.  
- Heatmap (`corr_heatmap_after.png`) gespeichert.  
- Pipeline-Status im Repository nachvollziehbar.  
