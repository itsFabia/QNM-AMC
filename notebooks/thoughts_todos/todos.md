# To-Do Liste (laufend)  
**Stand:** 2025-10-19 – Bias-Review abgeschlossen, Modellierung offen  

---

## 5. **Modellphase (nächster Schritt)**  
- Rolling/Expanding Window CV mit Purge & Embargo (5D).  
- Trainingsstart ab **2021-01-01** für faire Vergleichbarkeit.  
- Pro-Ticker Standardisierung (`log1p` + z-Score im Train-Split).  
- Cold-Start-Policy für **ALC** und **AMRZ** (Score erst ab N ≥ 250).  

---

## 6. **Erklärbarkeit & Monitoring**  
- SHAP / Permutation Importance implementieren.  
- Feature Drift & mean_shift_sigma je Fold tracken.  
- Modellversionierung starten (`model_v2_bias_checked`).  

---

## 7. **Dokumentation**  
- Bias Review 2025-10-19 ins Repository aufnehmen (`reports/bias_review.md`).  
- Fold-Report-Template ergänzen (Feature Drift, OOS-IR, Spearman).  
- README um Pipeline-Übersicht erweitern (Clean → Bias → Model).  

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

## **Bias-Review**  
- 20 SMI-Titel geprüft, gleichverteilt (5 % je Ticker).  
- Level-Bias erkannt (MktCap, Volume, DivYld).  
- IPO-Effekt bei **ALC** & **AMRZ** hervorgehoben.  
- Empfehlungen in To-Do integriert.  

---

## **Dokumentation**  
- `cleaning_log.md` aktualisiert (Methoden, Risiken, Ergebnisse).  
- Heatmap (`corr_heatmap_after.png`) gespeichert.  
- Repository-Struktur nachvollziehbar (Clean → Bias → Model).  
