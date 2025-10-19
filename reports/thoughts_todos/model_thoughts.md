# Model Notes & To-Do List  
**Datum:** 2025-10-19  
**Autor:** Fabia Holzer  

---

## 1. Beobachtungen aus der Korrelation
- Starke Korrelationen zwischen `Price`, `MktCap`, und allen `Return_lag*` → potenziell redundante Information.  
- Makrovariablen zeigen teils schwache, aber systematische Muster (z. B. ähnliche Bewegungen zwischen `GDBR10` und `GSWISS10`).  
- Gefahr: Modell könnte unbewusst Makrotrends statt Titel-Spezifika lernen.  

---

## 2. Erste Bias-Überlegungen
- **Survivorship Bias:** Falls nur aktuelle Titel enthalten sind, fehlen delistete Aktien → verzerrt Outperformance.  
- **Look-ahead Bias:** scheint sauber, aber bleibt im Auge zu behalten, falls neue Features hinzukommen.  
- **Frequency Bias:** Makrodaten mit Monats- oder Quartalstakt glätten die Volatilität → kann „falsche Ruhe“ im Modell erzeugen.  
- **Overfitting-Risiko:** hohe Feature-Redundanz bei Returns und MAs → Drop oder PCA prüfen.  
- **Regime Bias:** 10-Jahres-Fenster deckt ungleich viele Bullenjahre → Backtests mit Subperioden nötig.  

---

## 3. To-Do Liste (laufend)
1. **Feature Cleaning**
   - Korrelierte Features > 0.95 prüfen und ggf. reduzieren.  
   - Makrodaten glätten, aber Frequenz beibehalten (keine zu starke Interpolation).  
2. **Data Integrity**
   - Überprüfen, ob alle Ticker durchgehend Daten haben.  
   - Fehlende Makro-Werte via forward-fill + rolling mean testen.  
3. **Bias-Tests**
   - Separate Backtests nach Marktphasen (bull/bear).  
   - Cross-Validation zeitlich rollend, nicht zufällig.  
4. **Erklärbarkeit**
   - SHAP oder Permutation Importance vorbereiten → Bias-Erkennung in Feature Importance.  
5. **Dokumentation**
   - Sanity Check Log aktualisieren (Version 2 nach Cleaning).  

---

_Notiz:_  
„Bias taucht nicht auf, weil man Fehler macht – sondern weil man Muster sucht, wo keine sind. Also immer Gegenbeweis prüfen.“  
