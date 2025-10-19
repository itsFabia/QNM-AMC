To-Do Liste (laufend)  

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


Done 