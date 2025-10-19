# Outperformance-Strategie-Toolkit

Dieses Repository enthält ein kleines Toolkit, mit dem sich aus öffentlich
verfügbaren Kursdaten eine einfache Outperformance-Strategie bauen lässt:

1. **Langfristige Selektion** – wir bestimmen die stärksten Titel innerhalb
   eines Index (Total-Return) und überprüfen, ob sie über mehrere Jahre
   hinweg den Markt geschlagen haben.
2. **Kurzfristige Prognose** – für den besten Titel wird ein
   Machine-Learning-Modell (Logistic Regression) trainiert, das vorhersagt,
   ob die Aktie am nächsten Tag eine positive Excess-Return-Performance
   gegenüber dem Markt erzielt.
3. **Handelslogik & Backtest** – anhand der Prognosen entsteht eine einfache
   Handelsstrategie, die den Titel long (und implizit den Markt short über
   Beta-Hedging) handelt. Der Backtest liefert Kennzahlen wie CAGR, Sharpe
   Ratio und maximale Drawdowns.

## Verwendung


Der Standardlauf nutzt den Schweizer Leitindex SMI. Die zentralen Parameter
(wie Lookback-Periode, Anzahl der Top-Aktien, Schwellenwert für Signale
usw.) können über die Funktionen im Modul angepasst und wiederverwendet
werden.

> **Hinweis:** Für den Download der Kursdaten wird `yfinance` verwendet. Beim
> erstmaligen Ausführen ist daher eine Internetverbindung erforderlich.
