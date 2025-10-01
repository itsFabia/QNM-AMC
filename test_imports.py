# Beispiel: SMI
SMI_TICKERS = [
    "ABBN.SW", "ALC.SW", "GEBN.SW", "GIVN.SW", "HOLN.SW", "KNIN.SW", "LOGN.SW", "LONN.SW", "NESN.SW", "NOVN.SW", "PGHN.SW", "CFR.SW", "ROG.SW", "SIKA.SW", "SOON.SW", "SLHN.SW", "SREN.SW", "SCMN.SW", "UBSG.SW", "ZURN.SW"
]
TOP_N = 5
START = "2010-01-01"  # großzügig; Lookback steuert das eigentliche Fenster
LOOKBACK_YEARS = 10
SMI_BENCH = "^SSMI"   # SMI Index in Yahoo

top5, smi_tr = compute_top_outperformers(
    index_members=SMI_TICKERS,
    benchmark_ticker=SMI_BENCH,
    start=START,
    lookback_years=LOOKBACK_YEARS,
    top_n=TOP_N,
)
print(top5)