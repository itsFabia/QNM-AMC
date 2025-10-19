import pandas as pd
df = pd.read_csv("data/AMC_model_input_reduced.csv", parse_dates=["Date"])

# 1) Erste 5 Preis-Datumswerte je Ticker (nur valide Preise)
price_col = "Price"
df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
valid = df[df[price_col].notna() & (df[price_col] > 0)].copy()

print(valid.groupby("Ticker")["Date"].min().sort_values())    # frühestes Datum je Ticker
print(valid[valid["Ticker"]=="AMRZ SE Equity"][["Date", price_col]].head(10))

# 2) Jahresabdeckung AMRZ
print(valid[valid["Ticker"]=="AMRZ SE Equity"]
      .assign(year=lambda x: x["Date"].dt.year)
      .groupby("year")[price_col].count())

# 3) Ticker, die erst sehr spät starten (z.B. > 2024-01-01)
late = valid.groupby("Ticker")["Date"].min()
print(late[late > "2024-01-01"].sort_values())

if __name__ == "__main__":
    main()