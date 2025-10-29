import pandas as pd

train.groupby("Ticker")[["Price","Volume","MktCap"]].agg(["mean","std"]).head(10)


if __name__ == "__main__":
    main()