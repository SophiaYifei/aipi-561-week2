import pandas as pd
CUTOFF = pd.Timestamp("2026-01-16")
df = pd.read_parquet("week3/data/demand_enriched_corrupted.parquet")
corrupted = df[df["time_bucket"] >= CUTOFF]

print("Issue1 negative trip_count:", (corrupted["trip_count"] < 0).sum())
print("Issue1 trip_count==99999 :", (corrupted["trip_count"] == 99999).sum())
print("Issue1 trip_count>1000   :", (corrupted["trip_count"] > 1000).sum())
print("Issue2 duplicate rows    :", corrupted.duplicated(subset=["PULocationID","time_bucket"]).sum())
print("Issue3 cbd unique vals   :", sorted(corrupted["cbd_pricing_active"].unique()))