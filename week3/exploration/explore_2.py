import pandas as pd

CUTOFF = pd.Timestamp("2026-01-16")
df = pd.read_parquet("week3/data/demand_enriched_corrupted.parquet")
baseline = df[df["time_bucket"] < CUTOFF]
corrupted = df[df["time_bucket"] >= CUTOFF]

# Core numeric columns: the target plus the key features
cols = ["trip_count", "lag_1h", "lag_1day", "lag_1week",
        "roll_mean_1h", "roll_mean_1day", "zone_slot_baseline",
        "PULocationID", "hour", "is_holiday", "is_weekend"]

print("=== BASELINE describe ===")
print(baseline[cols].describe().T[["min", "mean", "max", "std"]])

print("\n=== CORRUPTED describe ===")
print(corrupted[cols].describe().T[["min", "mean", "max", "std"]])

# Duplicate-row check
print("\n=== DUPLICATES ===")
key = ["PULocationID", "time_bucket"]  # each zone + time bucket should be unique
print(f"baseline dup rows on {key}: {baseline.duplicated(subset=key).sum()}")
print(f"corrupted dup rows on {key}: {corrupted.duplicated(subset=key).sum()}")

# Value sets of the categorical columns
print("\n=== CATEGORICAL value sets ===")
for c in ["is_holiday", "is_weekend", "cbd_pricing_active", "is_airport_zone"]:
    print(f"{c}: baseline={sorted(baseline[c].unique())}, corrupted={sorted(corrupted[c].unique())}")