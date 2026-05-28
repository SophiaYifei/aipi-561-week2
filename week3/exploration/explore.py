import pandas as pd

CUTOFF = pd.Timestamp("2026-01-16")
df = pd.read_parquet("week3/data/demand_enriched_corrupted.parquet")

baseline = df[df["time_bucket"] < CUTOFF]
corrupted = df[df["time_bucket"] >= CUTOFF]

print("=== SHAPE ===")
print(f"total: {df.shape}, baseline: {baseline.shape}, corrupted: {corrupted.shape}")

print("\n=== COLUMNS & DTYPES ===")
print(df.dtypes)

print("\n=== DATE RANGE ===")
print(f"baseline: {baseline['time_bucket'].min()} -> {baseline['time_bucket'].max()}")
print(f"corrupted: {corrupted['time_bucket'].min()} -> {corrupted['time_bucket'].max()}")

print("\n=== NULL RATES (baseline) ===")
print(baseline.isna().mean())
print("\n=== NULL RATES (corrupted) ===")
print(corrupted.isna().mean())