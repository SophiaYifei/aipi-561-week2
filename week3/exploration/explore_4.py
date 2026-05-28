import pandas as pd
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from validation.check_data_quality import DataQualityValidator

CUTOFF = pd.Timestamp("2026-01-16")
df = pd.read_parquet("week3/data/demand_enriched_corrupted.parquet")
baseline = df[df["time_bucket"] < CUTOFF]
corrupted = df[df["time_bucket"] >= CUTOFF]

validator = DataQualityValidator(baseline_df=baseline)

print("=== BASELINE ===")
r1 = validator.validate(baseline)
print(f"is_valid={r1['is_valid']}, num_issues={r1['num_issues']}")
for i in r1["issues"]:
    print(" ", i["severity"], i["type"], i["count"])

print("\n=== CORRUPTED ===")
r2 = validator.validate(corrupted)
print(f"is_valid={r2['is_valid']}, num_issues={r2['num_issues']}")
for i in r2["issues"]:
    print(" ", i["severity"], i["type"], i["count"])