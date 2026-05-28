"""
Data quality validation for the incoming demand batch.

The validator runs three focused checks that map to the corruption actually
found in the incoming window: illegal trip_count values, duplicate keys, and a
categorical feature that collapsed to a constant. Thresholds are anchored to a
clean baseline segment when one is supplied, so the same code stays calibrated
as the underlying demand grows.
"""

import logging
import pandas as pd
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class DataQualityValidator:
    """Validates data against quality expectations."""

    # Columns that must be unique together (one zone has one row per time bucket)
    KEY_COLUMNS = ["PULocationID", "time_bucket"]
    # Sentinel value upstream systems use to mark bad/missing data
    SENTINEL_VALUE = 99999
    # Multiplier applied to the baseline max to flag extreme outliers
    OUTLIER_BASELINE_MULTIPLIER = 2.0
    # Hard fallback ceiling if no baseline is available
    DEFAULT_TRIP_COUNT_CEILING = 1000
    # Categorical features expected to vary (not be constant) in valid data
    EXPECTED_VARYING_FEATURES = ["cbd_pricing_active"]

    def __init__(self, baseline_df: Optional[pd.DataFrame] = None):
        """
        Args:
            baseline_df: Clean reference data for comparison. Optional —
                         absolute-rule checks work without it; distribution
                         checks become stricter when it is provided.
        """
        self.baseline = baseline_df
        self.issues: List[Dict] = []

    def validate(self, df: pd.DataFrame) -> Dict:
        """
        Run all validation checks.

        Returns:
            Dictionary with:
            - is_valid: boolean
            - num_issues: count of issues found
            - issues: list of issue details
        """
        self.issues = []

        self.check_value_ranges(df)      # Issue 1: trip_count illegal values
        self.check_duplicates(df)        # Issue 2: duplicate rows
        self.check_distributions(df)     # Issue 3: feature collapse

        return {
            "is_valid": len(self.issues) == 0,
            "num_issues": len(self.issues),
            "issues": self.issues,
        }

    def check_value_ranges(self, df: pd.DataFrame):
        """trip_count must be a non-negative count with no sentinel/outlier values."""
        if "trip_count" not in df.columns:
            self._add_issue(
                "missing_column", "critical",
                "Required column 'trip_count' is missing.", count=0,
            )
            return

        tc = df["trip_count"]

        # (a) negative values — a trip count can never be < 0
        neg = int((tc < 0).sum())
        if neg > 0:
            self._add_issue(
                "negative_value", "high",
                f"trip_count has {neg} negative value(s); a trip count cannot be < 0.",
                count=neg,
            )

        # (b) sentinel value — upstream marker for bad/missing data
        sentinel = int((tc == self.SENTINEL_VALUE).sum())
        if sentinel > 0:
            self._add_issue(
                "sentinel_value", "high",
                f"trip_count contains {sentinel} row(s) equal to the sentinel {self.SENTINEL_VALUE}.",
                count=sentinel,
            )

        # (c) extreme outliers — ceiling from baseline if available, else fixed
        if self.baseline is not None and "trip_count" in self.baseline.columns:
            ceiling = self.baseline["trip_count"].max() * self.OUTLIER_BASELINE_MULTIPLIER
        else:
            ceiling = self.DEFAULT_TRIP_COUNT_CEILING

        outliers = int(((tc > ceiling) & (tc != self.SENTINEL_VALUE)).sum())
        if outliers > 0:
            self._add_issue(
                "outlier", "medium",
                f"trip_count has {outliers} value(s) above the ceiling {ceiling:.0f}.",
                count=outliers,
            )

    def check_distributions(self, df: pd.DataFrame):
        """A feature expected to vary should not collapse to a single value."""
        for col in self.EXPECTED_VARYING_FEATURES:
            if col not in df.columns:
                continue
            if df[col].nunique(dropna=True) <= 1:
                only = df[col].dropna().unique().tolist()
                self._add_issue(
                    "constant_feature", "medium",
                    f"Feature '{col}' collapsed to a single value {only}; it carries no signal.",
                    count=len(df),
                )

    def check_duplicates(self, df: pd.DataFrame):
        """Each (PULocationID, time_bucket) pair should appear at most once."""
        if not all(c in df.columns for c in self.KEY_COLUMNS):
            return  # a missing key column is a schema problem, not a dup problem
        dups = int(df.duplicated(subset=self.KEY_COLUMNS).sum())
        if dups > 0:
            self._add_issue(
                "duplicate_rows", "high",
                f"{dups} duplicate row(s) on key {self.KEY_COLUMNS}.",
                count=dups,
            )

    def _add_issue(
        self,
        issue_type: str,
        severity: str,
        description: str,
        count: int = None,
        **details
    ):
        """Helper to add issue to list."""
        issue = {
            "type": issue_type,
            "severity": severity,  # 'critical', 'high', 'medium', 'low'
            "description": description,
            "count": count,
            **details,
        }
        self.issues.append(issue)


def validate_and_log(
    df: pd.DataFrame,
    baseline: Optional[pd.DataFrame] = None,
    log: Optional[logging.Logger] = None,
) -> Optional[Dict]:
    """Validate a batch and log any issues without ever raising.

    This is the graceful-degradation boundary: callers (the API at startup,
    the CI job) run it on incoming data and keep going regardless of outcome.
    A clean batch logs an INFO heartbeat; problems are logged at WARNING so
    operators notice; an unexpected failure inside validation is logged at
    ERROR and swallowed. Returns the result dict, or None if validation could
    not run.
    """
    log = log or logger
    try:
        result = DataQualityValidator(baseline_df=baseline).validate(df)
    except Exception as exc:
        log.error("Data quality validator raised an exception: %s", exc)
        return None

    if result["is_valid"]:
        log.info("Data quality check passed — no issues in %s row(s).", len(df))
        return result

    log.warning(
        "Data quality issues detected: %d issue(s) in %s row(s).",
        result["num_issues"], len(df),
    )
    for issue in result["issues"]:
        log.warning(
            "  [%s] %s: %s (count=%s)",
            issue["severity"].upper(),
            issue["type"],
            issue["description"],
            issue["count"],
        )
    return result
