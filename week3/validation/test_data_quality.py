"""
Data quality validation tests.

These run on synthetic data only, so they exercise the validation logic
without needing the real (gitignored) parquet — which is what lets them run in
CI. They cover three things: clean data passes, each corruption is detected,
and the logging boundary degrades gracefully instead of crashing.
"""

import logging

import pytest
import pandas as pd

from validation.check_data_quality import DataQualityValidator, validate_and_log

def make_clean_df(n_zones: int = 5, n_buckets: int = 20) -> pd.DataFrame:
    """Build a small, clean synthetic dataset that passes all checks.

    One row per (zone, time_bucket). trip_count is a realistic small count.
    cbd_pricing_active intentionally varies so it is NOT flagged as constant.
    """
    rows = []
    base_time = pd.Timestamp("2026-01-01")
    for zone in range(1, n_zones + 1):
        for b in range(n_buckets):
            rows.append({
                "PULocationID": zone,
                "time_bucket": base_time + pd.Timedelta(minutes=15 * b),
                "trip_count": 10 + (b % 7),          # small, non-negative
                "cbd_pricing_active": b % 2,          # alternates 0/1 -> varies
            })
    return pd.DataFrame(rows)


@pytest.fixture
def baseline_data():
    """Clean baseline dataframe used as the reference segment."""
    return make_clean_df()


@pytest.fixture
def corrupted_data():
    """A batch carrying every corruption at once."""
    df = make_clean_df().copy()
    # Issue 1a: negative trip_count
    df.loc[0, "trip_count"] = -5
    # Issue 1b: sentinel value
    df.loc[1, "trip_count"] = 99999
    # Issue 2: duplicate (PULocationID, time_bucket)
    dup = df.iloc[[2]].copy()
    df = pd.concat([df, dup], ignore_index=True)
    # Issue 3: collapse cbd_pricing_active to a single value
    df["cbd_pricing_active"] = 1
    return df



@pytest.fixture
def validator(baseline_data):
    """Validator initialized with the clean baseline."""
    return DataQualityValidator(baseline_df=baseline_data)

@pytest.fixture
def df_with_negative():
    df = make_clean_df().copy()
    df.loc[0, "trip_count"] = -1
    return df


@pytest.fixture
def df_with_sentinel():
    df = make_clean_df().copy()
    df.loc[0, "trip_count"] = 99999
    return df


@pytest.fixture
def df_with_duplicates():
    df = make_clean_df().copy()
    return pd.concat([df, df.iloc[[0]]], ignore_index=True)


@pytest.fixture
def df_with_constant_feature():
    df = make_clean_df().copy()
    df["cbd_pricing_active"] = 1  # collapse to a single value
    return df


class TestBaselineData:
    """Tests that baseline data should pass validation."""

    def test_baseline_passes_validation(self, baseline_data, validator):
        """Baseline data should have no quality issues."""
        result = validator.validate(baseline_data)
        assert result["is_valid"], f"Baseline incorrectly flagged: {result['issues']}"
        assert result["num_issues"] == 0


class TestDataQualityIssues:
    """Tests that verify each issue is detected."""

    def _issue_types(self, result):
        return {i["type"] for i in result["issues"]}

    def test_detect_negative_trip_count(self, df_with_negative, validator):
        """Issue 1a: trip_count < 0 must be flagged as negative_value."""
        result = validator.validate(df_with_negative)
        assert not result["is_valid"]
        assert "negative_value" in self._issue_types(result)

    def test_detect_sentinel_value(self, df_with_sentinel, validator):
        """Issue 1b: trip_count == 99999 must be flagged as sentinel_value."""
        result = validator.validate(df_with_sentinel)
        assert not result["is_valid"]
        assert "sentinel_value" in self._issue_types(result)

    def test_detect_duplicates(self, df_with_duplicates, validator):
        """Issue 2: duplicate (zone, time_bucket) rows must be flagged."""
        result = validator.validate(df_with_duplicates)
        assert not result["is_valid"]
        types = self._issue_types(result)
        assert "duplicate_rows" in types
        # And: a plain duplication should NOT trip other checks.
        assert "negative_value" not in types
        assert "sentinel_value" not in types

    def test_detect_constant_feature(self, df_with_constant_feature, validator):
        """Issue 3: a feature that collapses to one value must be flagged."""
        result = validator.validate(df_with_constant_feature)
        assert not result["is_valid"]
        assert "constant_feature" in self._issue_types(result)

    def test_detects_multiple_issues_at_once(self, corrupted_data, validator):
        """When several issues coexist, validator reports each independently."""
        result = validator.validate(corrupted_data)
        assert not result["is_valid"]
        types = self._issue_types(result)
        assert {"negative_value", "sentinel_value",
                "duplicate_rows", "constant_feature"}.issubset(types)


class TestGracefulDegradation:
    """The logging boundary must never crash the caller (the API at startup)."""

    def test_does_not_crash_on_malformed_input(self, caplog):
        """A broken input (not even a DataFrame) is swallowed, not raised."""
        with caplog.at_level(logging.ERROR):
            result = validate_and_log(object())  # has no .columns -> validator errors
        assert result is None  # validation could not run, but no exception escaped
        assert any(r.levelno == logging.ERROR for r in caplog.records)

    def test_issues_are_logged_as_warnings(self, corrupted_data, baseline_data, caplog):
        """When corruption is found, every issue surfaces at WARNING level."""
        with caplog.at_level(logging.WARNING):
            result = validate_and_log(corrupted_data, baseline=baseline_data)
        assert result is not None and not result["is_valid"]
        warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
        assert warnings, "expected at least one WARNING record"
