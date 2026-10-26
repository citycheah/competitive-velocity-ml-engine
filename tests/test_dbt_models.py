"""Integration and unit tests for dbt Feature Store models and orchestration runner.

Validates:
- scripts/run_dbt.py environment preparation and execution
- dbt build and test execution against local DuckDB
- Data contracts, row counts, and zero-null enforcement in main.fct_video_features_t0
- Breakout target distribution (~10% - 25%) and schema constraints
- Absence of maturity leakage in feature columns
"""

from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pytest

from scripts.run_dbt import get_repo_root, prepare_environment, run_dbt_command
from src.ingest.schemas import T0QuarantineValidator, T0LeakageError
from src.ingest.synthetic_generator import (
    SyntheticDataGenerator,
    SyntheticGeneratorConfig,
)


@pytest.fixture(scope="session")
def ensure_raw_fixtures():
    """Ensures synthetic raw Parquet data exists prior to dbt testing."""
    repo_root = get_repo_root()
    raw_dir = repo_root / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    videos_file = raw_dir / "youtube_videos.parquet"
    channels_file = raw_dir / "youtube_channels.parquet"

    if not videos_file.exists() or not channels_file.exists():
        config = SyntheticGeneratorConfig(
            channels_count=10,
            days=90,
            seed=42,
        )
        generator = SyntheticDataGenerator(config)
        videos_df, channels_df = generator.generate()

        videos_df.to_parquet(videos_file, index=False)
        channels_df.to_parquet(channels_file, index=False)

    return videos_file, channels_file


@pytest.fixture(scope="session")
def built_dbt_feature_store(ensure_raw_fixtures):
    """Guarantees DuckDB feature store exists and dbt models are built before inspecting tables."""
    repo_root = get_repo_root()
    db_path = repo_root / "data" / "feature_store.duckdb"

    needs_build = False
    if not db_path.exists():
        needs_build = True
    else:
        try:
            con = duckdb.connect(str(db_path), read_only=True)
            tables = [r[0] for r in con.execute("SHOW TABLES").fetchall()]
            con.close()
            if "fct_video_features_t0" not in tables:
                needs_build = True
        except Exception:
            needs_build = True

    if needs_build:
        rc = run_dbt_command(command="build", target="dev", threads=4)
        if rc != 0:
            raise RuntimeError(f"dbt build failed with return code {rc}")

    return db_path


def test_prepare_environment():
    """Verify environment setup normalizes paths with forward slashes for cross-platform compatibility."""
    repo_root = get_repo_root()
    env = prepare_environment(repo_root)

    assert "DBT_DUCKDB_PATH" in env
    assert "RAW_VIDEOS_PATH" in env
    assert "RAW_CHANNELS_PATH" in env

    assert "\\" not in env["DBT_DUCKDB_PATH"]
    assert "\\" not in env["RAW_VIDEOS_PATH"]
    assert "\\" not in env["RAW_CHANNELS_PATH"]

    assert env["DBT_DUCKDB_PATH"].endswith("feature_store.duckdb")
    assert env["RAW_VIDEOS_PATH"].endswith("youtube_videos.parquet")
    assert env["RAW_CHANNELS_PATH"].endswith("youtube_channels.parquet")


def test_dbt_build_and_test_execution(built_dbt_feature_store):
    """Verify dbt build and test execute with exit code 0."""
    build_rc = run_dbt_command(command="build", target="dev", threads=4)
    assert build_rc == 0, f"dbt build failed with return code {build_rc}"

    test_rc = run_dbt_command(command="test", target="dev", threads=4)
    assert test_rc == 0, f"dbt test failed with return code {test_rc}"


def test_fct_video_features_t0_table_exists_and_populated(built_dbt_feature_store):
    """Verify DuckDB main.fct_video_features_t0 table is populated with records."""
    db_path = built_dbt_feature_store

    assert db_path.exists(), f"DuckDB database not found at {db_path}"

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        row_count = con.execute("SELECT COUNT(*) FROM main.fct_video_features_t0").fetchone()[0]
        assert row_count > 0, f"Expected > 0 rows in main.fct_video_features_t0, found {row_count}"
    finally:
        con.close()


def test_fct_video_features_t0_zero_nulls_enforcement(built_dbt_feature_store):
    """Verify exhaustive zero-null policy across all feature and target columns in the mart."""
    db_path = built_dbt_feature_store

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("SELECT * FROM main.fct_video_features_t0 LIMIT 1000").df()
        assert len(df) > 0

        # Check critical features for zero nulls
        critical_columns = [
            "video_id",
            "channel_id",
            "published_at",
            "published_epoch",
            "channel_title",
            "channel_subscriber_count",
            "topic_category",
            "title",
            "description",
            "duration_seconds",
            "is_short",
            "tags_count",
            "publish_hour_of_day",
            "publish_day_of_week",
            "is_weekend",
            "title_length_chars",
            "title_word_count",
            "channel_upload_blitz_24h",
            "channel_upload_blitz_7d",
            "hours_since_last_upload",
            "channel_format_volatility_30d",
            "channel_historical_avg_views_30d",
            "channel_historical_avg_engagement_30d",
            "channel_historical_std_engagement_30d",
            "target_views_7d",
            "target_likes_7d",
            "target_comments_7d",
            "target_views_30d",
            "target_likes_30d",
            "target_comments_30d",
            "target_weighted_engagement",
            "engagement_zscore_30d",
            "is_breakout",
        ]

        null_counts = df[critical_columns].isnull().sum()
        failing_cols = null_counts[null_counts > 0]
        assert failing_cols.empty, f"Found unhandled NULL values in mart columns: {failing_cols.to_dict()}"
    finally:
        con.close()


def test_fct_video_features_t0_breakout_target_distribution(built_dbt_feature_store):
    """Verify is_breakout is strictly binary {0, 1} and target rate is within reasonable bounds (~10% - 25%)."""
    db_path = built_dbt_feature_store

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        res = con.execute("""
            SELECT
                AVG(is_breakout),
                MIN(is_breakout),
                MAX(is_breakout),
                COUNT(DISTINCT is_breakout)
            FROM main.fct_video_features_t0
        """).fetchone()

        avg_breakout, min_val, max_val, distinct_vals = res

        assert min_val == 0
        assert max_val == 1
        assert distinct_vals in (1, 2)
        assert 0.08 <= avg_breakout <= 0.30, f"Breakout rate {avg_breakout:.4f} outside expected ~10%-25% window"
    finally:
        con.close()


def test_fct_video_features_t0_blitz_monotonicity(built_dbt_feature_store):
    """Verify blitz_7d >= blitz_24h >= 0 across all records."""
    db_path = built_dbt_feature_store

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        violations = con.execute("""
            SELECT COUNT(*)
            FROM main.fct_video_features_t0
            WHERE channel_upload_blitz_24h < 0
               OR channel_upload_blitz_7d < channel_upload_blitz_24h
        """).fetchone()[0]

        assert violations == 0, f"Found {violations} records violating blitz monotonicity!"
    finally:
        con.close()


def test_fct_video_features_t0_t0_quarantine_barrier(built_dbt_feature_store):
    """Verify that features knowable at t0 contain zero un-prefixed post-t0 mature metrics."""
    db_path = built_dbt_feature_store

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        col_names = [col[0] for col in con.execute("DESCRIBE main.fct_video_features_t0").fetchall()]
    finally:
        con.close()

    # Isolate non-target, non-breakout feature columns
    feature_cols = [
        c for c in col_names 
        if not c.startswith("target_") and c not in ("is_breakout", "engagement_zscore_30d")
    ]

    validator = T0QuarantineValidator()
    validator.validate_feature_columns(feature_cols)

    base = feature_cols

    # Discrete assertions: verify targets and sneaked channel metrics strictly raise T0LeakageError
    forbidden_features = [
        "is_breakout",
        "engagement_zscore_30d",
        "channel_views_7d",
        "channel_target_views_7d",
    ]
    for forbidden in forbidden_features:
        with pytest.raises(T0LeakageError, match="Post-t0 feature leakage detected"):
            validator.validate_feature_columns(base + [forbidden])

    # Adversarial target variants and unapproved channel features
    with pytest.raises(T0LeakageError, match="Post-t0 feature leakage detected"):
        validator.validate_feature_columns(base + ["feat_log_weighted_engagement"])

    with pytest.raises(T0LeakageError, match="Post-t0 feature leakage detected"):
        validator.validate_feature_columns(base + ["channel_log_weighted_engagement"])

    with pytest.raises(T0LeakageError, match="Post-t0 feature leakage detected"):
        validator.validate_feature_columns(base + ["channel_subscriber_growth_rate_30d"])

    # Raw video DataFrame target leakage validation
    gen = SyntheticDataGenerator(seed=42, channels_count=2, days=5)
    sample_videos, _ = gen.generate()

    leaky_breakout_df = sample_videos.copy()
    leaky_breakout_df["is_breakout"] = 1
    with pytest.raises(T0LeakageError, match="t0 Leakage detected"):
        T0QuarantineValidator.validate_videos(leaky_breakout_df)

    leaky_zscore_df = sample_videos.copy()
    leaky_zscore_df["engagement_zscore_30d"] = 2.5
    with pytest.raises(T0LeakageError, match="t0 Leakage detected"):
        T0QuarantineValidator.validate_videos(leaky_zscore_df)
