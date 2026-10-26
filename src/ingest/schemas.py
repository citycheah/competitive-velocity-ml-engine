"""Data schemas, PyArrow type specifications, and strict t0 leakage quarantine enforcement barrier.

Adheres strictly to Interface Contract 1 in PROJECT.md and eliminates maturity bias.
"""

from __future__ import annotations

import re
from datetime import datetime
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field


class T0LeakageError(ValueError):
    """Raised when post-t0 mature attributes leak into raw feature columns without quarantine prefix."""
    pass


class RawVideoRecord(BaseModel):
    """Schema for a single raw video record adhering strictly to Interface Contract 1."""

    model_config = ConfigDict(extra="ignore")

    # --- t0 Intrinsic Metadata (Permitted for Feature Store) ---
    video_id: str
    channel_id: str
    published_at: datetime
    published_epoch: int
    title: str
    description: str = ""
    duration_seconds: int
    is_short: bool
    tags_count: int = 0
    category_id: str = "28"  # Default Science & Technology
    channel_title: Optional[str] = None
    duration_iso: Optional[str] = None
    tags: Optional[str] = None
    definition: Optional[str] = "hd"
    caption: Optional[bool] = False
    channel_subscriber_count: Optional[int] = None
    channel_total_videos: Optional[int] = None

    # --- Quarantined Mature Metrics (Strictly for Ground Truth / Target Formulation) ---
    target_views_7d: int = 0
    target_likes_7d: int = 0
    target_comments_7d: int = 0
    target_views_30d: Optional[int] = None
    target_likes_30d: Optional[int] = None
    target_comments_30d: Optional[int] = None
    target_weighted_engagement: Optional[float] = None
    ingested_at: Optional[datetime] = None


class ChannelMetadata(BaseModel):
    """Schema for competitor channel metadata."""

    model_config = ConfigDict(extra="ignore")

    channel_id: str
    channel_title: str
    subscriber_count: int
    topic_category: str
    tier: Optional[str] = None
    niche: Optional[str] = None
    country: Optional[str] = "US"
    video_count: Optional[int] = 0
    view_count: Optional[int] = 0
    uploads_playlist_id: Optional[str] = None
    created_at: Optional[datetime] = None
    channel_created_at: Optional[datetime] = None
    ingested_at: Optional[datetime] = None


class IngestionResult(BaseModel):
    """Summary telemetry of an ingestion execution."""

    mode: str
    channels_count: int
    videos_count: int
    videos_file_path: str
    channels_file_path: str
    t0_quarantine_passed: bool
    quota_units_used: int = 0
    execution_time_seconds: float = 0.0


class T0QuarantineValidator:
    """Enforces absolute physical and schema isolation of post-t0 attributes.
    
    Guarantees zero maturity bias by banning unprefixed post-publication metrics
    and verifying data integrity across video and channel datasets.
    """

    FORBIDDEN_RAW_COLUMNS: Set[str] = {
        "views",
        "view_count",
        "likes",
        "like_count",
        "comments",
        "comment_count",
        "shares",
        "share_count",
        "watch_time",
        "ctr",
        "retention_rate",
        "view_velocity_1h",
        "engagement_rate",
        "weighted_engagement",
        "dislikes",
        "dislike_count",
    }

    FORBIDDEN_TARGET_COLUMNS: Set[str] = {
        "is_breakout",
        "breakout_label",
        "breakout",
        "engagement_zscore_30d",
        "engagement_zscore",
        "log_weighted_engagement",
        "target_weighted_engagement",
        "target_log_engagement",
        "target",
        "label",
    }

    ALLOWED_PRE_T0_CHANNEL_COLUMNS: Set[str] = {
        "channel_id",
        "channel_title",
        "channel_subscriber_count",
        "channel_topic_category",
        "channel_tier",
        "channel_country",
        "channel_created_at",
        "channel_total_videos",
        "channel_view_count",
        "channel_upload_blitz_24h",
        "channel_upload_blitz_7d",
        "channel_format_volatility_30d",
        "channel_historical_avg_views_30d",
        "channel_historical_avg_engagement_30d",
        "channel_historical_std_engagement_30d",
    }

    LEAKAGE_PATTERN: re.Pattern = re.compile(
        r"(view|like|comment|share|watch_time|ctr|retention)",
        re.IGNORECASE,
    )

    REQUIRED_T0_COLUMNS: Set[str] = {
        "video_id",
        "channel_id",
        "published_at",
        "published_epoch",
        "title",
        "duration_seconds",
        "is_short",
    }

    @classmethod
    def validate_videos(cls, df: pd.DataFrame) -> None:
        """Asserts dataframe conforms strictly to t0 quarantine standards.
        
        Raises:
            T0LeakageError: If any forbidden un-prefixed post-t0 metric column is found.
            ValueError: If required columns are missing, inconsistent, contain nested structures, or unexpected nulls/negatives.
        """
        if df.empty:
            raise ValueError("Video dataframe is empty; cannot validate schema.")

        columns = set(df.columns)

        # 1. Check for forbidden un-prefixed leakage columns (case-insensitive & pattern-based)
        normalized_map = {col: str(col).lower().strip() for col in df.columns}
        leaks: Set[str] = set()

        for col, norm in normalized_map.items():
            if norm in cls.FORBIDDEN_RAW_COLUMNS:
                leaks.add(str(col))
            elif norm.startswith("target_"):
                # Quarantined raw target metrics (target_views_7d, target_likes_7d, etc.) are valid,
                # but ground-truth target labels (breakout, zscore) are forbidden in raw videos.
                if any(target_kw in norm for target_kw in ("breakout", "zscore", "log_engagement")):
                    leaks.add(str(col))
                else:
                    pass  # Quarantined target metrics are valid in raw video dataset
            elif (
                norm in cls.FORBIDDEN_TARGET_COLUMNS
                or any(
                    target_kw in norm
                    for target_kw in ("breakout", "zscore", "weighted_engagement")
                )
            ):
                leaks.add(str(col))
            elif norm.startswith("channel_"):
                if norm not in cls.ALLOWED_PRE_T0_CHANNEL_COLUMNS:
                    if (
                        cls.LEAKAGE_PATTERN.search(norm)
                        or "target" in norm
                        or (("views" in norm or "likes" in norm) and "historical_avg_" not in norm)
                    ):
                        leaks.add(str(col))
            elif cls.LEAKAGE_PATTERN.search(norm):
                leaks.add(str(col))

        if leaks:
            raise T0LeakageError(
                f"FATAL: t0 Leakage detected! Found forbidden mature metric columns: {sorted(list(leaks))}. "
                f"Mature metrics MUST be prefixed with 'target_' to prevent maturity bias."
            )

        # 2. Disallow dictionary, list, or struct types in raw video DataFrame
        for col in df.columns:
            col_dtype = df[col].dtype
            if hasattr(col_dtype, "pyarrow_dtype"):
                pa_type = col_dtype.pyarrow_dtype
                if pa.types.is_struct(pa_type) or pa.types.is_list(pa_type) or pa.types.is_map(pa_type):
                    raise ValueError(
                        f"Disallowed nested struct or list type in raw video DataFrame: column '{col}' has type '{pa_type}'"
                    )
            if col_dtype == object:
                has_nested = df[col].apply(lambda x: isinstance(x, (dict, list, set, tuple))).any()
                if has_nested:
                    raise ValueError(
                        f"Disallowed nested dictionary, list, or struct type in raw video DataFrame: column '{col}' contains nested objects"
                    )

        # 3. Check for required t0 attributes
        missing_t0 = cls.REQUIRED_T0_COLUMNS - columns
        if missing_t0:
            raise ValueError(f"Missing required t0 columns in video dataset: {sorted(list(missing_t0))}")

        # 4. Check for at least one complete set of quarantined target columns
        has_7d_targets = {"target_views_7d", "target_likes_7d", "target_comments_7d"}.issubset(columns)
        has_30d_targets = {"target_views_30d", "target_likes_30d", "target_comments_30d"}.issubset(columns)
        if not (has_7d_targets or has_30d_targets):
            raise ValueError(
                "Missing required target columns in video dataset. "
                "Must include either {'target_views_7d', 'target_likes_7d', 'target_comments_7d'} "
                "or {'target_views_30d', 'target_likes_30d', 'target_comments_30d'}."
            )

        # 5. Check that duration and is_short are consistent
        short_mask = df["duration_seconds"] <= 60
        inconsistent_shorts = df[short_mask != df["is_short"]]
        if not inconsistent_shorts.empty:
            raise ValueError(
                f"Found {len(inconsistent_shorts)} records where is_short != (duration_seconds <= 60)"
            )

        # 6. Check null counts in required t0 columns
        null_counts = df[list(cls.REQUIRED_T0_COLUMNS)].isnull().sum()
        leaky_nulls = null_counts[null_counts > 0]
        if not leaky_nulls.empty:
            raise ValueError(f"Found unexpected null values in t0 columns: {leaky_nulls.to_dict()}")

        # 7. Validate target columns integrity: no nulls and strictly non-negative (>= 0)
        target_cols = [
            c for c in [
                "target_views_7d", "target_likes_7d", "target_comments_7d",
                "target_views_30d", "target_likes_30d", "target_comments_30d",
            ]
            if c in df.columns
        ]
        for c in target_cols:
            null_count = int(df[c].isnull().sum())
            if null_count > 0:
                raise ValueError(f"Target column '{c}' contains {null_count} null values; targets must be non-null.")
            try:
                numeric_vals = pd.to_numeric(df[c])
            except (ValueError, TypeError):
                raise ValueError(f"Target column '{c}' must contain numeric values.")
            if (numeric_vals < 0).any():
                neg_count = int((numeric_vals < 0).sum())
                raise ValueError(
                    f"Target column '{c}' contains {neg_count} negative values; target metrics must be strictly non-negative (>= 0)."
                )

    @classmethod
    def validate_feature_columns(cls, feature_columns: Any) -> None:
        """Validates that a collection of feature columns contains zero post-t0 mature metrics.
        
        Args:
            feature_columns: Iterable of column name strings or a pandas DataFrame.
            
        Raises:
            T0LeakageError: If any post-t0 mature metric (e.g., views, likes, comments,
                shares, CTR, or target_ prefixed metrics) is detected in the feature set.
        """
        if isinstance(feature_columns, pd.DataFrame):
            cols = list(feature_columns.columns)
        elif hasattr(feature_columns, "__iter__") and not isinstance(feature_columns, (str, bytes)):
            cols = list(feature_columns)
        else:
            cols = [str(feature_columns)]

        for col in cols:
            col_str = str(col)
            norm = col_str.lower().strip()

            # 1. Ground truth target label leakage check
            if (
                norm in cls.FORBIDDEN_TARGET_COLUMNS
                or any(
                    target_kw in norm
                    for target_kw in (
                        "breakout",
                        "zscore",
                        "weighted_engagement",
                        "target_weighted_engagement",
                        "target_log_engagement",
                    )
                )
            ):
                raise T0LeakageError(
                    f"Post-t0 feature leakage detected: target label column '{col_str}' cannot be used as an input feature"
                )

            # 2. Quarantined target columns belong strictly in the target partition, never in features X
            if norm.startswith("target_"):
                raise T0LeakageError(
                    f"Post-t0 feature leakage detected: target label column '{col_str}' cannot be used as an input feature"
                )

            # 3. Raw un-prefixed mature metrics
            if norm in cls.FORBIDDEN_RAW_COLUMNS:
                raise T0LeakageError(
                    f"Post-t0 feature leakage detected: forbidden column '{col_str}'"
                )

            # 4. Channel-prefixed attributes: enforce strict pre-t0 allowlist
            if norm.startswith("channel_"):
                if norm not in cls.ALLOWED_PRE_T0_CHANNEL_COLUMNS:
                    raise T0LeakageError(
                        f"Post-t0 feature leakage detected: unapproved channel column '{col_str}' cannot be used as an input feature"
                    )
                continue

            # 5. Check for mature performance keyword leakage in feature column names
            if cls.LEAKAGE_PATTERN.search(norm):
                raise T0LeakageError(
                    f"Post-t0 feature leakage detected: pattern match on '{col_str}'"
                )

    @classmethod
    def validate_channels(cls, df: pd.DataFrame) -> None:
        """Validates channel dataset schema and constraints.
        
        Raises:
            ValueError: If required columns are missing or if duplicate channel_ids exist.
        """
        if df.empty:
            raise ValueError("Channel dataframe is empty; cannot validate schema.")

        required = {"channel_id", "channel_title", "subscriber_count", "topic_category"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing required columns in channel dataset: {sorted(list(missing))}")
        if df["channel_id"].duplicated().any():
            duplicate_ids = df[df["channel_id"].duplicated()]["channel_id"].tolist()
            raise ValueError(f"Duplicate channel_id found in channels dataset: {duplicate_ids}")


def get_video_pyarrow_schema() -> pa.Schema:
    """Returns canonical PyArrow schema for youtube_videos adhering to Interface Contract 1."""
    return pa.schema([
        ("video_id", pa.string()),
        ("channel_id", pa.string()),
        ("published_at", pa.timestamp("us", tz="UTC")),
        ("published_epoch", pa.int64()),
        ("title", pa.string()),
        ("description", pa.string()),
        ("duration_seconds", pa.int32()),
        ("is_short", pa.bool_()),
        ("tags_count", pa.int32()),
        ("target_views_7d", pa.int64()),
        ("target_likes_7d", pa.int64()),
        ("target_comments_7d", pa.int64()),
        ("target_views_30d", pa.int64()),
        ("target_likes_30d", pa.int64()),
        ("target_comments_30d", pa.int64()),
        ("target_weighted_engagement", pa.float64()),
    ])


def get_channel_pyarrow_schema() -> pa.Schema:
    """Returns canonical PyArrow schema for youtube_channels adhering to Interface Contract 1."""
    return pa.schema([
        ("channel_id", pa.string()),
        ("channel_title", pa.string()),
        ("subscriber_count", pa.int64()),
        ("topic_category", pa.string()),
        ("tier", pa.string()),
        ("country", pa.string()),
        ("created_at", pa.timestamp("us", tz="UTC")),
    ])
