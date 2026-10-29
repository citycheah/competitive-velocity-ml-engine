"""Comprehensive Unit & Property Verification Tests for Milestone 1 Ingestion Subsystem.

Covers:
  - Duration parsing and ISO 8601 regex
  - Uploads playlist ID derivation
  - Synthetic multi-channel generation determinism, stratification, and blitzes
  - Calibrated ~15% breakout rates
  - Strict t0 leakage quarantine barrier (T0LeakageError on forbidden columns)
  - Live YouTube API client mocking, chaining architecture, and backoff
  - Unified CLI synthetic execution and argument validation
"""

import os
import shutil
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest

from src.ingest.cli import parse_args, run_ingestion
from src.ingest.schemas import (
    ChannelMetadata,
    RawVideoRecord,
    T0LeakageError,
    T0QuarantineValidator,
    get_channel_pyarrow_schema,
    get_video_pyarrow_schema,
)
from src.ingest.synthetic_generator import (
    ChannelProfile,
    SyntheticDataGenerator,
    SyntheticGenerator,
    SyntheticGeneratorConfig,
)
from src.ingest.youtube_client import (
    QuotaExhaustedException,
    YouTubeClient,
    YouTubeClientConfig,
    derive_uploads_playlist_id,
    parse_iso8601_duration,
)


@pytest.fixture
def tmp_output_dir():
    d = tempfile.mkdtemp()
    yield Path(d)
    shutil.rmtree(d, ignore_errors=True)


# ===========================================================================
# 1. ISO 8601 Duration Parser & Playlist ID Tests
# ===========================================================================

def test_parse_iso8601_duration_cases():
    """Verify parser correctly handles all standard ISO 8601 duration formats."""
    assert parse_iso8601_duration("PT45S") == 45
    assert parse_iso8601_duration("PT1M") == 60
    assert parse_iso8601_duration("PT1M1S") == 61
    assert parse_iso8601_duration("PT15M33S") == 933
    assert parse_iso8601_duration("PT1H2M10S") == 3730
    assert parse_iso8601_duration("P1DT2H") == 93600  # 1 day (86400) + 2 hours (7200)
    assert parse_iso8601_duration("P0D") == 0
    assert parse_iso8601_duration("") == 0
    assert parse_iso8601_duration(None) == 0
    assert parse_iso8601_duration(float("nan")) == 0
    assert parse_iso8601_duration(np.nan) == 0
    assert parse_iso8601_duration("INVALID") == 0


def test_parse_iso8601_duration_nan_and_non_string():
    """Verify parse_iso8601_duration returns 0 without raising AttributeError on NaN or non-string."""
    assert parse_iso8601_duration(float("nan")) == 0
    assert parse_iso8601_duration(np.nan) == 0
    assert parse_iso8601_duration(123) == 0  # type: ignore[arg-type]
    assert parse_iso8601_duration("   ") == 0


def test_derive_uploads_playlist_id():
    """Verify deterministic 'UC' -> 'UU' string mapping."""
    assert derive_uploads_playlist_id("UC_x5XG1OV2P6uZZ5FSM9Ttw") == "UU_x5XG1OV2P6uZZ5FSM9Ttw"
    assert derive_uploads_playlist_id("UC12345") == "UU12345"
    assert derive_uploads_playlist_id("XX_non_uc") == "XX_non_uc"


# ===========================================================================
# 2. Synthetic Generator Tests
# ===========================================================================

def test_synthetic_generator_determinism():
    """Verify identical seeds produce bit-for-bit identical dataframes."""
    gen1 = SyntheticDataGenerator(seed=42, channels_count=10, days=60)
    v1, c1 = gen1.generate()

    gen2 = SyntheticDataGenerator(seed=42, channels_count=10, days=60)
    v2, c2 = gen2.generate()

    pd.testing.assert_frame_equal(v1, v2)
    pd.testing.assert_frame_equal(c1, c2)


def test_synthetic_generator_seed_divergence():
    """Verify divergent seeds produce different video datasets."""
    gen1 = SyntheticDataGenerator(seed=42, channels_count=10, days=60)
    v1, _ = gen1.generate()

    gen2 = SyntheticDataGenerator(seed=999, channels_count=10, days=60)
    v2, _ = gen2.generate()

    assert not v1["title"].equals(v2["title"])


def test_synthetic_channels_stratification():
    """Verify 30 channels are properly distributed across 3 niches and 3 tiers."""
    gen = SyntheticDataGenerator(seed=42, channels_count=30, days=60)
    df_channels = gen.generate_channels_df()

    assert len(df_channels) == 30
    assert set(df_channels["topic_category"].unique()) == {"AI & Tech", "Finance", "Data Science"}
    assert set(df_channels["tier"].unique()) == {"mega", "mid", "upstart"}
    assert df_channels["channel_id"].nunique() == 30


def test_synthetic_generator_shorts_format_consistency():
    """Verify is_short flag strictly matches duration_seconds <= 60."""
    gen = SyntheticDataGenerator(seed=42, channels_count=20, days=90)
    videos_df, _ = gen.generate()

    assert (videos_df["is_short"] == (videos_df["duration_seconds"] <= 60)).all()
    # Ensure both shorts and longform exist
    assert videos_df["is_short"].sum() > 0
    assert (~videos_df["is_short"]).sum() > 0


def test_synthetic_generator_cadence_blitz_presence():
    """Verify rapid 24h upload blitzes occur in the simulated timeline."""
    gen = SyntheticDataGenerator(seed=42, channels_count=30, days=90)
    videos_df, _ = gen.generate()

    videos_df["diff_sec"] = videos_df.groupby("channel_id")["published_epoch"].diff()
    blitz_intervals = videos_df[videos_df["diff_sec"].between(1, 86400)]
    assert len(blitz_intervals) > 0, "No upload cadence blitzes detected in simulation!"


def test_synthetic_generator_calibrated_breakout_rate():
    """Verify natural breakout rate (Z >= 1.0) is calibrated around ~15% (10%-22%)."""
    gen = SyntheticDataGenerator(seed=42, channels_count=30, days=180)
    videos_df, _ = gen.generate()

    videos_df = videos_df.sort_values(["channel_id", "published_epoch"]).reset_index(drop=True)
    log_eng = np.log1p(
        videos_df["target_views_7d"]
        + 10.0 * videos_df["target_likes_7d"]
        + 20.0 * videos_df["target_comments_7d"]
    )
    videos_df["log_eng"] = log_eng

    z_scores = videos_df.groupby("channel_id")["log_eng"].transform(
        lambda s: (s - s.mean()) / (s.std() + 1e-4)
    )
    breakout_rate = float((z_scores >= 1.0).mean())
    assert 0.10 <= breakout_rate <= 0.22, f"Breakout rate {breakout_rate:.3f} deviates from target window!"


# ===========================================================================
# 3. Strict t0 Leakage Quarantine Barrier Tests
# ===========================================================================

def test_t0_quarantine_barrier_clean_data():
    """Verify clean generated dataset passes quarantine validation without error."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, c = gen.generate()
    T0QuarantineValidator.validate_videos(v)
    T0QuarantineValidator.validate_channels(c)


def test_t0_quarantine_barrier_catches_views_leakage():
    """Injecting forbidden 'views' column must immediately raise T0LeakageError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    leaky_v = v.copy()
    leaky_v["views"] = leaky_v["target_views_7d"]

    with pytest.raises(T0LeakageError, match="t0 Leakage detected! Found forbidden"):
        T0QuarantineValidator.validate_videos(leaky_v)


def test_t0_quarantine_barrier_catches_likes_and_comments_leakage():
    """Injecting forbidden 'like_count' or 'comments' must raise T0LeakageError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    leaky_v = v.copy()
    leaky_v["like_count"] = leaky_v["target_likes_7d"]

    with pytest.raises(T0LeakageError, match="t0 Leakage detected"):
        T0QuarantineValidator.validate_videos(leaky_v)

    leaky_c = v.copy()
    leaky_c["comments"] = leaky_c["target_comments_7d"]

    with pytest.raises(T0LeakageError, match="t0 Leakage detected"):
        T0QuarantineValidator.validate_videos(leaky_c)


def test_t0_quarantine_barrier_catches_nulls_in_features():
    """Injecting null values into t0 feature columns must raise ValueError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    corrupt_v = v.copy()
    corrupt_v.loc[0, "duration_seconds"] = np.nan

    with pytest.raises(ValueError, match="Found unexpected null values in t0 columns"):
        T0QuarantineValidator.validate_videos(corrupt_v)


def test_t0_quarantine_barrier_catches_duplicate_channels():
    """Duplicate channel_id in channels dataset must raise ValueError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    _, c = gen.generate()

    dup_c = pd.concat([c, c.iloc[[0]]], ignore_index=True)
    with pytest.raises(ValueError, match="Duplicate channel_id found"):
        T0QuarantineValidator.validate_channels(dup_c)


def test_t0_quarantine_barrier_catches_case_variation_leakage():
    """Verify case variations and whitespace padding in mature column names raise T0LeakageError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    case_variations = ["Views", "VIEW_COUNT", "Like_Count", "COMMENTS", " views", "  Likes  "]
    for col_name in case_variations:
        leaky_df = v.copy()
        leaky_df[col_name] = 1000
        with pytest.raises(T0LeakageError, match="t0 Leakage detected! Found forbidden mature metric columns"):
            T0QuarantineValidator.validate_videos(leaky_df)


def test_t0_quarantine_barrier_catches_pattern_leakage():
    """Verify columns containing forbidden mature metric patterns without target_/channel_ raise T0LeakageError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    sneaky_columns = [
        "views_7d",
        "raw_views",
        "video_likes",
        "comments_count",
        "shares_total",
        "watch_time_hours",
        "thumbnail_ctr",
        "audience_retention",
    ]
    for col_name in sneaky_columns:
        leaky_df = v.copy()
        leaky_df[col_name] = 500
        with pytest.raises(T0LeakageError, match="t0 Leakage detected! Found forbidden mature metric columns"):
            T0QuarantineValidator.validate_videos(leaky_df)

    # Confirm that legitimate target_ and channel_ columns do not trigger false positive leakage
    valid_df = v.copy()
    valid_df["channel_view_count"] = 1000000
    valid_df["target_views_7d"] = 5000
    T0QuarantineValidator.validate_videos(valid_df)


def test_t0_quarantine_barrier_catches_target_label_leakage():
    """Verify raw video DataFrames containing target labels strictly raise T0LeakageError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=2, days=5)
    v, _ = gen.generate()

    target_columns = ["is_breakout", "engagement_zscore_30d", "log_weighted_engagement", "breakout"]
    for col_name in target_columns:
        leaky_df = v.copy()
        leaky_df[col_name] = 1.0
        with pytest.raises(T0LeakageError, match="t0 Leakage detected! Found forbidden"):
            T0QuarantineValidator.validate_videos(leaky_df)


def test_t0_quarantine_barrier_catches_nested_structures():
    """Verify dictionary, list, or struct types in raw video DataFrame raise ValueError."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    # Test nested dictionary (e.g. unflattened API response item['statistics'])
    nested_dict_df = v.copy()
    nested_dict_df["statistics"] = [{"viewCount": 5000, "likeCount": 200}] * len(v)
    with pytest.raises(ValueError, match="Disallowed nested dictionary, list, or struct type"):
        T0QuarantineValidator.validate_videos(nested_dict_df)

    # Test nested list (e.g. unflattened tags list)
    nested_list_df = v.copy()
    nested_list_df["raw_tags"] = [["tag1", "tag2"]] * len(v)
    with pytest.raises(ValueError, match="Disallowed nested dictionary, list, or struct type"):
        T0QuarantineValidator.validate_videos(nested_list_df)


def test_t0_quarantine_barrier_catches_null_or_negative_targets():
    """Verify target columns cannot contain null or negative values."""
    gen = SyntheticDataGenerator(seed=42, channels_count=5, days=30)
    v, _ = gen.generate()

    target_columns = ["target_views_7d", "target_likes_7d", "target_comments_7d"]

    for col in target_columns:
        # Null check
        null_df = v.copy()
        null_df.loc[0, col] = np.nan
        with pytest.raises(ValueError, match="null values"):
            T0QuarantineValidator.validate_videos(null_df)

        # Negative check
        neg_df = v.copy()
        neg_df.loc[0, col] = -10
        with pytest.raises(ValueError, match="negative values"):
            T0QuarantineValidator.validate_videos(neg_df)


# ===========================================================================
# 4. Live YouTube Client Mocking Tests (98% Quota Reduction & Backoff)
# ===========================================================================

def test_youtube_client_quota_saving_call_chain():
    """Verify client chains channels -> playlistItems -> videos, never search.list."""
    mock_service = MagicMock()

    # Mock channels().list
    mock_channels = MagicMock()
    mock_channels.list.return_value.execute.return_value = {
        "items": [{
            "id": "UC123",
            "snippet": {"title": "Test Channel", "country": "US"},
            "statistics": {"subscriberCount": "10000", "videoCount": "50", "viewCount": "500000"},
            "contentDetails": {"relatedPlaylists": {"uploads": "UU123"}},
            "topicDetails": {"topicCategories": ["https://en.wikipedia.org/wiki/Science"]},
        }]
    }
    mock_service.channels.return_value = mock_channels

    # Mock playlistItems().list
    mock_playlist_items = MagicMock()
    mock_playlist_items.list.return_value.execute.return_value = {
        "items": [{
            "contentDetails": {
                "videoId": "vid_01",
                "videoPublishedAt": "2026-10-15T12:00:00Z"
            }
        }],
        "nextPageToken": None,
    }
    mock_service.playlistItems.return_value = mock_playlist_items

    # Mock videos().list
    mock_videos = MagicMock()
    mock_videos.list.return_value.execute.return_value = {
        "items": [{
            "id": "vid_01",
            "snippet": {
                "channelId": "UC123",
                "channelTitle": "Test Channel",
                "publishedAt": "2026-10-15T12:00:00Z",
                "title": "Test Video Title",
                "description": "Test Description",
                "tags": ["python", "ml"],
                "categoryId": "28"
            },
            "contentDetails": {
                "duration": "PT2M30S",
                "definition": "hd",
                "caption": "true"
            },
            "statistics": {
                "viewCount": "5000",
                "likeCount": "250",
                "commentCount": "30"
            }
        }]
    }
    mock_service.videos.return_value = mock_videos

    client = YouTubeClient(service=mock_service)
    videos_df, channels_df = client.fetch_all(channel_ids=["UC123"])

    # Assert search was NEVER called (preserving 98% quota)
    assert not hasattr(mock_service, "search") or not mock_service.search.called

    # Assert channels, playlistItems, and videos were called
    assert mock_service.channels().list.called
    assert mock_service.playlistItems().list.called
    assert mock_service.videos().list.called

    # Validate output schema
    assert len(videos_df) == 1
    assert videos_df.loc[0, "video_id"] == "vid_01"
    assert videos_df.loc[0, "duration_seconds"] == 150
    assert not videos_df.loc[0, "is_short"]
    assert videos_df.loc[0, "target_views_7d"] == 5000
    assert "view_count" not in videos_df.columns
    assert "views" not in videos_df.columns


def test_youtube_client_incremental_watermark_early_exit():
    """Verify incremental sync stops immediately when encountering items <= watermark."""
    mock_service = MagicMock()
    watermark_dt = pd.to_datetime("2026-10-15T00:00:00Z").to_pydatetime()

    mock_playlist_items = MagicMock()
    # First item is newer than watermark, second item is older than watermark
    mock_playlist_items.list.return_value.execute.return_value = {
        "items": [
            {
                "contentDetails": {
                    "videoId": "vid_new",
                    "videoPublishedAt": "2026-10-16T12:00:00Z"
                }
            },
            {
                "contentDetails": {
                    "videoId": "vid_old",
                    "videoPublishedAt": "2026-10-14T12:00:00Z"
                }
            }
        ],
        "nextPageToken": "page2_token"
    }
    mock_service.playlistItems.return_value = mock_playlist_items

    client = YouTubeClient(service=mock_service)
    items = client.fetch_playlist_video_ids(playlist_id="UU123", watermark=watermark_dt)

    # Should only contain vid_new; halted before requesting page 2
    assert len(items) == 1
    assert items[0]["video_id"] == "vid_new"
    assert mock_playlist_items.list.call_count == 1


def test_youtube_client_quota_exhausted_exception():
    """Verify HTTP 403 quotaExceeded triggers QuotaExhaustedException."""
    mock_service = MagicMock()
    mock_request = MagicMock()

    class FakeHttpError(Exception):
        def __init__(self):
            self.resp = MagicMock(status=403)
        def __str__(self):
            return "The request cannot be completed because you have exceeded your quotaExceeded"

    mock_request.execute.side_effect = FakeHttpError()
    mock_service.channels().list.return_value = mock_request

    client = YouTubeClient(service=mock_service)
    with pytest.raises(QuotaExhaustedException, match="Daily YouTube API quota exhausted"):
        client._execute_with_retry(mock_request)


def test_youtube_client_exponential_backoff_retry():
    """Verify HTTP 429 triggers exponential backoff retry and succeeds on subsequent attempt."""
    mock_service = MagicMock()
    mock_request = MagicMock()

    class FakeHttpError429(Exception):
        def __init__(self):
            self.resp = MagicMock(status=429)
        def __str__(self):
            return "Too Many Requests (429)"

    mock_request.execute.side_effect = [
        FakeHttpError429(),
        FakeHttpError429(),
        {"items": [{"id": "vid_ok"}]},
    ]

    config = YouTubeClientConfig(base_backoff_seconds=0.01, max_backoff_seconds=0.05, max_retries=3)
    client = YouTubeClient(service=mock_service, config=config)

    res = client._execute_with_retry(mock_request, quota_cost=1)
    assert res == {"items": [{"id": "vid_ok"}]}
    assert mock_request.execute.call_count == 3
    assert client.quota_used == 1


# ===========================================================================
# 5. Unified CLI Tests
# ===========================================================================

def test_cli_synthetic_execution(tmp_output_dir):
    """Test full CLI synthetic run writes valid parquet files adhering to t0 quarantine."""
    args = parse_args([
        "--mode", "synthetic",
        "--output-dir", str(tmp_output_dir),
        "--channels-count", "10",
        "--days", "30",
        "--seed", "42",
        "--format", "parquet",
        "--quiet",
    ])
    result = run_ingestion(args)

    assert result.channels_count == 10
    assert result.videos_count > 0
    assert result.t0_quarantine_passed is True

    videos_file = tmp_output_dir / "youtube_videos.parquet"
    channels_file = tmp_output_dir / "youtube_channels.parquet"

    assert videos_file.exists()
    assert channels_file.exists()

    df_v = pd.read_parquet(videos_file)
    df_c = pd.read_parquet(channels_file)

    assert len(df_v) == result.videos_count
    assert len(df_c) == 10
    assert "target_views_7d" in df_v.columns
    assert "views" not in df_v.columns


def test_cli_argument_validation_errors():
    """Verify CLI raises errors on invalid argument bounds."""
    with pytest.raises(ValueError, match="channels-count must be between 1 and 100"):
        parse_args_and_run = parse_args(["--channels-count", "0"])
        run_ingestion(parse_args_and_run)

    with pytest.raises(ValueError, match="days must be strictly positive"):
        parse_args_and_run = parse_args(["--days", "-5"])
        run_ingestion(parse_args_and_run)


def test_cli_channels_alias_support():
    """Verify both --channels-count and --channels alias parse unambiguously."""
    args_count = parse_args(["--channels-count", "15"])
    assert args_count.channels_count == 15

    args_alias = parse_args(["--channels", "25"])
    assert args_alias.channels_count == 25

