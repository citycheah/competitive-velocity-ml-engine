"""Ingestion subsystem: Live YouTube API v3 client, Synthetic Multi-Channel Generator, Schemas & Quarantine Gate."""

from src.ingest.schemas import (
    ChannelMetadata,
    IngestionResult,
    RawVideoRecord,
    T0LeakageError,
    T0QuarantineValidator,
    get_channel_pyarrow_schema,
    get_video_pyarrow_schema,
)
from src.ingest.synthetic_generator import (
    ChannelProfile,
    SyntheticDataGenerator,
    SyntheticDataset,
    SyntheticGeneratorConfig,
)
from src.ingest.youtube_client import (
    QuotaExhaustedException,
    YouTubeClient,
    YouTubeClientConfig,
    derive_uploads_playlist_id,
    parse_iso8601_duration,
)

__all__ = [
    "ChannelMetadata",
    "ChannelProfile",
    "IngestionResult",
    "QuotaExhaustedException",
    "RawVideoRecord",
    "SyntheticDataGenerator",
    "SyntheticDataset",
    "SyntheticGeneratorConfig",
    "T0LeakageError",
    "T0QuarantineValidator",
    "YouTubeClient",
    "YouTubeClientConfig",
    "derive_uploads_playlist_id",
    "get_channel_pyarrow_schema",
    "get_video_pyarrow_schema",
    "parse_iso8601_duration",
]
