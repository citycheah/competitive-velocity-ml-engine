"""Live YouTube Data API v3 Client with Quota-Optimized Uploads Playlist Chaining.

Implements the 3-step chaining architecture:
  channels.list -> playlistItems.list -> videos.list
This reduces quota consumption by 98% compared to search.list, while strictly quarantining
mature engagement metrics (target_views_7d, target_likes_7d, target_comments_7d) to prevent
t0 leakage.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.ingest.schemas import T0QuarantineValidator

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
    else:
        load_dotenv()
except ImportError:
    pass

logger = logging.getLogger(__name__)

# Zero-dependency regex for ISO 8601 video duration format
ISO_8601_DURATION_REGEX = re.compile(
    r"^P(?:(?P<days>\d+)D)?(?:T(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?)?$"
)

# Optional import for googleapiclient with graceful fallback
try:
    from googleapiclient.discovery import build
    from googleapiclient.errors import HttpError
    GOOGLE_API_AVAILABLE = True
except ImportError:
    build = None  # type: ignore
    HttpError = Exception  # type: ignore
    GOOGLE_API_AVAILABLE = False


class QuotaExhaustedException(Exception):
    """Raised when YouTube Data API v3 daily quota limit is reached (HTTP 403 quotaExceeded)."""
    pass


def parse_iso8601_duration(duration_str: Optional[str]) -> int:
    """Parses an ISO 8601 duration string into integer seconds.
    
    Examples:
        'PT45S'    -> 45
        'PT1M'     -> 60
        'PT1M1S'   -> 61
        'PT15M33S' -> 933
        'PT1H2M10S'-> 3730
        'P1DT2H'   -> 93600 (1 day = 86400s, 2 hours = 7200s -> 93600s)
        'P0D'      -> 0 (live stream / invalid duration)
        '' or None -> 0
    """
    if not isinstance(duration_str, str) or not duration_str.strip():
        return 0
    match = ISO_8601_DURATION_REGEX.match(duration_str.strip())
    if not match:
        return 0
    parts = match.groupdict()
    days = int(parts.get("days") or 0)
    hours = int(parts.get("hours") or 0)
    minutes = int(parts.get("minutes") or 0)
    seconds = int(parts.get("seconds") or 0)
    return days * 86400 + hours * 3600 + minutes * 60 + seconds


def derive_uploads_playlist_id(channel_id: str) -> str:
    """Derives the channel's uploads playlist ID by replacing 'UC' prefix with 'UU'.
    
    In YouTube Data API v3, a channel ID starts with 'UC' and its uploads playlist ID
    begins with 'UU' followed by the identical character sequence.
    """
    if channel_id.startswith("UC") and len(channel_id) > 2:
        return "UU" + channel_id[2:]
    return channel_id


@dataclass
class YouTubeClientConfig:
    """Configuration parameters for YouTube Data API v3 client."""
    api_key: Optional[str] = field(default_factory=lambda: os.getenv("YOUTUBE_API_KEY"))
    max_results_per_page: int = 50
    max_videos_per_channel: int = 200
    lookback_days: int = 365
    max_retries: int = 5
    base_backoff_seconds: float = 1.0
    max_backoff_seconds: float = 60.0
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))
    watermark_file: Path = field(default_factory=lambda: Path("data/raw/ingest_watermarks.json"))
    channels_file: Optional[Path] = field(default_factory=lambda: Path("configs/channels_registry.yaml"))


class YouTubeClient:
    """Production-grade YouTube Data API v3 Client.
    
    Uses uploads-playlist batch chaining to preserve 98% of API quota and
    strictly quarantines post-t0 attributes to eliminate maturity bias.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[YouTubeClientConfig] = None,
        service: Any = None,
        channels_file: Optional[str | Path] = None,
        max_channels: Optional[int] = None,
        days_lookback: Optional[int] = None,
    ) -> None:
        self.config = config or YouTubeClientConfig()
        if api_key:
            self.config.api_key = api_key
        if channels_file:
            self.config.channels_file = Path(channels_file) if channels_file else None
        if days_lookback:
            self.config.lookback_days = days_lookback

        self.service = service
        self.quota_used: int = 0
        self.max_channels = max_channels
        self.channel_ids: List[str] = []

        if self.service is None and GOOGLE_API_AVAILABLE and self.config.api_key and build is not None:
            self.service = build("youtube", "v3", developerKey=self.config.api_key, cache_discovery=False)

        # Load channel IDs from registry if available
        if self.config.channels_file and Path(self.config.channels_file).exists():
            self._load_channel_registry(Path(self.config.channels_file))

    def _load_channel_registry(self, path: Path) -> None:
        """Loads competitor channel IDs from YAML registry."""
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
            raw_channels = data.get("channels", [])
            loaded_ids = [ch["channel_id"] for ch in raw_channels if "channel_id" in ch]
            if self.max_channels:
                loaded_ids = loaded_ids[: self.max_channels]
            self.channel_ids = loaded_ids
            logger.info("Loaded %d competitor channel IDs from %s", len(self.channel_ids), path)
        except Exception as exc:
            logger.warning("Could not load channels registry from %s: %s", path, exc)

    def _execute_with_retry(self, request: Any, quota_cost: int = 1) -> Dict[str, Any]:
        """Executes an API request with exponential backoff and randomized jitter."""
        retries = 0
        while retries <= self.config.max_retries:
            try:
                response = request.execute()
                self.quota_used += quota_cost
                return response
            except Exception as e:
                # Check for HTTP errors
                is_http_err = isinstance(e, HttpError) or hasattr(e, "resp")
                status = getattr(getattr(e, "resp", None), "status", None)
                content = str(e)

                # Detect Quota Exceeded (HTTP 403 quotaExceeded)
                if status == 403 and ("quotaExceeded" in content or "dailyLimitExceeded" in content):
                    logger.error("YouTube API Daily Quota Exceeded (HTTP 403): %s", content)
                    raise QuotaExhaustedException("Daily YouTube API quota exhausted.") from e

                # Detect Retryable Errors: 429, 500, 502, 503, 504
                if status in [429, 500, 502, 503, 504] or not is_http_err:
                    if retries < self.config.max_retries:
                        retries += 1
                        delay = min(
                            self.config.max_backoff_seconds,
                            self.config.base_backoff_seconds * (2 ** (retries - 1)),
                        )
                        jitter = delay * random.uniform(-0.2, 0.2)
                        total_delay = max(0.1, delay + jitter)
                        logger.warning(
                            "Transient error encountered (status=%s). Retrying %d/%d in %.2fs...",
                            status, retries, self.config.max_retries, total_delay,
                        )
                        time.sleep(total_delay)
                        continue
                raise

    def fetch_channel_metadata(self, channel_ids: List[str]) -> List[Dict[str, Any]]:
        """Step 1: Fetches channel metadata and uploads playlist IDs in batches of 50.
        
        Quota cost: 1 unit per 50 channels.
        """
        if not self.service:
            raise ValueError("YouTube service is not initialized. Provide an API key or mock service.")

        results: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc).isoformat()

        # Batch in chunks of 50
        for i in range(0, len(channel_ids), 50):
            chunk = channel_ids[i : i + 50]
            request = self.service.channels().list(
                part="snippet,contentDetails,statistics,topicDetails",
                id=",".join(chunk),
                maxResults=len(chunk),
            )
            response = self._execute_with_retry(request, quota_cost=1)

            for item in response.get("items", []):
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                statistics = item.get("statistics", {})
                topic_details = item.get("topicDetails", {})

                ch_id = item.get("id", "")
                # Authoritative uploads playlist or deterministic derivation
                related_playlists = content_details.get("relatedPlaylists", {})
                uploads_id = related_playlists.get("uploads") or derive_uploads_playlist_id(ch_id)

                topic_cats = topic_details.get("topicCategories", [])
                primary_category = topic_cats[0].split("/")[-1] if topic_cats else "General"

                rec = {
                    "channel_id": ch_id,
                    "channel_title": snippet.get("title", ""),
                    "subscriber_count": int(statistics.get("subscriberCount") or 0),
                    "video_count": int(statistics.get("videoCount") or 0),
                    "view_count": int(statistics.get("viewCount") or 0),
                    "uploads_playlist_id": uploads_id,
                    "topic_category": primary_category,
                    "tier": "mid",
                    "country": snippet.get("country", "US"),
                    "created_at": snippet.get("publishedAt", now_utc),
                    "ingested_at": now_utc,
                }
                results.append(rec)

        return results

    def fetch_playlist_video_ids(
        self,
        playlist_id: str,
        max_videos: Optional[int] = None,
        watermark: Optional[datetime] = None,
    ) -> List[Dict[str, Any]]:
        """Step 2: Paginates playlistItems.list reverse-chronologically.
        
        Halts immediately on high-watermark match (early-exit optimization).
        Quota cost: 1 unit per 50 videos.
        """
        if not self.service:
            raise ValueError("YouTube service is not initialized.")

        video_items: List[Dict[str, Any]] = []
        limit = max_videos or self.config.max_videos_per_channel
        page_token: Optional[str] = None

        while len(video_items) < limit:
            batch_size = min(50, limit - len(video_items))
            request = self.service.playlistItems().list(
                part="snippet,contentDetails",
                playlistId=playlist_id,
                maxResults=batch_size,
                pageToken=page_token,
            )
            response = self._execute_with_retry(request, quota_cost=1)
            items = response.get("items", [])
            if not items:
                break

            early_exit = False
            for item in items:
                content_details = item.get("contentDetails", {})
                snippet = item.get("snippet", {})
                vid_id = content_details.get("videoId") or snippet.get("resourceId", {}).get("videoId")
                pub_str = content_details.get("videoPublishedAt") or snippet.get("publishedAt")

                if not vid_id:
                    continue

                if watermark and pub_str:
                    try:
                        pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                        wm = watermark
                        if wm.tzinfo is None:
                            wm = wm.replace(tzinfo=timezone.utc)
                        if pub_dt <= wm:
                            early_exit = True
                            break
                    except Exception:
                        pass

                video_items.append({
                    "video_id": vid_id,
                    "published_at": pub_str,
                })

            if early_exit or len(video_items) >= limit:
                break

            page_token = response.get("nextPageToken")
            if not page_token:
                break

        return video_items

    def fetch_video_details(
        self,
        video_ids: List[str],
        channel_metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
    ) -> List[Dict[str, Any]]:
        """Step 3: Hydrates video records via videos.list in batches of 50.
        
        Parses ISO 8601 duration, flags Shorts, and quarantines post-t0 engagement metrics.
        Quota cost: 1 unit per 50 videos.
        """
        if not self.service:
            raise ValueError("YouTube service is not initialized.")

        channel_map = channel_metadata_map or {}
        records: List[Dict[str, Any]] = []
        now_utc = datetime.now(timezone.utc)

        for i in range(0, len(video_ids), 50):
            chunk = video_ids[i : i + 50]
            request = self.service.videos().list(
                part="snippet,contentDetails,statistics",
                id=",".join(chunk),
                maxResults=len(chunk),
            )
            response = self._execute_with_retry(request, quota_cost=1)

            for item in response.get("items", []):
                v_id = item.get("id", "")
                snippet = item.get("snippet", {})
                content_details = item.get("contentDetails", {})
                statistics = item.get("statistics", {})

                ch_id = snippet.get("channelId", "")
                ch_meta = channel_map.get(ch_id, {})

                # Duration parsing and shorts classification
                duration_iso = content_details.get("duration", "PT0S")
                duration_seconds = parse_iso8601_duration(duration_iso)
                is_short = bool(duration_seconds <= 60)

                # Tags parsing
                tags_list = snippet.get("tags", [])
                tags_str = ",".join(tags_list) if tags_list else ""
                tags_count = len(tags_list)

                # Publication epoch timestamp
                pub_str = snippet.get("publishedAt", "")
                try:
                    pub_dt = datetime.fromisoformat(pub_str.replace("Z", "+00:00"))
                except Exception:
                    pub_dt = now_utc
                pub_epoch = int(pub_dt.timestamp())

                # Post-t0 metrics: strictly quarantined under target_* prefix
                views = int(statistics.get("viewCount") or 0)
                likes = int(statistics.get("likeCount") or 0)
                comments = int(statistics.get("commentCount") or 0)
                weighted_eng = float(views + 10.0 * likes + 20.0 * comments)

                rec = {
                    "video_id": v_id,
                    "channel_id": ch_id,
                    "channel_title": snippet.get("channelTitle", ch_meta.get("channel_title", "")),
                    "published_at": pub_dt,
                    "published_epoch": pub_epoch,
                    "title": snippet.get("title", ""),
                    "description": snippet.get("description", ""),
                    "duration_iso": duration_iso,
                    "duration_seconds": duration_seconds,
                    "is_short": is_short,
                    "tags": tags_str,
                    "tags_count": tags_count,
                    "category_id": snippet.get("categoryId", "28"),
                    "definition": content_details.get("definition", "hd"),
                    "caption": bool(content_details.get("caption") == "true"),
                    "channel_subscriber_count": ch_meta.get("subscriber_count", 0),
                    "channel_total_videos": ch_meta.get("video_count", 0),
                    # Quarantined post-t0 engagement metrics (Interface Contract 1)
                    "target_views_7d": views,
                    "target_likes_7d": likes,
                    "target_comments_7d": comments,
                    "target_views_30d": views,
                    "target_likes_30d": likes,
                    "target_comments_30d": comments,
                    "target_weighted_engagement": weighted_eng,
                    "ingested_at": now_utc,
                }
                records.append(rec)

        return records

    def load_watermarks(self) -> Dict[str, str]:
        """Loads persistent channel watermarks from disk."""
        if self.config.watermark_file.exists():
            try:
                with open(self.config.watermark_file, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning("Error reading watermarks file: %s", e)
        return {}

    def save_watermarks(self, watermarks: Dict[str, str]) -> None:
        """Persists high-watermark timestamps to disk."""
        self.config.watermark_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(self.config.watermark_file, "w", encoding="utf-8") as f:
                json.dump(watermarks, f, indent=2)
        except Exception as e:
            logger.warning("Error saving watermarks file: %s", e)

    def fetch_all(
        self, channel_ids: Optional[List[str]] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Runs the complete 3-step uploads playlist chaining pipeline for all target channels."""
        targets = channel_ids or self.channel_ids
        if not targets:
            raise ValueError("No channel IDs provided or loaded from registry.")

        logger.info("Executing 3-step YouTube Data API ingestion for %d channels...", len(targets))

        # Step 1: Channels metadata
        channels_meta = self.fetch_channel_metadata(targets)
        channel_map = {ch["channel_id"]: ch for ch in channels_meta}

        # Step 2: Playlist items (Uploads playlist chaining)
        all_video_ids: List[str] = []
        for ch in channels_meta:
            uploads_id = ch["uploads_playlist_id"]
            video_items = self.fetch_playlist_video_ids(uploads_id)
            all_video_ids.extend([item["video_id"] for item in video_items])

        logger.info("Discovered %d candidate video IDs across uploads playlists.", len(all_video_ids))

        # Step 3: Video details hydration & quarantine
        videos_records = self.fetch_video_details(all_video_ids, channel_metadata_map=channel_map)

        videos_df = pd.DataFrame(videos_records)
        channels_df = pd.DataFrame(channels_meta)

        if not videos_df.empty:
            videos_df = videos_df.sort_values(by=["published_epoch", "video_id"]).reset_index(drop=True)
            videos_df["published_at"] = pd.to_datetime(videos_df["published_at"], format="ISO8601", utc=True)
        if not channels_df.empty:
            channels_df["created_at"] = pd.to_datetime(channels_df["created_at"], format="ISO8601", utc=True)

        return videos_df, channels_df

    def save_to_staging(
        self,
        videos_df: pd.DataFrame,
        channels_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Tuple[Path, Path]:
        """Validates quarantine barrier and writes Parquet tables to staging directory."""
        out_dir = output_dir or self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        T0QuarantineValidator.validate_videos(videos_df)
        T0QuarantineValidator.validate_channels(channels_df)

        videos_path = out_dir / "youtube_videos.parquet"
        channels_path = out_dir / "youtube_channels.parquet"

        videos_df.to_parquet(videos_path, index=False, engine="pyarrow")
        channels_df.to_parquet(channels_path, index=False, engine="pyarrow")

        logger.info("Staged %d videos to %s", len(videos_df), videos_path)
        logger.info("Staged %d channels to %s", len(channels_df), channels_path)
        return videos_path, channels_path
