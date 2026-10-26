"""Offline Synthetic Multi-Channel Generator.

Generates realistic, multi-channel YouTube video and channel metadata
for offline development, CI/CD testing, and dbt feature store verification.
Guarantees zero t0 leakage, reproducible determinism, and calibrated ~15% breakouts.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

from src.ingest.schemas import T0QuarantineValidator, get_channel_pyarrow_schema, get_video_pyarrow_schema

logger = logging.getLogger(__name__)


class SyntheticDataset(NamedTuple):
    """Container for generated video and channel DataFrames supporting both attribute and tuple unpacking access."""
    videos_df: pd.DataFrame
    channels_df: pd.DataFrame


@dataclass(frozen=True)
class ChannelProfile:
    """Immutable specification for a simulated competitor channel."""
    channel_id: str
    channel_title: str
    topic_category: str
    tier: str  # 'mega', 'mid', 'upstart'
    subscriber_count: int
    cadence_lambda: float  # base uploads per day
    shorts_ratio: float  # target ratio of YouTube shorts [0.0, 1.0]
    log_mean: float  # baseline log-engagement mean (mu_c)
    log_std: float = 0.55  # baseline log-engagement standard deviation (sigma_c)
    niche: Optional[str] = None
    country: str = "US"
    created_at: str = "2020-01-15T00:00:00Z"


@dataclass
class SyntheticGeneratorConfig:
    """Configuration settings for synthetic data generation."""
    n_channels: int = 30
    days: int = 180
    start_date: str = "2025-01-01T00:00:00Z"
    seed: int = 42
    blitz_probability: float = 0.08
    predictability_rho: float = 0.50
    channels_file: Optional[Path] = field(default_factory=lambda: Path("configs/channels_registry.yaml"))
    output_dir: Path = field(default_factory=lambda: Path("data/raw"))

    def __init__(
        self,
        n_channels: Optional[int] = None,
        days: Optional[int] = None,
        start_date: Optional[str] = None,
        seed: Optional[int] = None,
        blitz_probability: Optional[float] = None,
        predictability_rho: Optional[float] = None,
        channels_file: Optional[Path] = None,
        output_dir: Optional[Path] = None,
        # Flexible aliases for test suites and callers
        channels_count: Optional[int] = None,
        num_channels: Optional[int] = None,
        num_videos: Optional[int] = None,
        timespan_days: Optional[int] = None,
        random_seed: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if channels_count is not None:
            self.n_channels = channels_count
        elif num_channels is not None:
            self.n_channels = num_channels
        elif n_channels is not None:
            self.n_channels = n_channels
        else:
            self.n_channels = 30

        if timespan_days is not None:
            self.days = timespan_days
        elif days is not None:
            self.days = days
        else:
            self.days = 180

        self.start_date = start_date if start_date is not None else "2025-01-01T00:00:00Z"

        if random_seed is not None:
            self.seed = random_seed
        elif seed is not None:
            self.seed = seed
        else:
            self.seed = 42

        self.blitz_probability = blitz_probability if blitz_probability is not None else 0.08
        self.predictability_rho = predictability_rho if predictability_rho is not None else 0.50
        self.channels_file = channels_file if channels_file is not None else Path("configs/channels_registry.yaml")
        self.output_dir = output_dir if output_dir is not None else Path("data/raw")


class SyntheticDataGenerator:
    """Simulates a multi-channel competitive YouTube media stream.
    
    Produces deterministic, schema-compliant Parquet outputs with
    quarantined target attributes and calibrated breakout frequencies.
    """

    DEFAULT_CHANNELS: List[ChannelProfile] = [
        # --- AI & Tech Niche (10 channels) ---
        ChannelProfile("UC_AIT_MEGA_001", "NeuralArchitects", "AI & Tech", "mega", 2850000, 0.35, 0.15, 13.10, niche="ai_tech"),
        ChannelProfile("UC_AIT_MEGA_002", "SiliconVector", "AI & Tech", "mega", 1750000, 0.40, 0.20, 12.65, niche="ai_tech"),
        ChannelProfile("UC_AIT_MID_001", "KarpathyLogs", "AI & Tech", "mid", 780000, 0.45, 0.25, 11.75, niche="ai_tech"),
        ChannelProfile("UC_AIT_MID_002", "TransformerLab", "AI & Tech", "mid", 560000, 0.50, 0.20, 11.40, niche="ai_tech"),
        ChannelProfile("UC_AIT_MID_003", "QuantumComputeNow", "AI & Tech", "mid", 430000, 0.35, 0.15, 11.15, niche="ai_tech"),
        ChannelProfile("UC_AIT_MID_004", "GPUHackers", "AI & Tech", "mid", 320000, 0.55, 0.30, 10.85, niche="ai_tech"),
        ChannelProfile("UC_AIT_MID_005", "EdgeAIPioneers", "AI & Tech", "mid", 210000, 0.40, 0.25, 10.45, niche="ai_tech"),
        ChannelProfile("UC_AIT_UPS_001", "LocalLLMRunner", "AI & Tech", "upstart", 115000, 0.70, 0.45, 9.80, niche="ai_tech"),
        ChannelProfile("UC_AIT_UPS_002", "PromptToCode", "AI & Tech", "upstart", 82000, 0.65, 0.50, 9.45, niche="ai_tech"),
        ChannelProfile("UC_AIT_UPS_003", "VisionModelDev", "AI & Tech", "upstart", 45000, 0.60, 0.40, 8.85, niche="ai_tech"),

        # --- Finance Niche (10 channels) ---
        ChannelProfile("UC_FIN_MEGA_001", "TheMacroDesk", "Finance", "mega", 3400000, 0.30, 0.10, 13.35, niche="macro_finance"),
        ChannelProfile("UC_FIN_MID_001", "YieldCurveAlpha", "Finance", "mid", 890000, 0.40, 0.15, 11.90, niche="macro_finance"),
        ChannelProfile("UC_FIN_MID_002", "QuantTerminal", "Finance", "mid", 670000, 0.45, 0.15, 11.60, niche="macro_finance"),
        ChannelProfile("UC_FIN_MID_003", "CapitalAllocatorsDaily", "Finance", "mid", 480000, 0.35, 0.10, 11.30, niche="macro_finance"),
        ChannelProfile("UC_FIN_MID_004", "MarketRegimeBreakdown", "Finance", "mid", 360000, 0.45, 0.20, 11.00, niche="macro_finance"),
        ChannelProfile("UC_FIN_MID_005", "BondMarketInsider", "Finance", "mid", 240000, 0.35, 0.15, 10.60, niche="macro_finance"),
        ChannelProfile("UC_FIN_UPS_001", "OptionsVolatilityLab", "Finance", "upstart", 125000, 0.65, 0.35, 9.90, niche="macro_finance"),
        ChannelProfile("UC_FIN_UPS_002", "CryptoOrderBook", "Finance", "upstart", 98000, 0.75, 0.40, 9.65, niche="macro_finance"),
        ChannelProfile("UC_FIN_UPS_003", "MacroFlowTracker", "Finance", "upstart", 68000, 0.55, 0.30, 9.30, niche="macro_finance"),
        ChannelProfile("UC_FIN_UPS_004", "HedgeFundTeardown", "Finance", "upstart", 32000, 0.50, 0.25, 8.55, niche="macro_finance"),

        # --- Data Science Niche (10 channels) ---
        ChannelProfile("UC_DAT_MEGA_001", "PythonSpeedrun", "Data Science", "mega", 2200000, 0.45, 0.30, 12.85, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MEGA_002", "DataStackDaily", "Data Science", "mega", 1600000, 0.50, 0.25, 12.55, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MID_001", "MLOpsDecoded", "Data Science", "mid", 710000, 0.55, 0.30, 11.65, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MID_002", "PandasToRust", "Data Science", "mid", 540000, 0.45, 0.25, 11.35, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MID_003", "PolarsLab", "Data Science", "mid", 420000, 0.50, 0.35, 11.10, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MID_004", "DuckDBEnthusiasts", "Data Science", "mid", 350000, 0.55, 0.30, 10.90, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_MID_005", "FeatureStoreDeepDive", "Data Science", "mid", 230000, 0.40, 0.20, 10.55, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_UPS_001", "PyTorchHacker", "Data Science", "upstart", 110000, 0.75, 0.45, 9.75, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_UPS_002", "VectorDBWeekly", "Data Science", "upstart", 75000, 0.70, 0.50, 9.40, niche="data_dev_tools"),
        ChannelProfile("UC_DAT_UPS_003", "StreamingDataOps", "Data Science", "upstart", 38000, 0.60, 0.40, 8.70, niche="data_dev_tools"),
    ]

    def __init__(
        self,
        config: Optional[SyntheticGeneratorConfig] = None,
        seed: Optional[int] = None,
        channels_count: Optional[int] = None,
        days: Optional[int] = None,
        channels_file: Optional[str | Path] = None,
        start_date: Optional[str] = None,
    ) -> None:
        self.config = config or SyntheticGeneratorConfig()
        if seed is not None:
            self.config.seed = seed
        if channels_count is not None:
            self.config.n_channels = channels_count
        if days is not None:
            self.config.days = days
        if channels_file is not None:
            self.config.channels_file = Path(channels_file) if channels_file else None
        if start_date is not None:
            self.config.start_date = start_date

        self.rng = np.random.default_rng(self.config.seed)
        self.start_datetime = datetime.fromisoformat(
            self.config.start_date.replace("Z", "+00:00")
        )
        self.end_datetime = self.start_datetime + timedelta(days=self.config.days)
        self.channels = self._init_channel_registry()

    def _init_channel_registry(self) -> List[ChannelProfile]:
        """Loads channel profiles from YAML file if available, or falls back to defaults."""
        profiles: List[ChannelProfile] = []
        c_file = self.config.channels_file

        if c_file and Path(c_file).exists():
            try:
                with open(c_file, "r", encoding="utf-8") as f:
                    data = yaml.safe_load(f)
                raw_channels = data.get("channels", [])
                for ch in raw_channels:
                    profiles.append(
                        ChannelProfile(
                            channel_id=ch["channel_id"],
                            channel_title=ch.get("channel_title", ch["channel_id"]),
                            topic_category=ch.get("topic_category") or ("AI & Tech" if "ai" in ch.get("niche", "") else "Finance" if "finance" in ch.get("niche", "") else "Data Science"),
                            tier=ch.get("tier") or ("mega" if ch.get("subscriber_tier") == "tier1" else "mid" if ch.get("subscriber_tier") == "tier2" else "upstart"),
                            subscriber_count=int(ch.get("subscriber_count", 100000)),
                            cadence_lambda=float(ch.get("cadence_lambda") or (1.0 / max(0.5, float(ch.get("baseline_cadence_days", 3.0))))),
                            shorts_ratio=float(ch.get("shorts_ratio", 0.25)),
                            log_mean=float(ch.get("log_mean", 11.0)),
                            log_std=float(ch.get("log_std", 0.55)),
                            niche=ch.get("niche"),
                            country=ch.get("country", "US"),
                            created_at=ch.get("created_at", "2020-01-15T00:00:00Z"),
                        )
                    )
                if profiles:
                    logger.info("Loaded %d channel profiles from %s", len(profiles), c_file)
            except Exception as e:
                logger.warning("Error reading %s (%s). Using DEFAULT_CHANNELS.", c_file, e)

        if not profiles:
            profiles = list(self.DEFAULT_CHANNELS)

        return profiles[: self.config.n_channels]

    def generate_channels_df(self) -> pd.DataFrame:
        """Constructs the competitor channels dimension dataframe."""
        records = []
        for ch in self.channels:
            records.append({
                "channel_id": ch.channel_id,
                "channel_title": ch.channel_title,
                "subscriber_count": np.int64(ch.subscriber_count),
                "topic_category": ch.topic_category,
                "tier": ch.tier,
                "country": ch.country,
                "created_at": pd.to_datetime(ch.created_at, utc=True),
            })
        return pd.DataFrame(records)

    def generate_videos_df(self) -> pd.DataFrame:
        """Simulates all channel timelines and compiles the raw video dataframe."""
        all_videos: List[Dict[str, Any]] = []
        video_seq = 1

        for channel in self.channels:
            channel_videos = self._simulate_channel_timeline(channel, video_seq_start=video_seq)
            all_videos.extend(channel_videos)
            video_seq += len(channel_videos)

        df = pd.DataFrame(all_videos)
        if not df.empty:
            df = df.sort_values(by=["published_epoch", "video_id"]).reset_index(drop=True)
        return df

    def _simulate_channel_timeline(
        self, channel: ChannelProfile, video_seq_start: int
    ) -> List[Dict[str, Any]]:
        """Simulates upload arrival timeline for a single channel with blitzes and diurnal clustering."""
        videos = []
        curr_time = self.start_datetime + timedelta(
            hours=float(self.rng.uniform(1.0, 48.0))
        )
        v_idx = video_seq_start

        while curr_time < self.end_datetime:
            # Check for blitz regime trigger (8% probability)
            is_blitz = bool(self.rng.uniform(0.0, 1.0) < self.config.blitz_probability)
            burst_count = int(self.rng.integers(2, 5)) if is_blitz else 1

            for burst_i in range(burst_count):
                if curr_time >= self.end_datetime:
                    break

                # 1. Format: Short vs Long-Form
                is_short = bool(self.rng.uniform(0.0, 1.0) < channel.shorts_ratio)
                if is_short:
                    duration_seconds = int(self.rng.integers(15, 60))
                else:
                    duration_seconds = int(np.clip(self.rng.lognormal(6.65, 0.45), 480, 2700))

                # 2. Title, Description, Tags & Power Keywords
                title, description, tags_count, has_power_keyword = self._generate_metadata(
                    channel.topic_category, is_short
                )

                # 3. Peak timing feature (for signal modeling)
                hour = curr_time.hour
                is_peak_hour = bool(13 <= hour <= 19)

                # 4. Latent performance variable & target calibration
                # Predictable signal component s_i
                s_raw = (
                    0.40 * (1.0 if is_blitz else 0.0)
                    + 0.35 * (0.0 if is_short else 1.0)
                    + 0.30 * (1.0 if has_power_keyword else 0.0)
                    + 0.20 * (1.0 if is_peak_hour else 0.0)
                    - 0.50 * (1.0 if (is_short and is_blitz) else 0.0)  # Shorts fatigue penalty
                )
                # Standardize s_raw to roughly zero mean, unit variance
                s_norm = (s_raw - 0.35) / 0.45

                # Innovation noise epsilon_i ~ N(0, 1)
                eps = float(self.rng.standard_normal())

                # Composite latent performance zeta_i ~ N(0, 1)
                rho = self.config.predictability_rho
                zeta = rho * s_norm + np.sqrt(1.0 - rho**2) * eps

                # Log-engagement: E_tilde = mu_c + sigma_c * zeta
                e_tilde = channel.log_mean + channel.log_std * zeta
                composite_e = float(np.exp(e_tilde) - 1.0)

                # Partition into mature views, likes, comments via empirical ratios
                like_rate = float(self.rng.beta(40, 960))  # ~0.040
                comment_rate = float(self.rng.beta(4, 996))  # ~0.004
                multiplier = 1.0 + 10.0 * like_rate + 20.0 * comment_rate

                target_views = max(10, int(round(composite_e / multiplier)))
                target_likes = max(0, int(round(target_views * like_rate)))
                target_comments = max(0, int(round(target_views * comment_rate)))

                # Formulate video record adhering strictly to Interface Contract 1
                video_record = {
                    "video_id": f"vid_{v_idx:06d}",
                    "channel_id": channel.channel_id,
                    "published_at": curr_time.replace(tzinfo=timezone.utc),
                    "published_epoch": np.int64(curr_time.timestamp()),
                    "title": title,
                    "description": description,
                    "duration_seconds": np.int32(duration_seconds),
                    "is_short": bool(is_short),
                    "tags_count": np.int32(tags_count),
                    # Strictly quarantined target metrics (Interface Contract 1)
                    "target_views_7d": np.int64(target_views),
                    "target_likes_7d": np.int64(target_likes),
                    "target_comments_7d": np.int64(target_comments),
                    "target_views_30d": np.int64(target_views),
                    "target_likes_30d": np.int64(target_likes),
                    "target_comments_30d": np.int64(target_comments),
                    "target_weighted_engagement": float(target_views + 10.0 * target_likes + 20.0 * target_comments),
                }
                videos.append(video_record)
                v_idx += 1

                # Advance intra-blitz interval (2-8 hours)
                if burst_i < burst_count - 1:
                    curr_time += timedelta(hours=float(self.rng.uniform(2.0, 8.0)))

            # Advance baseline inter-arrival interval (Exponential)
            inter_days = float(self.rng.exponential(1.0 / max(0.05, channel.cadence_lambda)))
            curr_time += timedelta(days=inter_days)

            # Jitter toward diurnal release window
            target_hour = self._sample_release_hour(curr_time.weekday() >= 5)
            curr_time = curr_time.replace(
                hour=target_hour,
                minute=int(self.rng.choice([0, 15, 30, 45])),
                second=int(self.rng.integers(0, 60)),
            )

        return videos

    def _sample_release_hour(self, is_weekend: bool) -> int:
        """Samples realistic YouTube release hours."""
        if is_weekend:
            h = int(round(self.rng.normal(13.0, 2.0)))
            return int(np.clip(h, 10, 19))
        # Weekday mixture
        if self.rng.uniform(0.0, 1.0) < 0.55:
            h = int(round(self.rng.normal(14.0, 1.4)))
        else:
            h = int(round(self.rng.normal(18.0, 1.8)))
        return int(np.clip(h, 11, 22))

    def _generate_metadata(
        self, niche: str, is_short: bool
    ) -> Tuple[str, str, int, bool]:
        """Synthesizes titles, descriptions, and tags with latent quality markers."""
        if "ai" in niche.lower():
            topics = ["Large Language Models", "Diffusion Transformers", "CUDA Kernels", "Reasoning Models", "Autonomous Agents", "Vector Search"]
            verbs = ["Building a {t} from Scratch", "Why {t} Changes Everything", "Deep Dive: How {t} Actually Works", "{t} Benchmarks: Complete Breakdown"]
            anchors = ["in Python & PyTorch", "with Rust & C++", "on NVIDIA H100s", "in Production"]
            power_markers = ["From Scratch", "Deep Dive", "Benchmarks", "Changes Everything"]
        elif "fin" in niche.lower():
            topics = ["Treasury Yield Curve", "Liquidity Squeeze", "Federal Reserve Cuts", "Private Credit Bubble", "Vol Surface Arbitrage"]
            verbs = ["The {t} Warning Sign Nobody Sees", "Why {t} Just Triggered a Shock", "Inside the {t}: Full Analysis", "The Math Behind the {t}"]
            anchors = ["2025 Market Outlook", "Fed Liquidity Index", "Bond Market Signal", "Macro Regime Shift"]
            power_markers = ["Warning Sign", "Shock", "Liquidity Squeeze", "The Math Behind"]
        else:  # Data Science
            topics = ["DuckDB Feature Stores", "Polars vs Pandas Benchmarks", "Apache Arrow Pipelines", "MLOps CI/CD", "Real-Time Streaming"]
            verbs = ["Stop Using Legacy Tools, Use {t} Instead", "How I Scaled {t} to 1 Billion Rows", "Mastering {t} in 15 Minutes", "Production {t}: 5 Critical Lessons"]
            anchors = ["100x Faster", "Zero Cloud Cost", "Production Grade", "Complete Tutorial"]
            power_markers = ["1 Billion Rows", "100x Faster", "Production Grade", "Stop Using"]

        topic = str(self.rng.choice(topics))
        template = str(self.rng.choice(verbs))
        anchor = str(self.rng.choice(anchors))

        base_title = template.format(t=topic)
        if not is_short and self.rng.uniform(0.0, 1.0) < 0.70:
            title = f"{base_title} ({anchor})"
        else:
            title = base_title

        has_power_marker = any(m.lower() in title.lower() for m in power_markers)
        tags_count = int(self.rng.integers(6, 22))

        description = (
            f"Detailed technical walkthrough covering {topic}. "
            f"We explore architectural patterns and real-world benchmarks.\n\n"
            f"Code Repository: https://github.com/competitor-benchmark/{topic.lower().replace(' ', '-')}\n"
            f"Timestamps:\n00:00 Introduction\n02:15 Architectural Overview\n"
            f"08:30 Benchmarking Results\n14:00 Engineering Conclusions\n\n"
            f"#Technology #{niche.replace(' ', '').replace('&', '')} #Coding"
        )
        return title, description, tags_count, has_power_marker

    def generate(self) -> SyntheticDataset:
        """Runs the simulation and returns SyntheticDataset(videos_df, channels_df)."""
        logger.info(
            "Generating synthetic multi-channel dataset (channels=%d, days=%d, seed=%d)",
            self.config.n_channels, self.config.days, self.config.seed,
        )
        channels_df = self.generate_channels_df()
        videos_df = self.generate_videos_df()
        logger.info("Generated %d channels and %d videos.", len(channels_df), len(videos_df))
        return SyntheticDataset(videos_df=videos_df, channels_df=channels_df)

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Alias for generate() adhering to worker specifications."""
        return self.generate()

    def save(
        self,
        videos_df: pd.DataFrame,
        channels_df: pd.DataFrame,
        output_dir: Optional[Path] = None,
    ) -> Dict[str, Path]:
        """Enforces t0 quarantine and writes Parquet tables to disk."""
        out_dir = output_dir or self.config.output_dir
        out_dir.mkdir(parents=True, exist_ok=True)

        T0QuarantineValidator.validate_videos(videos_df)
        T0QuarantineValidator.validate_channels(channels_df)

        videos_path = out_dir / "youtube_videos.parquet"
        channels_path = out_dir / "youtube_channels.parquet"

        try:
            videos_table = pa.Table.from_pandas(videos_df, schema=get_video_pyarrow_schema(), preserve_index=False)
            pq.write_table(videos_table, videos_path, compression="snappy")
        except Exception as e:
            logger.debug("Writing videos parquet via pandas to_parquet: %s", e)
            videos_df.to_parquet(videos_path, index=False, engine="pyarrow", compression="snappy")

        try:
            channels_table = pa.Table.from_pandas(channels_df, schema=get_channel_pyarrow_schema(), preserve_index=False)
            pq.write_table(channels_table, channels_path, compression="snappy")
        except Exception as e:
            logger.debug("Writing channels parquet via pandas to_parquet: %s", e)
            channels_df.to_parquet(channels_path, index=False, engine="pyarrow", compression="snappy")

        logger.info("Saved videos to %s and channels to %s", videos_path, channels_path)
        return {"videos": videos_path, "channels": channels_path}


# Alias for backward-compatibility with explorer blueprints
SyntheticGenerator = SyntheticDataGenerator
