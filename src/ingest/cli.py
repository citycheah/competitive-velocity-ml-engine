"""Unified Ingestion CLI for Competitive Content Velocity & Drift Engine.

Supports dual-mode execution:
  --mode synthetic (zero-credential offline generator with calibrated breakouts)
  --mode live      (YouTube Data API v3 quota-optimized client)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd

from src.ingest.schemas import IngestionResult, T0QuarantineValidator
from src.ingest.synthetic_generator import SyntheticDataGenerator
from src.ingest.youtube_client import YouTubeClient

try:
    from dotenv import load_dotenv
    _env_path = Path(__file__).resolve().parent.parent.parent / ".env"
    if _env_path.exists():
        load_dotenv(dotenv_path=_env_path)
    else:
        load_dotenv()
except ImportError:
    pass


def setup_logger(verbose: bool = False, quiet: bool = False) -> logging.Logger:
    """Configures structured logging for CLI execution."""
    level = logging.WARNING if quiet else (logging.DEBUG if verbose else logging.INFO)
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    return logging.getLogger("ingest_cli")


def parse_args(args: Optional[list] = None) -> argparse.Namespace:
    """Parses command-line arguments for the ingestion CLI."""
    parser = argparse.ArgumentParser(
        prog="competitive-ingest",
        description="Dual-Mode Ingestion CLI for Competitive Velocity ML Engine",
    )
    parser.add_argument(
        "--mode",
        choices=["synthetic", "live"],
        default="synthetic",
        help="Ingestion mode: 'synthetic' (default, offline) or 'live' (YouTube API v3)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/raw",
        help="Directory to save raw parquet/json drops (default: data/raw)",
    )
    parser.add_argument(
        "--channels-count",
        "--channels",
        dest="channels_count",
        type=int,
        default=30,
        help="Number of competitor channels to simulate or sample (default: 30, range 1-100)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=180,
        help="Historical lookback / generation horizon in days (default: 180)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible synthetic generation (default: 42)",
    )
    parser.add_argument(
        "--format",
        choices=["parquet", "json"],
        default="parquet",
        help="Output serialization format (default: parquet)",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("YOUTUBE_API_KEY", ""),
        help="YouTube Data API v3 key (or set YOUTUBE_API_KEY environment variable)",
    )
    parser.add_argument(
        "--channels-file",
        type=str,
        default="configs/channels_registry.yaml",
        help="Path to competitor channels registry YAML file",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging",
    )
    parser.add_argument(
        "-q", "--quiet",
        action="store_true",
        help="Suppress non-error stdout logging",
    )
    return parser.parse_args(args)


def run_ingestion(parsed_args: argparse.Namespace) -> IngestionResult:
    """Executes the dual-mode ingestion workflow."""
    start_time = time.time()
    logger = setup_logger(parsed_args.verbose, parsed_args.quiet)
    output_path = Path(parsed_args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Validate arguments
    if parsed_args.channels_count < 1 or parsed_args.channels_count > 100:
        raise ValueError(
            f"channels-count must be between 1 and 100, got {parsed_args.channels_count}"
        )
    if parsed_args.days < 1:
        raise ValueError(f"days must be strictly positive, got {parsed_args.days}")
    if parsed_args.seed < 0:
        raise ValueError(f"seed must be non-negative, got {parsed_args.seed}")

    logger.info(
        "Initializing ingestion: mode=%s, channels=%d, days=%d, format=%s",
        parsed_args.mode,
        parsed_args.channels_count,
        parsed_args.days,
        parsed_args.format,
    )

    quota_used = 0
    if parsed_args.mode == "synthetic":
        generator = SyntheticDataGenerator(
            seed=parsed_args.seed,
            channels_count=parsed_args.channels_count,
            days=parsed_args.days,
            channels_file=parsed_args.channels_file,
        )
        videos_df, channels_df = generator.generate()
    elif parsed_args.mode == "live":
        if not parsed_args.api_key:
            raise ValueError(
                "YouTube Data API key is required for --mode live. "
                "Provide --api-key or set YOUTUBE_API_KEY environment variable."
            )
        client = YouTubeClient(
            api_key=parsed_args.api_key,
            channels_file=parsed_args.channels_file,
            max_channels=parsed_args.channels_count,
            days_lookback=parsed_args.days,
        )
        videos_df, channels_df = client.fetch_all()
        quota_used = client.quota_used
    else:
        raise ValueError(f"Unsupported mode: {parsed_args.mode}")

    # Enforce strict t0 leakage quarantine barrier
    logger.info("Enforcing strict t0 leakage quarantine barrier on raw dataframes...")
    T0QuarantineValidator.validate_videos(videos_df)
    T0QuarantineValidator.validate_channels(channels_df)

    # Persist files
    ext = parsed_args.format
    videos_file = output_path / f"youtube_videos.{ext}"
    channels_file = output_path / f"youtube_channels.{ext}"

    if ext == "parquet":
        videos_df.to_parquet(videos_file, index=False, engine="pyarrow")
        channels_df.to_parquet(channels_file, index=False, engine="pyarrow")
    else:
        videos_df.to_json(videos_file, orient="records", date_format="iso", indent=2)
        channels_df.to_json(channels_file, orient="records", indent=2)

    elapsed = time.time() - start_time
    logger.info("Saved %d video records to %s", len(videos_df), videos_file)
    logger.info("Saved %d channel records to %s", len(channels_df), channels_file)

    return IngestionResult(
        mode=parsed_args.mode,
        channels_count=len(channels_df),
        videos_count=len(videos_df),
        videos_file_path=str(videos_file),
        channels_file_path=str(channels_file),
        t0_quarantine_passed=True,
        quota_units_used=quota_used,
        execution_time_seconds=round(elapsed, 3),
    )


def main() -> int:
    """Entry point for CLI execution."""
    try:
        args = parse_args()
        result = run_ingestion(args)
        if not args.quiet:
            print("\n=== Ingestion Completed Successfully ===")
            print(f"Mode:                   {result.mode}")
            print(f"Channels Count:         {result.channels_count}")
            print(f"Videos Count:           {result.videos_count}")
            print(f"Videos File:            {result.videos_file_path}")
            print(f"Channels File:          {result.channels_file_path}")
            print(f"t0 Quarantine Status:   PASSED")
            print(f"Execution Time:         {result.execution_time_seconds:.3f}s")
            if result.mode == "live":
                print(f"Quota Units Used:       {result.quota_units_used}")
        return 0
    except Exception as exc:
        print(f"ERROR: Ingestion failed: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
