"""Automated dbt Runner and Orchestration Utility for Competitive Velocity ML Engine.

Handles automatic environment setup, cross-platform path resolution,
and programmatic invocation of dbt build, run, test, and clean workflows.
"""

from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def setup_logger(verbose: bool = False) -> logging.Logger:
    """Configures structured logging for dbt orchestration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="[%(asctime)s] [%(levelname)s] [run_dbt]: %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%SZ",
    )
    return logging.getLogger("run_dbt")


def get_repo_root() -> Path:
    """Locates the repository root directory."""
    current = Path(__file__).resolve().parent
    if (current.parent / "dbt_feature_store").exists():
        return current.parent
    return Path.cwd()


def prepare_environment(repo_root: Path) -> dict[str, str]:
    """Prepares environment variables with canonical absolute paths for dbt."""
    env = os.environ.copy()

    data_dir = repo_root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    raw_dir = data_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    db_path = (data_dir / "feature_store.duckdb").resolve()
    raw_videos = (raw_dir / "youtube_videos.parquet").resolve()
    raw_channels = (raw_dir / "youtube_channels.parquet").resolve()

    # Normalize to forward slashes for cross-platform SQLite/DuckDB/BigQuery compatibility
    env["DBT_DUCKDB_PATH"] = str(db_path).replace("\\", "/")
    env["RAW_VIDEOS_PATH"] = str(raw_videos).replace("\\", "/")
    env["RAW_CHANNELS_PATH"] = str(raw_channels).replace("\\", "/")
    env["DBT_RAW_SCHEMA"] = env.get("DBT_RAW_SCHEMA", "main")

    return env


def run_dbt_command(
    command: str = "build",
    select: Optional[str] = None,
    target: str = "dev",
    full_refresh: bool = False,
    threads: int = 4,
    verbose: bool = False,
) -> int:
    """Executes a dbt command with robust environment and directory settings."""
    logger = setup_logger(verbose)
    repo_root = get_repo_root()
    dbt_dir = repo_root / "dbt_feature_store"

    if not dbt_dir.exists():
        logger.error("dbt project directory not found at: %s", dbt_dir)
        return 1

    env = prepare_environment(repo_root)

    cmd: List[str] = [
        sys.executable,
        "-m",
        "dbt.cli.main",
        command,
        "--project-dir",
        str(dbt_dir),
        "--profiles-dir",
        str(dbt_dir),
        "--target",
        target,
        "--threads",
        str(threads),
    ]

    if select:
        cmd.extend(["--select", select])
    if full_refresh:
        cmd.append("--full-refresh")

    logger.info("Executing dbt command: %s", " ".join(cmd))
    logger.info("Target DuckDB: %s", env.get("DBT_DUCKDB_PATH"))
    logger.info("Raw Videos:    %s", env.get("RAW_VIDEOS_PATH"))
    logger.info("Raw Channels:  %s", env.get("RAW_CHANNELS_PATH"))

    try:
        result = subprocess.run(cmd, env=env, cwd=str(repo_root))
        if result.returncode == 0:
            logger.info("dbt %s executed successfully.", command)
        else:
            logger.error("dbt %s failed with exit code %d", command, result.returncode)
        return result.returncode
    except Exception as exc:
        logger.error("Exception during dbt execution: %s", exc)
        return 1


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parses command-line arguments."""
    parser = argparse.ArgumentParser(
        prog="run-dbt",
        description="dbt Feature Store Runner for Competitive Velocity Engine",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="build",
        choices=["run", "test", "build", "compile", "clean", "debug"],
        help="dbt command to execute (default: build)",
    )
    parser.add_argument(
        "-s", "--select",
        type=str,
        default=None,
        help="Specify models/nodes to include (e.g. 'staging', 'fct_video_features_t0')",
    )
    parser.add_argument(
        "-t", "--target",
        type=str,
        default="dev",
        choices=["dev", "prod"],
        help="Target profile output (default: dev)",
    )
    parser.add_argument(
        "--full-refresh",
        action="store_true",
        help="Rebuild tables in full refresh mode",
    )
    parser.add_argument(
        "--threads",
        type=int,
        default=4,
        help="Thread concurrency for model builds (default: 4)",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose debug logging",
    )
    return parser.parse_args(args)


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    return run_dbt_command(
        command=args.command,
        select=args.select,
        target=args.target,
        full_refresh=args.full_refresh,
        threads=args.threads,
        verbose=args.verbose,
    )


if __name__ == "__main__":
    sys.exit(main())
