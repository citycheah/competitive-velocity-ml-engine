"""Gold velocity scores and drift report serialization for the federated Portal and Copilot."""

from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from intel_core.storage import write_parquet_table


def export_gold_velocity_artifacts(
    scored_df: pd.DataFrame,
    breakout_probs: np.ndarray | Sequence[float],
    *,
    alert_threshold: float,
    drift_report: Any,
    output_dir: str | Path,
    benchmark_scores: Sequence[Any] | None = None,
    mirror_dir: str | Path | None = None,
) -> dict[str, Path]:
    """Serializes `velocity_scores.parquet` and `drift_report.json` to `output_dir` (and optional mirror)."""
    probs = np.asarray(breakout_probs, dtype=float)
    df = scored_df.copy().reset_index(drop=True)
    if len(probs) != len(df):
        raise ValueError(f"Length mismatch: {len(probs)} probabilities for {len(df)} scored rows.")

    export_df = pd.DataFrame(
        {
            "video_id": df["video_id"].astype(str),
            "channel_id": df["channel_id"].astype(str),
            "channel_title": (
                df["channel_title"].astype(str)
                if "channel_title" in df.columns
                else df["channel_id"].astype(str)
            ),
            "published_at": df["published_at"].astype(str),
            "published_epoch": pd.to_numeric(df["published_epoch"], errors="coerce").fillna(0).astype(int),
            "title": df["title"].astype(str) if "title" in df.columns else "",
            "is_short": df["is_short"].astype(bool) if "is_short" in df.columns else False,
            "duration_seconds": (
                pd.to_numeric(df["duration_seconds"], errors="coerce").fillna(0).astype(int)
                if "duration_seconds" in df.columns
                else 0
            ),
            "breakout_prob": np.round(probs, 6),
            "alert_threshold": float(alert_threshold),
            "is_predicted_breakout": probs >= float(alert_threshold),
            "is_breakout_actual": (
                df["is_breakout"].astype(int) if "is_breakout" in df.columns else -1
            ),
            "channel_upload_blitz_24h": (
                pd.to_numeric(df["channel_upload_blitz_24h"], errors="coerce").fillna(0).astype(int)
                if "channel_upload_blitz_24h" in df.columns
                else 0
            ),
            "channel_upload_blitz_7d": (
                pd.to_numeric(df["channel_upload_blitz_7d"], errors="coerce").fillna(0).astype(int)
                if "channel_upload_blitz_7d" in df.columns
                else 0
            ),
            "channel_historical_avg_views_30d": (
                pd.to_numeric(df["channel_historical_avg_views_30d"], errors="coerce")
                .fillna(0.0)
                .astype(float)
                if "channel_historical_avg_views_30d" in df.columns
                else 0.0
            ),
        }
    ).sort_values("breakout_prob", ascending=False).reset_index(drop=True)

    drift_payload: dict[str, Any] = (
        drift_report.to_dict() if hasattr(drift_report, "to_dict") else dict(drift_report)
    )
    drift_payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    drift_payload["alert_threshold"] = round(float(alert_threshold), 4)
    drift_payload["scored_records"] = int(len(export_df))
    drift_payload["predicted_breakout_count"] = int(export_df["is_predicted_breakout"].sum())
    if benchmark_scores:
        drift_payload["benchmark_scores"] = [
            asdict(s) if hasattr(s, "__dataclass_fields__") else dict(s) for s in benchmark_scores
        ]

    targets = [Path(output_dir)]
    if mirror_dir is not None:
        targets.append(Path(mirror_dir))

    written: dict[str, Path] = {}
    for idx, target_dir in enumerate(targets):
        target_dir.mkdir(parents=True, exist_ok=True)
        scores_path = target_dir / "velocity_scores.parquet"
        drift_path = target_dir / "drift_report.json"
        write_parquet_table(export_df, scores_path)
        drift_path.write_text(json.dumps(drift_payload, indent=2), encoding="utf-8")
        if idx == 0:
            written["velocity_scores_path"] = scores_path
            written["drift_report_path"] = drift_path
        else:
            written["mirror_velocity_scores_path"] = scores_path
            written["mirror_drift_report_path"] = drift_path
    return written
