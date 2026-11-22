"""Tests for Gold velocity exports, Streamlit deprecation stub, and 30-day peer maturity SQL guard."""

from __future__ import annotations

import json
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd
import pytest

from intel_velocity.evaluation.exports import export_gold_velocity_artifacts
from intel_velocity.monitoring.drift_psi import FeatureDriftMonitor


def test_export_gold_velocity_artifacts_writes_parquet_and_json(tmp_path: Path) -> None:
    df = pd.DataFrame(
        {
            "video_id": ["v1", "v2", "v3"],
            "channel_id": ["c1", "c1", "c2"],
            "channel_title": ["BMW M", "BMW M", "Porsche"],
            "published_at": ["2026-01-01T00:00:00Z", "2026-01-02T00:00:00Z", "2026-01-03T00:00:00Z"],
            "published_epoch": [1767225600, 1767312000, 1767398400],
            "title": ["Track Lap", "Charge Test", "GT3 Review"],
            "is_short": [False, True, False],
            "duration_seconds": [180, 45, 240],
            "is_breakout": [1, 0, 1],
            "channel_upload_blitz_24h": [2, 0, 1],
            "channel_upload_blitz_7d": [3, 1, 2],
            "channel_historical_avg_views_30d": [12000.0, 11000.0, 18000.0],
        }
    )
    probs = np.array([0.82, 0.21, 0.64])

    monitor = FeatureDriftMonitor(num_bins=4)
    base_feats = pd.DataFrame({"duration_seconds": [100, 150, 200, 250, 300, 120]})
    cur_feats = pd.DataFrame({"duration_seconds": [110, 160, 210, 240, 290, 130]})
    monitor.fit_baseline(base_feats, ["duration_seconds"])
    drift = monitor.compute_drift(cur_feats)

    primary_dir = tmp_path / "gold_velocity"
    mirror_dir = tmp_path / "lakehouse_mirror"
    res = export_gold_velocity_artifacts(
        df,
        probs,
        alert_threshold=0.55,
        drift_report=drift,
        output_dir=primary_dir,
        mirror_dir=mirror_dir,
    )

    assert res["velocity_scores_path"].exists()
    assert res["drift_report_path"].exists()
    assert res["mirror_velocity_scores_path"].exists()
    assert res["mirror_drift_report_path"].exists()

    scored = pd.read_parquet(res["velocity_scores_path"])
    assert scored["video_id"].tolist() == ["v1", "v3", "v2"]
    assert scored["is_predicted_breakout"].tolist() == [True, True, False]

    drift_json = json.loads(res["drift_report_path"].read_text(encoding="utf-8"))
    assert drift_json["overall_status"] == drift.overall_status
    assert drift_json["predicted_breakout_count"] == 2
    assert drift_json["scored_records"] == 3


def test_int_channel_rolling_stats_excludes_unmatured_peers() -> None:
    sql_path = (
        Path(__file__).resolve().parents[1]
        / "dbt_feature_store"
        / "models"
        / "intermediate"
        / "int_channel_rolling_stats.sql"
    )
    raw_sql = sql_path.read_text(encoding="utf-8")
    assert "age_days" in raw_sql
    assert ">= 30" in raw_sql

    # Render Jinja config/ref tags for direct DuckDB unit verification
    rendered = re.sub(r"\{\{\s*config\(.*?\)\s*\}\}", "", raw_sql, flags=re.DOTALL)
    rendered = re.sub(r"\{\{\s*ref\('stg_youtube_videos'\)\s*\}\}", "stg_youtube_videos", rendered)

    now = datetime.now(timezone.utc)
    rows = []
    # 3 mature peers (>= 45 days old) with modest engagement ~ 1,000
    for i, days_ago in enumerate([58, 52, 46]):
        ts = now - timedelta(days=days_ago)
        rows.append(("mature_" + str(i), "ch1", ts.replace(tzinfo=None), int(ts.timestamp()), 1000, 0, 0, 1000, 0, 0, 1000.0))
    # 1 unmatured recent peer (3 days old) with extreme incomplete/anomalous 10_000_000 value
    ts_recent_peer = now - timedelta(days=3)
    rows.append(("immature_peer", "ch1", ts_recent_peer.replace(tzinfo=None), int(ts_recent_peer.timestamp()), 10_000_000, 0, 0, 10_000_000, 0, 0, 10_000_000.0))
    # Target live video published today whose rolling window spans [-30d, -1s]
    ts_target = now - timedelta(days=1)
    rows.append(("live_target", "ch1", ts_target.replace(tzinfo=None), int(ts_target.timestamp()), 0, 0, 0, 0, 0, 0, 0.0))

    df = pd.DataFrame(
        rows,
        columns=[
            "video_id",
            "channel_id",
            "published_at",
            "published_epoch",
            "target_views_30d",
            "target_views_7d",
            "target_likes_30d",
            "target_likes_7d",
            "target_comments_30d",
            "target_comments_7d",
            "target_weighted_engagement",
        ],
    )

    con = duckdb.connect(":memory:")
    con.register("stg_youtube_videos", df)
    out = con.execute(rendered).df()
    con.close()

    target_row = out[out["video_id"] == "live_target"].iloc[0]
    # Because immature_peer is <30 days old, it must NOT enter 30d count or baseline!
    assert int(target_row["channel_upload_count_30d"]) == 0
    assert int(target_row["channel_lifetime_upload_count"]) == 3
    assert float(target_row["channel_historical_avg_engagement_30d"]) == pytest.approx(np.log1p(1000.0), rel=1e-3)


def test_deprecated_streamlit_entrypoint_raises() -> None:
    import app.streamlit_app as old_ui

    with pytest.warns(DeprecationWarning, match="retired"), pytest.raises(RuntimeError, match="retired"):
        old_ui.main()
