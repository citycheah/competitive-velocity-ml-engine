"""Integration and unit tests for Milestone 3 (Hybrid Modeling Pipeline & Calibration).

Validates:
- Text embedding pipeline fit/transform contract and 16-d SVD projection.
- Feature pipeline t0 quarantine enforcement (zero leakage in design matrix X).
- Chronological temporal train/holdout test split and maturity filtering.
- 4-Tier Model Ladder training, probability bounds in [0, 1], and calibration quality.
- Decision threshold optimization with precision floor and lift calculation.
- Model artifact serialization and deserialization integrity.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.features.feature_pipeline import (
    FeaturePipeline,
    chronological_train_test_split,
    get_time_series_cv,
)
from src.features.text_pipeline import TextEmbeddingPipeline
from src.ingest.schemas import T0QuarantineValidator, T0LeakageError
from src.training.threshold_optimizer import DecisionThresholdOptimizer, ThresholdMetrics
from src.training.train_and_calibrate import (
    HeuristicBaselineModel,
    train_benchmark_ladder,
)


@pytest.fixture
def sample_feature_df() -> pd.DataFrame:
    """Constructs a deterministic synthetic mart DataFrame matching fct_video_features_t0."""
    n_rows = 200
    np.random.seed(42)

    epochs = np.sort(np.random.randint(1_700_000_000, 1_720_000_000, size=n_rows))
    channels = [f"UC_TEST_{i % 5:02d}" for i in range(n_rows)]

    titles = [
        f"Video {i}: How to build AI models for breakout velocity and drift" if i % 2 == 0
        else f"Short clip {i} fast tips"
        for i in range(n_rows)
    ]

    durations = [np.random.choice([30, 45, 600, 1200, 2400]) for _ in range(n_rows)]
    is_short = [1 if d <= 60 else 0 for d in durations]

    # Synthesize ~15% breakouts
    is_breakout = (np.random.rand(n_rows) < 0.15).astype(int)

    df = pd.DataFrame({
        "video_id": [f"vid_{i:04d}" for i in range(n_rows)],
        "channel_id": channels,
        "published_epoch": epochs,
        "published_at": pd.to_datetime(epochs, unit="s", utc=True),
        "title": titles,
        "duration_seconds": durations,
        "is_short": is_short,
        "channel_subscriber_count": np.random.randint(1000, 500_000, size=n_rows),
        "channel_upload_blitz_24h": np.random.poisson(0.5, size=n_rows),
        "channel_upload_blitz_7d": np.random.poisson(3.0, size=n_rows),
        "hours_since_last_upload": np.random.exponential(48.0, size=n_rows),
        "channel_format_volatility_30d": np.random.uniform(0.1, 0.9, size=n_rows),
        "channel_historical_avg_engagement_30d": np.random.normal(10.5, 1.2, size=n_rows),
        "channel_historical_std_engagement_30d": np.random.uniform(0.5, 1.5, size=n_rows),
        "title_length_chars": [len(t) for t in titles],
        "title_word_count": [len(t.split()) for t in titles],
        "publish_hour_of_day": np.random.randint(0, 24, size=n_rows),
        "publish_day_of_week": np.random.randint(0, 7, size=n_rows),
        "is_weekend": (np.random.randint(0, 7, size=n_rows) >= 5).astype(int),
        "is_breakout": is_breakout,
    })
    return df


def test_text_embedding_pipeline():
    """Verify TextEmbeddingPipeline generates exactly 16 dense SVD components."""
    texts = [
        "Building LightGBM Models with Probability Calibration",
        "Point in time feature store with dbt and DuckDB",
        "Deep learning for time series forecasting",
        "Python data engineering portfolio project",
        "YouTube metadata and engagement velocity analysis",
    ]
    pipe = TextEmbeddingPipeline(n_components=16, random_state=42)

    with pytest.raises(RuntimeError):
        pipe.transform(texts)

    embeddings = pipe.fit_transform(texts)
    assert embeddings.shape == (len(texts), 16)
    assert not np.isnan(embeddings).any()

    # Verify handling of missing/empty values
    nan_texts = [None, "", "Valid title string", float("nan")]
    transformed = pipe.transform(nan_texts)
    assert transformed.shape == (len(nan_texts), 16)
    assert not np.isnan(transformed).any()


def test_feature_pipeline_t0_quarantine(sample_feature_df):
    """Verify FeaturePipeline transforms DataFrame and strictly satisfies t0 quarantine."""
    pipe = FeaturePipeline(n_text_components=16, include_format_interaction=True, random_state=42)
    X = pipe.fit_transform(sample_feature_df)

    feature_names = pipe.get_feature_names_out()
    assert X.shape[1] == len(feature_names)
    assert X.shape[0] == len(sample_feature_df)

    # Validate that quarantine validator accepts all generated feature names
    validator = T0QuarantineValidator()
    validator.validate_feature_columns(feature_names)

    # Verify that forbidden post-t0 attributes raise leakage errors
    with pytest.raises(T0LeakageError):
        validator.validate_feature_columns(feature_names + ["target_views_7d"])

    with pytest.raises(T0LeakageError):
        validator.validate_feature_columns(feature_names + ["is_breakout"])


def test_chronological_train_test_split(sample_feature_df):
    """Verify chronological split guarantees zero temporal look-ahead leakage."""
    train_df, test_df = chronological_train_test_split(
        sample_feature_df,
        test_ratio=0.20,
        maturity_days_filter=10,
    )

    # Holdout test epochs must be strictly greater than training epochs
    assert train_df["published_epoch"].max() < test_df["published_epoch"].min()
    assert len(train_df) > 0
    assert len(test_df) > 0

    # Verify TimeSeriesSplit cv generator
    tscv = get_time_series_cv(train_df, n_splits=3)
    for tr_idx, val_idx in tscv.split(train_df):
        assert max(tr_idx) < min(val_idx)


def test_model_ladder_training_and_predictions(sample_feature_df):
    """Verify all 4 benchmark ladder models train and predict calibrated probabilities."""
    train_df, test_df = chronological_train_test_split(sample_feature_df, test_ratio=0.25)

    models_dict, scores, pipe = train_benchmark_ladder(train_df, test_df, random_state=42)

    assert len(scores) == 4
    model_names = [s.model_name for s in scores]
    assert "1. Heuristic Baseline" in model_names
    assert "2. ElasticNet Linear" in model_names
    assert "3. LightGBM (v1)" in model_names
    assert "4. Calibrated LightGBM (v2)" in model_names

    # Check Champion Calibrated LightGBM v2 probabilities
    champion = models_dict["calibrated_lgbm_v2"]
    X_test = pipe.transform(test_df)
    probs = champion.predict_proba(X_test)[:, 1]

    assert len(probs) == len(test_df)
    assert (probs >= 0.0).all() and (probs <= 1.0).all()

    # Calibrated score must have valid Brier score and ROC-AUC
    score_v2 = [s for s in scores if s.model_name == "4. Calibrated LightGBM (v2)"][0]
    assert 0.0 <= score_v2.brier_score <= 1.0
    assert 0.0 <= score_v2.pr_auc <= 1.0
    assert score_v2.calibrated is True


def test_decision_threshold_optimizer():
    """Verify DecisionThresholdOptimizer identifies valid threshold meeting precision/lift goals."""
    np.random.seed(42)
    y_true = np.array([0] * 85 + [1] * 15)  # 15% base rate
    # Simulated calibrated probabilities with positive signal correlation
    y_prob = np.where(y_true == 1, np.random.uniform(0.3, 0.9, 100), np.random.uniform(0.01, 0.4, 100))

    optimizer = DecisionThresholdOptimizer(target_precision_floor=0.40)
    res = optimizer.optimize(y_true, y_prob)

    assert 0.05 <= res.optimal_threshold <= 0.95
    assert res.optimal_metrics.precision >= 0.35
    assert res.optimal_metrics.lift_over_base_rate >= 1.5
    assert res.optimal_metrics.true_positives > 0


def test_model_artifact_serialization(sample_feature_df):
    """Verify champion model and pipeline can be saved to and loaded from disk."""
    import joblib

    train_df, test_df = chronological_train_test_split(sample_feature_df, test_ratio=0.25)
    models_dict, _, pipe = train_benchmark_ladder(train_df, test_df, random_state=42)

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir)
        model_file = tmp_path / "model.joblib"
        pipe_file = tmp_path / "pipe.joblib"

        joblib.dump(models_dict["calibrated_lgbm_v2"], model_file)
        joblib.dump(pipe, pipe_file)

        loaded_model = joblib.load(model_file)
        loaded_pipe = joblib.load(pipe_file)

        X_test = loaded_pipe.transform(test_df)
        preds = loaded_model.predict_proba(X_test)

        assert preds.shape == (len(test_df), 2)
        assert (preds >= 0.0).all() and (preds <= 1.0).all()
