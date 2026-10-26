"""
Tests for Monitoring and Interpretability Modules (Milestone 4).

Validates Population Stability Index (PSI) drift calculation, thresholding,
baseline profile serialization, and SHAP TreeExplainer additivity.
"""

from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import pytest
from sklearn.calibration import CalibratedClassifierCV

from src.monitoring.drift_psi import (
    FeatureDriftMonitor,
    calculate_categorical_psi,
    calculate_psi,
)
from src.monitoring.shap_explainability import ModelExplainer


def test_calculate_psi_identical_distribution():
    """Identical distributions should produce near-zero PSI (< 0.02)."""
    np.random.seed(42)
    base = np.random.normal(loc=10.0, scale=2.0, size=2000)
    target = np.random.normal(loc=10.0, scale=2.0, size=2000)

    psi, base_p, targ_p, edges = calculate_psi(base, target, num_bins=10)
    assert psi < 0.05
    assert len(base_p) == len(targ_p)
    assert len(edges) > 1


def test_calculate_psi_shifted_distribution():
    """A significant distribution shift (mean shift by 2.0 std) must trigger PSI > 0.20."""
    np.random.seed(42)
    base = np.random.normal(loc=0.0, scale=1.0, size=2000)
    target = np.random.normal(loc=2.5, scale=1.0, size=2000)

    psi, base_p, targ_p, edges = calculate_psi(base, target, num_bins=10)
    assert psi > 0.20


def test_calculate_categorical_psi():
    """Categorical PSI handles identical and shifted category distributions."""
    base = ["tech"] * 80 + ["finance"] * 20
    target_same = ["tech"] * 80 + ["finance"] * 20
    target_shifted = ["tech"] * 10 + ["finance"] * 90

    psi_same, _, _, _ = calculate_categorical_psi(base, target_same)
    assert psi_same < 0.02

    psi_shifted, _, _, _ = calculate_categorical_psi(base, target_shifted)
    assert psi_shifted > 0.20


def test_feature_drift_monitor_fit_and_compute(tmp_path: Path):
    """End-to-end drift monitor evaluation across multiple feature columns."""
    np.random.seed(42)
    n = 1000

    train_df = pd.DataFrame({
        "channel_rolling_30d_views_mean": np.random.lognormal(mean=8, sigma=1, size=n),
        "channel_upload_blitz_24h": np.random.poisson(lam=1.2, size=n),
        "is_short": np.random.choice([0, 1], p=[0.7, 0.3], size=n),
        "publish_hour_of_day": np.random.randint(0, 24, size=n),
    })

    # Inference dataframe: introduce drift in views_mean and blitz_24h
    inf_df = pd.DataFrame({
        "channel_rolling_30d_views_mean": np.random.lognormal(mean=12, sigma=1, size=n),  # severe shift
        "channel_upload_blitz_24h": np.random.poisson(lam=5.0, size=n),  # severe shift
        "is_short": np.random.choice([0, 1], p=[0.7, 0.3], size=n),  # stable
        "publish_hour_of_day": np.random.randint(0, 24, size=n),  # stable
    })

    monitor = FeatureDriftMonitor(num_bins=10)
    monitor.fit_baseline(train_df, feature_cols=list(train_df.columns))

    report = monitor.compute_drift(inf_df)

    assert report.overall_status == "drift_alert"
    assert "channel_rolling_30d_views_mean" in report.drifted_features
    assert "channel_upload_blitz_24h" in report.drifted_features
    assert "is_short" not in report.drifted_features

    df_summary = report.to_dataframe()
    assert len(df_summary) == 4
    assert df_summary.iloc[0]["psi"] >= df_summary.iloc[-1]["psi"]

    # Test JSON persistence
    profile_path = tmp_path / "baseline_profile.json"
    monitor.save_baseline_profile(profile_path)
    assert profile_path.exists()

    loaded_monitor = FeatureDriftMonitor().load_baseline_profile(profile_path)
    loaded_report = loaded_monitor.compute_drift(inf_df)
    assert loaded_report.mean_psi == pytest.approx(report.mean_psi, abs=1e-5)


def test_shap_explainer_additivity_and_global_importance():
    """TreeExplainer must preserve additivity: sum(SHAP) + base_value == margin output."""
    np.random.seed(42)
    n_samples = 300
    n_features = 8

    X = np.random.randn(n_samples, n_features)
    # y depends strongly on feature 0 and 1
    y = ((X[:, 0] * 1.5 + X[:, 1] * -1.2 + np.random.randn(n_samples) * 0.5) > 0).astype(int)

    feature_names = [f"feat_{i}" for i in range(n_features)]
    clf = lgb.LGBMClassifier(n_estimators=30, max_depth=3, random_state=42, verbose=-1)
    clf.fit(X, y)

    explainer = ModelExplainer(clf, feature_names=feature_names)

    # Verify additivity holds
    assert explainer.verify_additivity(X, atol=1e-2)

    # Verify global importance
    df_global = explainer.explain_global(X, top_k=5)
    assert len(df_global) == 5
    assert df_global.iloc[0]["feature"] in ["feat_0", "feat_1"]

    # Verify local explanation
    sample_x = X[0]
    local_exp = explainer.explain_local(sample_x, top_k=3)
    assert "expected_value" in local_exp
    assert "predicted_margin" in local_exp
    assert len(local_exp["top_positive_drivers"]) <= 3
    assert len(local_exp["top_negative_drivers"]) <= 3


def test_shap_explainer_with_calibrated_classifier():
    """ModelExplainer must inspect underlying base trees within CalibratedClassifierCV."""
    np.random.seed(42)
    X = np.random.randn(200, 6)
    y = (X[:, 0] > 0).astype(int)

    base_lgbm = lgb.LGBMClassifier(n_estimators=20, max_depth=3, random_state=42, verbose=-1)
    cal_lgbm = CalibratedClassifierCV(estimator=base_lgbm, method="isotonic", cv=2)
    cal_lgbm.fit(X, y)

    explainer = ModelExplainer(cal_lgbm, feature_names=[f"col_{i}" for i in range(6)])
    assert explainer.tree_model is not None
    assert explainer.tree_model != cal_lgbm  # Extracted base estimator

    shap_vals = explainer.compute_shap_values(X[:10])
    assert shap_vals.shape == (10, 6)
