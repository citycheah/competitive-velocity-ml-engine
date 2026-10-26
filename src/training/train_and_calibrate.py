"""4-Tier Benchmark Model Ladder Training, Calibration, and Decision Threshold Optimization.

Executes:
  1. Temporal train / out-of-time holdout test splitting with 30-day maturity filter.
  2. Feature transformation with training-fold SVD text compression & t0 quarantine.
  3. Benchmarking 4-tier model ladder:
     - Baseline 1: Domain Heuristic Rule
     - Baseline 2: Regularized ElasticNet Logistic Regression
     - Model 3:    LightGBM Classifier v1
     - Model 4:    Calibrated LightGBM Classifier v2 (Isotonic Calibration + Format Interactions)
  4. Decision threshold optimization enforcing precision floor / alert lift.
  5. Artifact serialization to models/ directory.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import duckdb
import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluation.diagnostics import ModelBenchmarkScore, ModelEvaluator
from src.features.feature_pipeline import FeaturePipeline, chronological_train_test_split
from src.monitoring.drift_psi import FeatureDriftMonitor
from src.training.threshold_optimizer import DecisionThresholdOptimizer, OptimizedThresholdResult

logger = logging.getLogger("train_and_calibrate")


class HeuristicBaselineModel:
    """Domain heuristic baseline: flags breakout if 24h upload cadence spikes on standard/long-form videos."""

    def fit(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> HeuristicBaselineModel:
        return self

    def predict_proba(self, X: np.ndarray, blitz_24h_idx: int = 3, is_short_idx: int = 12) -> np.ndarray:
        """Predicts pseudo-probabilities based on cadence and non-short format."""
        probs = []
        for row in X:
            blitz = row[blitz_24h_idx]
            is_short = row[is_short_idx]
            if blitz > 0 and is_short < 0.5:
                # Active cadence blitz on full-length video
                p = 0.65
            elif blitz > 1:
                p = 0.45
            else:
                p = 0.12
            probs.append([1.0 - p, p])
        return np.array(probs)


def load_dataset_from_duckdb(db_path: Path | str) -> pd.DataFrame:
    """Loads fct_video_features_t0 table from local DuckDB feature store."""
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Feature store DuckDB not found at: {db_path}")

    con = duckdb.connect(str(db_path), read_only=True)
    try:
        df = con.execute("SELECT * FROM main.fct_video_features_t0 ORDER BY published_epoch ASC").df()
    finally:
        con.close()

    if df.empty:
        raise ValueError("fct_video_features_t0 table is empty.")

    logger.info("Loaded %d records from feature store: %s", len(df), db_path)
    return df


def train_benchmark_ladder(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    random_state: int = 42,
) -> Tuple[Dict[str, any], List[ModelBenchmarkScore], FeaturePipeline]:
    """Trains the 4-tier benchmark model ladder and evaluates on out-of-time holdout test records."""
    # 1. Fit Feature Pipeline strictly on training split
    feature_pipe = FeaturePipeline(
        n_text_components=16,
        include_format_interaction=True,
        random_state=random_state,
    )
    X_train = feature_pipe.fit_transform(train_df)
    X_test = feature_pipe.transform(test_df)

    y_train = train_df["is_breakout"].values.astype(int)
    y_test = test_df["is_breakout"].values.astype(int)

    logger.info(
        "Design matrices built: X_train=%s (pos=%.2f%%), X_test=%s (pos=%.2f%%)",
        X_train.shape,
        np.mean(y_train) * 100,
        X_test.shape,
        np.mean(y_test) * 100,
    )

    models_dict: Dict[str, any] = {}
    benchmark_scores: List[ModelBenchmarkScore] = []

    # --------------------------------------------------------------------------
    # Tier 1: Domain Heuristic Baseline
    # --------------------------------------------------------------------------
    logger.info("Evaluating Tier 1: Domain Heuristic Baseline...")
    heuristic = HeuristicBaselineModel()
    y_prob_heur = heuristic.predict_proba(X_test)[:, 1]
    score_heur = ModelEvaluator.evaluate_predictions(
        y_test,
        y_prob_heur,
        model_name="1. Heuristic Baseline",
        feature_set="Cadence Blitz Rule",
        calibrated=False,
    )
    benchmark_scores.append(score_heur)
    models_dict["heuristic"] = heuristic

    # --------------------------------------------------------------------------
    # Tier 2: Regularized ElasticNet Logistic Regression
    # --------------------------------------------------------------------------
    logger.info("Training Tier 2: Regularized ElasticNet...")
    elasticnet_pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            class_weight="balanced",
            max_iter=1000,
            random_state=random_state,
        )),
    ])
    elasticnet_pipe.fit(X_train, y_train)
    y_prob_enet = elasticnet_pipe.predict_proba(X_test)[:, 1]
    score_enet = ModelEvaluator.evaluate_predictions(
        y_test,
        y_prob_enet,
        model_name="2. ElasticNet Linear",
        feature_set="Standardized Tabular + SVD",
        calibrated=False,
    )
    benchmark_scores.append(score_enet)
    models_dict["elasticnet"] = elasticnet_pipe

    # --------------------------------------------------------------------------
    # Tier 3: LightGBM Classifier v1 (Baseline GBDT without calibration)
    # --------------------------------------------------------------------------
    logger.info("Training Tier 3: LightGBM v1 Baseline...")
    # Use unaugmented tabular features for v1 (first N - 2 columns before interaction terms)
    v1_feature_mask = np.ones(X_train.shape[1], dtype=bool)
    # In FeaturePipeline, format_interaction_surprise & cadence_blitz_acceleration are at indices 14, 15
    # For v1, exclude interaction indices
    v1_cols = [i for i in range(X_train.shape[1]) if i not in (14, 15)]
    X_train_v1 = X_train[:, v1_cols]
    X_test_v1 = X_test[:, v1_cols]

    lgbm_v1 = LGBMClassifier(
        n_estimators=100,
        max_depth=5,
        num_leaves=24,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )
    lgbm_v1.fit(X_train_v1, y_train)
    y_prob_v1 = lgbm_v1.predict_proba(X_test_v1)[:, 1]
    score_v1 = ModelEvaluator.evaluate_predictions(
        y_test,
        y_prob_v1,
        model_name="3. LightGBM (v1)",
        feature_set="Core Tabular + SVD (No Interactions)",
        calibrated=False,
    )
    benchmark_scores.append(score_v1)
    models_dict["lgbm_v1"] = lgbm_v1

    # --------------------------------------------------------------------------
    # Tier 4: Calibrated LightGBM Classifier v2 (Isotonic Calibration + Interactions)
    # --------------------------------------------------------------------------
    logger.info("Training Tier 4: Calibrated LightGBM v2 (Isotonic Calibration)...")
    lgbm_v2_base = LGBMClassifier(
        n_estimators=120,
        max_depth=5,
        num_leaves=28,
        learning_rate=0.04,
        subsample=0.85,
        colsample_bytree=0.85,
        min_child_samples=15,
        class_weight="balanced",
        random_state=random_state,
        verbose=-1,
    )

    # 3-fold cross-validated isotonic calibration on full feature design matrix
    calibrated_lgbm_v2 = CalibratedClassifierCV(
        estimator=lgbm_v2_base,
        method="isotonic",
        cv=3,
    )
    calibrated_lgbm_v2.fit(X_train, y_train)
    y_prob_v2 = calibrated_lgbm_v2.predict_proba(X_test)[:, 1]

    score_v2 = ModelEvaluator.evaluate_predictions(
        y_test,
        y_prob_v2,
        model_name="4. Calibrated LightGBM (v2)",
        feature_set="14 Features + Interactions + SVD + Isotonic",
        calibrated=True,
    )
    benchmark_scores.append(score_v2)
    models_dict["calibrated_lgbm_v2"] = calibrated_lgbm_v2

    return models_dict, benchmark_scores, feature_pipe


def run_training_pipeline(
    db_path: Path | str = "data/feature_store.duckdb",
    output_dir: Path | str = "models",
    maturity_days_filter: int = 30,
    test_ratio: float = 0.15,
    random_state: int = 42,
) -> Dict:
    """Executes full training, calibration, threshold optimization, and artifact serialization."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    df = load_dataset_from_duckdb(db_path)

    # 2. Chronological split with 30-day asymptotic maturity filter
    train_df, test_df = chronological_train_test_split(
        df,
        test_ratio=test_ratio,
        maturity_days_filter=maturity_days_filter,
    )

    # 3. Train benchmark ladder
    models_dict, benchmark_scores, feature_pipe = train_benchmark_ladder(
        train_df,
        test_df,
        random_state=random_state,
    )

    # 4. Threshold Optimization on holdout predictions of the champion model
    y_test = test_df["is_breakout"].values.astype(int)
    champion_model = models_dict["calibrated_lgbm_v2"]
    X_test = feature_pipe.transform(test_df)
    champion_probs = champion_model.predict_proba(X_test)[:, 1]

    optimizer = DecisionThresholdOptimizer(target_precision_floor=0.40)
    threshold_result = optimizer.optimize(y_test, champion_probs)

    # 5. Save Artifacts & Baseline Drift Profile
    model_save_path = output_dir / "calibrated_lgbm_v2.joblib"
    pipe_save_path = output_dir / "feature_pipeline.joblib"
    threshold_save_path = output_dir / "threshold_config.json"
    metrics_save_path = output_dir / "benchmark_metrics.json"
    drift_profile_path = output_dir / "baseline_drift_profile.json"

    joblib.dump(champion_model, model_save_path)
    joblib.dump(feature_pipe, pipe_save_path)
    DecisionThresholdOptimizer.save(threshold_result, threshold_save_path)

    # Fit and persist baseline drift monitor profile from training design matrix
    X_train = feature_pipe.transform(train_df)
    feature_names = feature_pipe.get_feature_names_out()
    train_features_df = pd.DataFrame(X_train, columns=feature_names)
    drift_monitor = FeatureDriftMonitor(num_bins=10)
    drift_monitor.fit_baseline(train_features_df, feature_cols=feature_names)
    drift_monitor.save_baseline_profile(drift_profile_path)

    # Compute drift on holdout test set as validation
    test_features_df = pd.DataFrame(X_test, columns=feature_names)
    test_drift_report = drift_monitor.compute_drift(test_features_df)
    logger.info("Test set drift validation: mean PSI = %.4f, status = %s", test_drift_report.mean_psi, test_drift_report.overall_status)

    metrics_payload = {
        "benchmark_scores": [asdict(s) for s in benchmark_scores],
        "champion_threshold": threshold_result.optimal_threshold,
        "champion_precision": threshold_result.optimal_metrics.precision,
        "champion_recall": threshold_result.optimal_metrics.recall,
        "champion_lift": threshold_result.optimal_metrics.lift_over_base_rate,
        "train_records": len(train_df),
        "test_records": len(test_df),
        "test_mean_psi": test_drift_report.mean_psi,
        "test_drift_status": test_drift_report.overall_status,
    }
    with open(metrics_save_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    # 6. Format and display summary table
    table_df = ModelEvaluator.format_ladder_table(benchmark_scores)
    print("\n" + "=" * 90)
    print("=== OUT-OF-TIME MODEL LADDER BENCHMARK SUMMARY (Holdout Test Set) ===")
    print("=" * 90)
    print(table_df.to_string(index=False))
    print("-" * 90)
    print(
        f"[*] Optimal Alert Threshold: tau = {threshold_result.optimal_threshold:.2f} | "
        f"Precision = {threshold_result.optimal_metrics.precision:.1%} | "
        f"Recall = {threshold_result.optimal_metrics.recall:.1%} | "
        f"Lift = {threshold_result.optimal_metrics.lift_over_base_rate:.2f}x over base rate"
    )
    print(f"[*] Model Artifacts Saved to: {output_dir.resolve()}")
    print("=" * 90 + "\n")

    return metrics_payload


def main() -> int:
    """CLI entry point for training and calibration pipeline."""
    parser = argparse.ArgumentParser(description="Competitive Velocity Model Training & Calibration")
    parser.add_argument("--db-path", default="data/feature_store.duckdb", help="Path to DuckDB feature store")
    parser.add_argument("--output-dir", default="models", help="Output directory for saved model artifacts")
    parser.add_argument("--maturity-days", type=int, default=30, help="Maturity filter horizon in days")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="Out-of-time test split ratio")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] [%(levelname)s] [%(name)s]: %(message)s")
    run_training_pipeline(
        db_path=args.db_path,
        output_dir=args.output_dir,
        maturity_days_filter=args.maturity_days,
        test_ratio=args.test_ratio,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
