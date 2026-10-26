"""Autonomous End-to-End Verification Pipeline Script for Competitive Content Velocity Engine.

Executes all core pipeline stages in an isolated, offline-first environment:
  1. Preflight Dependency Check (DuckDB, dbt, LightGBM, scikit-learn, joblib, shap, streamlit)
  2. Synthetic Multi-Channel Data Ingestion (30 channels, Poisson arrivals, blitz bursts)
  3. t0 Physical Quarantine Enforcement (T0QuarantineValidator)
  4. dbt Point-in-Time Feature Store Compilation & Execution (DuckDB Engine)
  5. 14 High-Signal Feature Pipeline + 16-d SVD Text Embeddings
  6. Chronological Holdout Split with 30-Day Maturity Cutoff
  7. 4-Tier Benchmark Model Ladder Training (Heuristic -> ElasticNet -> LightGBM v1 -> Calibrated v2)
  8. Decision Threshold Optimization (Alert Fatigue Constraint & Lift Calculation)
  9. Model Artifact Serialization & Integrity Check
  10. SHAP TreeExplainer Interpretability & Additivity Verification
  11. Population Stability Index (PSI) Drift Monitoring & Profile Check
  12. Streamlit Dashboard Headless Execution Preflight
  13. Verification Summary Report Generation
"""

from __future__ import annotations

import json
import logging
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

import duckdb
import joblib
import numpy as np
import pandas as pd

from src.evaluation.diagnostics import ModelEvaluator
from src.features.feature_pipeline import FeaturePipeline, chronological_train_test_split
from src.ingest.schemas import T0QuarantineValidator
from src.ingest.synthetic_generator import SyntheticDataGenerator, SyntheticGeneratorConfig
from src.monitoring.drift_psi import FeatureDriftMonitor
from src.monitoring.shap_explainability import ModelExplainer
from src.training.threshold_optimizer import DecisionThresholdOptimizer
from src.training.train_and_calibrate import train_benchmark_ladder


def run_pipeline() -> int:
    print("=" * 85)
    print("=== COMPETITIVE CONTENT VELOCITY & DRIFT ENGINE - MASTER E2E VERIFICATION ===")
    print("=" * 85)

    reports_dir = root_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report_data = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "stages": {},
        "metrics": {},
        "verdict": "FAILED",
    }

    try:
        # Stage 1: Preflight
        print("\n[Stage 1/13] Preflight Dependency Check...")
        import lightgbm
        import shap
        import sklearn
        import streamlit
        report_data["stages"]["stage_1_preflight"] = "PASSED"
        print(f"  [+] Scikit-learn: {sklearn.__version__} | LightGBM: {lightgbm.__version__} | SHAP: {shap.__version__} | DuckDB: {duckdb.__version__}")

        # Stage 2: Ingestion
        print("\n[Stage 2/13] Dual-Mode Synthetic Ingestion (30 Channels, 90 Days)...")
        config = SyntheticGeneratorConfig(channels_count=30, days=90, seed=42)
        generator = SyntheticDataGenerator(config)
        videos_df, channels_df = generator.generate()
        assert len(channels_df) == 30
        assert len(videos_df) > 500

        # Persist raw synthetic drops to data/raw/
        raw_dir = root_dir / "data" / "raw"
        raw_dir.mkdir(parents=True, exist_ok=True)
        videos_df.to_parquet(raw_dir / "youtube_videos.parquet", index=False)
        channels_df.to_parquet(raw_dir / "youtube_channels.parquet", index=False)

        report_data["stages"]["stage_2_ingestion"] = "PASSED"
        print(f"  [+] Generated and persisted {len(videos_df)} video records across {len(channels_df)} channels.")

        # Stage 3: t0 Quarantine
        print("\n[Stage 3/13] Enforcing Physical t0 Quarantine Barrier...")
        T0QuarantineValidator.validate_videos(videos_df)
        T0QuarantineValidator.validate_channels(channels_df)
        report_data["stages"]["stage_3_quarantine"] = "PASSED"
        print("  [+] Quarantine barrier verified: zero unprefixed mature metrics observable at t0.")

        # Stage 4: Feature Store Validation
        print("\n[Stage 4/13] Point-in-Time Feature Store Compilation & Validation (DuckDB)...")
        from scripts.run_dbt import run_dbt_command
        dbt_code = run_dbt_command(command="build", target="dev")
        if dbt_code != 0:
            raise RuntimeError(f"dbt build failed with exit code {dbt_code}")

        db_path = root_dir / "data" / "feature_store.duckdb"
        con = duckdb.connect(str(db_path), read_only=True)
        mart_count = con.execute("SELECT COUNT(*) FROM main.fct_video_features_t0").fetchone()[0]
        mart_df = con.execute("SELECT * FROM main.fct_video_features_t0 ORDER BY published_epoch ASC").df()
        con.close()
        assert mart_count > 0
        report_data["stages"]["stage_4_feature_store"] = "PASSED"
        print(f"  [+] Loaded {mart_count} records from main.fct_video_features_t0.")

        # Stage 5: Chronological Train/Holdout Split
        print("\n[Stage 5/13] Chronological Split with 30-Day Maturity Cutoff...")
        train_df, holdout_df = chronological_train_test_split(
            mart_df,
            test_ratio=0.15,
            maturity_days_filter=30,
        )
        assert train_df["published_epoch"].max() < holdout_df["published_epoch"].min()
        report_data["stages"]["stage_5_chronological_split"] = "PASSED"
        print(f"  [+] Chronological split: Train={len(train_df)} rows, Holdout Test={len(holdout_df)} rows.")

        # Stage 6: Feature Pipeline with SVD Text Compression
        print("\n[Stage 6/13] Feature Pipeline Fitting (14 High-Signal Features + 16-d SVD)...")
        pipe = FeaturePipeline(n_text_components=16, include_format_interaction=True, random_state=42)
        X_train = pipe.fit_transform(train_df)
        X_holdout = pipe.transform(holdout_df)
        feature_names = pipe.get_feature_names_out()
        T0QuarantineValidator.validate_feature_columns(feature_names)
        assert X_train.shape[1] == len(feature_names)
        report_data["stages"]["stage_6_feature_pipeline"] = "PASSED"
        print(f"  [+] Design matrix generated: {X_train.shape[1]} features (14 core + interactions + 16 text SVD).")

        # Stage 7: 4-Tier Model Ladder Training
        print("\n[Stage 7/13] 4-Tier Benchmark Ladder Training & Calibration...")
        models_dict, scores, fitted_pipe = train_benchmark_ladder(train_df, holdout_df, random_state=42)
        assert len(scores) == 4
        report_data["stages"]["stage_7_ladder"] = "PASSED"
        for s in scores:
            print(f"  [+] {s.model_name:<28} | PR-AUC={s.pr_auc:.4f} | ROC-AUC={s.roc_auc:.4f} | Brier={s.brier_score:.4f} | Lift@10%={s.lift_top_decile}x")

        # Stage 8: Decision Threshold Optimization
        print("\n[Stage 8/13] Decision Threshold Optimization (Precision Floor & Alert Triage)...")
        champion_model = models_dict["calibrated_lgbm_v2"]
        y_holdout = holdout_df["is_breakout"].values.astype(int)
        y_prob_champ = champion_model.predict_proba(X_holdout)[:, 1]

        optimizer = DecisionThresholdOptimizer(target_precision_floor=0.40)
        threshold_res = optimizer.optimize(y_holdout, y_prob_champ)
        report_data["stages"]["stage_8_threshold"] = "PASSED"
        report_data["metrics"] = {
            "optimal_threshold": threshold_res.optimal_threshold,
            "precision": threshold_res.optimal_metrics.precision,
            "recall": threshold_res.optimal_metrics.recall,
            "lift_over_base_rate": threshold_res.optimal_metrics.lift_over_base_rate,
            "pr_auc_calibrated_v2": scores[-1].pr_auc,
            "brier_score_calibrated_v2": scores[-1].brier_score,
        }
        print(f"  [+] Optimal Alert Threshold: tau* = {threshold_res.optimal_threshold:.2f} | Precision = {threshold_res.optimal_metrics.precision:.1%} | Lift = {threshold_res.optimal_metrics.lift_over_base_rate:.2f}x")

        # Stage 9: Artifact Serialization & Verification
        print("\n[Stage 9/13] Model Artifact Serialization & Deserialization...")
        models_dir = root_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(champion_model, models_dir / "calibrated_lgbm_v2.joblib")
        joblib.dump(fitted_pipe, models_dir / "feature_pipeline.joblib")
        DecisionThresholdOptimizer.save(threshold_res, models_dir / "threshold_config.json")

        loaded_model = joblib.load(models_dir / "calibrated_lgbm_v2.joblib")
        loaded_pipe = joblib.load(models_dir / "feature_pipeline.joblib")
        test_preds = loaded_model.predict_proba(loaded_pipe.transform(holdout_df[:5]))
        assert test_preds.shape == (5, 2)
        report_data["stages"]["stage_9_serialization"] = "PASSED"
        print(f"  [+] Serialized and verified model artifacts in {models_dir.resolve()}.")

        # Stage 10: SHAP TreeExplainer Interpretability
        print("\n[Stage 10/13] SHAP TreeExplainer Interpretability & Additivity...")
        explainer = ModelExplainer(champion_model, feature_names=feature_names)
        assert explainer.verify_additivity(X_holdout[:20], atol=1e-2)
        df_imp = explainer.explain_global(X_holdout[:50], top_k=5)
        local_exp = explainer.explain_local(X_holdout[0], top_k=3)
        assert len(df_imp) == 5
        assert len(local_exp["top_positive_drivers"]) > 0 or len(local_exp["top_negative_drivers"]) > 0
        report_data["stages"]["stage_10_shap"] = "PASSED"
        print(f"  [+] SHAP verified: additivity preserved, top global driver: {df_imp.iloc[0]['feature']}.")

        # Stage 11: PSI Drift Monitoring
        print("\n[Stage 11/13] Population Stability Index (PSI) Drift Monitoring...")
        train_feat_df = pd.DataFrame(X_train, columns=feature_names)
        holdout_feat_df = pd.DataFrame(X_holdout, columns=feature_names)
        drift_monitor = FeatureDriftMonitor(num_bins=10)
        drift_monitor.fit_baseline(train_feat_df, feature_cols=feature_names)
        drift_profile_path = models_dir / "baseline_drift_profile.json"
        drift_monitor.save_baseline_profile(drift_profile_path)
        report = drift_monitor.compute_drift(holdout_feat_df)
        report_data["metrics"]["holdout_mean_psi"] = report.mean_psi
        report_data["metrics"]["holdout_drift_status"] = report.overall_status
        report_data["stages"]["stage_11_drift"] = "PASSED"
        print(f"  [+] PSI Drift Monitor verified: holdout mean PSI = {report.mean_psi:.4f} (Status: {report.overall_status.upper()}).")

        # Stage 12: Streamlit Dashboard Headless Check
        print("\n[Stage 12/13] Streamlit Observability Dashboard Headless Verification...")
        from app.streamlit_app import load_data_and_artifacts
        st_df, st_model, st_pipe, st_thresh, st_metrics = load_data_and_artifacts()
        assert len(st_df) > 0
        assert st_model is not None
        assert st_pipe is not None
        report_data["stages"]["stage_12_streamlit"] = "PASSED"
        print(f"  [+] Streamlit app verified: {len(st_df)} records, model pipeline loaded headlessly.")

        # Stage 13: Summary Report Generation
        print("\n[Stage 13/13] Generating Final Master Verification Summary...")
        report_data["verdict"] = "PASSED"
        report_data["stages"]["stage_13_summary"] = "PASSED"

        summary_json = reports_dir / "verification_report.json"
        with open(summary_json, "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)

        summary_md = reports_dir / "verification_summary.md"
        with open(summary_md, "w", encoding="utf-8") as f:
            f.write("# Competitive Velocity Engine - Verification Summary Report\n\n")
            f.write(f"- **Verdict**: `{report_data['verdict']}`\n")
            f.write(f"- **Timestamp**: `{report_data['timestamp']}`\n\n")
            f.write("## Stages Completed (13/13)\n\n")
            for stage, st in report_data["stages"].items():
                f.write(f"- `{stage}`: **{st}**\n")
            f.write("\n## Benchmark Metrics (Out-of-Time Holdout)\n\n")
            for k, v in report_data["metrics"].items():
                f.write(f"- `{k}`: `{v}`\n")

        print(f"  [+] Final summary reports written to {summary_json} and {summary_md}")
        print("\n" + "=" * 85)
        print("=== ALL 13 PIPELINE VERIFICATION STAGES PASSED (EXIT CODE 0) ===")
        print("=" * 85 + "\n")
        return 0

    except Exception as e:
        print(f"\n[!] Verification Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        report_data["verdict"] = "FAILED"
        report_data["error"] = str(e)
        with open(reports_dir / "verification_report.json", "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2)
        return 1


if __name__ == "__main__":
    sys.exit(run_pipeline())
