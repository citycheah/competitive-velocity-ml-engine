"""Evaluation metrics, calibration diagnostics, and model ladder benchmarking."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

logger = logging.getLogger("diagnostics")


@dataclass
class ModelBenchmarkScore:
    model_name: str
    feature_set: str
    pr_auc: float
    roc_auc: float
    brier_score: float
    log_loss: float
    precision_top_decile: float
    lift_top_decile: float
    calibrated: bool


class ModelEvaluator:
    """Computes comprehensive evaluation metrics for competitive breakout ranking."""

    @staticmethod
    def evaluate_predictions(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        model_name: str = "Model",
        feature_set: str = "Default",
        calibrated: bool = False,
    ) -> ModelBenchmarkScore:
        """Evaluates probabilistic predictions against ground truth binary breakout labels."""
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        base_rate = float(np.mean(y_true))

        # 1. Primary Ranking & Discrimination Metrics
        pr_auc = float(average_precision_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.0
        try:
            roc_auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else 0.5
        except ValueError:
            roc_auc = 0.5

        # 2. Probability Calibration Metrics (Brier score & Log Loss)
        # Brier score: MSE of predicted probability vs binary truth (lower is better, 0 is perfect)
        brier = float(brier_score_loss(y_true, y_prob))
        try:
            clipped_prob = np.clip(y_prob, 1e-6, 1.0 - 1e-6)
            loss = float(log_loss(y_true, clipped_prob))
        except Exception:
            loss = 1.0

        # 3. Top Decile (Top 10%) Alert Triage Precision & Lift
        n_top = max(1, int(len(y_true) * 0.10))
        top_indices = np.argsort(y_prob)[::-1][:n_top]
        precision_top = float(np.mean(y_true[top_indices]))
        lift_top = float(precision_top / base_rate) if base_rate > 0 else 1.0

        score = ModelBenchmarkScore(
            model_name=model_name,
            feature_set=feature_set,
            pr_auc=round(pr_auc, 4),
            roc_auc=round(roc_auc, 4),
            brier_score=round(brier, 4),
            log_loss=round(loss, 4),
            precision_top_decile=round(precision_top, 4),
            lift_top_decile=round(lift_top, 2),
            calibrated=calibrated,
        )
        return score

    @staticmethod
    def format_ladder_table(scores: List[ModelBenchmarkScore]) -> pd.DataFrame:
        """Formats list of model scores into a clean benchmark comparison DataFrame."""
        records = []
        for s in scores:
            records.append({
                "Model Stage": s.model_name,
                "Feature Set": s.feature_set,
                "PR-AUC": s.pr_auc,
                "ROC-AUC": s.roc_auc,
                "Brier Score (Calib)": s.brier_score,
                "Precision @ Top 10%": s.precision_top_decile,
                "Lift @ Top 10%": f"{s.lift_top_decile}x",
                "Calibrated": "Yes" if s.calibrated else "No",
            })
        return pd.DataFrame(records)
