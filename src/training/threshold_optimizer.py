"""Decision threshold optimization for competitive breakout alerting.

Optimizes decision thresholds under alert fatigue constraints (enforcing precision floor / lift
over baseline) rather than accepting default 0.5 classification cuts.
"""

from __future__ import annotations

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn.metrics import precision_recall_curve

logger = logging.getLogger("threshold_optimizer")


@dataclass
class ThresholdMetrics:
    threshold: float
    precision: float
    recall: float
    f1_score: float
    lift_over_base_rate: float
    alert_volume_pct: float
    true_positives: int
    false_positives: int
    false_negatives: int
    true_negatives: int


@dataclass
class OptimizedThresholdResult:
    target_precision_floor: float
    base_rate: float
    optimal_threshold: float
    optimal_metrics: ThresholdMetrics
    precision_floor_satisfied: bool
    grid_metrics: List[Dict[str, float]]


class DecisionThresholdOptimizer:
    """Finds optimal decision probability threshold maximizing triage utility."""

    def __init__(
        self,
        target_precision_floor: float = 0.40,
        grid_steps: int = 100,
        min_threshold: float = 0.05,
        max_threshold: float = 0.95,
    ):
        self.target_precision_floor = target_precision_floor
        self.grid_steps = grid_steps
        self.min_threshold = min_threshold
        self.max_threshold = max_threshold

    def optimize(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
    ) -> OptimizedThresholdResult:
        """Evaluates threshold grid and identifies optimal decision cut."""
        y_true = np.asarray(y_true).astype(int)
        y_prob = np.asarray(y_prob).astype(float)

        base_rate = float(np.mean(y_true))
        thresholds = np.linspace(self.min_threshold, self.max_threshold, self.grid_steps)

        candidates: List[ThresholdMetrics] = []
        grid_metrics: List[Dict[str, float]] = []

        for tau in thresholds:
            y_pred = (y_prob >= tau).astype(int)

            tp = int(np.sum((y_true == 1) & (y_pred == 1)))
            fp = int(np.sum((y_true == 0) & (y_pred == 1)))
            fn = int(np.sum((y_true == 1) & (y_pred == 0)))
            tn = int(np.sum((y_true == 0) & (y_pred == 0)))

            pred_positives = tp + fp
            precision = float(tp / pred_positives) if pred_positives > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = (
                float(2 * precision * recall / (precision + recall))
                if (precision + recall) > 0
                else 0.0
            )
            lift = float(precision / base_rate) if base_rate > 0 else 1.0
            alert_pct = float(pred_positives / len(y_true))

            m = ThresholdMetrics(
                threshold=round(float(tau), 4),
                precision=round(precision, 4),
                recall=round(recall, 4),
                f1_score=round(f1, 4),
                lift_over_base_rate=round(lift, 4),
                alert_volume_pct=round(alert_pct, 4),
                true_positives=tp,
                false_positives=fp,
                false_negatives=fn,
                true_negatives=tn,
            )
            candidates.append(m)
            grid_metrics.append({
                "threshold": m.threshold,
                "precision": m.precision,
                "recall": m.recall,
                "f1_score": m.f1_score,
                "lift": m.lift_over_base_rate,
                "alert_pct": m.alert_volume_pct,
            })

        # Selection strategy:
        # 1. Filter candidates satisfying the precision floor with non-zero recall
        qualifying = [
            c for c in candidates
            if c.precision >= self.target_precision_floor and c.recall > 0.0 and c.true_positives >= 3
        ]

        floor_satisfied = len(qualifying) > 0
        if floor_satisfied:
            # Maximize recall among those meeting the precision floor
            optimal = max(qualifying, key=lambda c: (c.recall, c.f1_score))
        else:
            # Fallback: select threshold maximizing F1 score, or highest precision with non-zero TP
            valid = [c for c in candidates if c.true_positives > 0]
            if valid:
                optimal = max(valid, key=lambda c: (c.f1_score, c.precision))
            else:
                optimal = candidates[len(candidates) // 2]

        logger.info(
            "Optimal threshold=%.4f | Precision=%.4f | Recall=%.4f | Lift=%.2fx (Floor Satisfied: %s)",
            optimal.threshold,
            optimal.precision,
            optimal.recall,
            optimal.lift_over_base_rate,
            floor_satisfied,
        )

        return OptimizedThresholdResult(
            target_precision_floor=self.target_precision_floor,
            base_rate=round(base_rate, 4),
            optimal_threshold=optimal.threshold,
            optimal_metrics=optimal,
            precision_floor_satisfied=floor_satisfied,
            grid_metrics=grid_metrics,
        )

    @staticmethod
    def save(result: OptimizedThresholdResult, output_path: Path | str) -> None:
        """Serializes threshold optimization result to JSON."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, indent=2)

    @staticmethod
    def load(input_path: Path | str) -> Dict:
        """Loads serialized threshold configuration."""
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)
