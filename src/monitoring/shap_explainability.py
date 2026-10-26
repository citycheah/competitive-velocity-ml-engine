"""
SHAP Explainability Module for Competitive Velocity Engine.

Leverages TreeExplainer on LightGBM tree models to generate global feature importance
and local waterfall attributions for individual video breakout predictions.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import shap


class ModelExplainer:
    """
    SHAP-based model explainer for LightGBM models.
    Supports both raw LGBMClassifier and CalibratedClassifierCV wrappers.
    """

    def __init__(self, model: Any, feature_names: Optional[List[str]] = None):
        self.raw_model = model
        self.feature_names = feature_names or []
        self.tree_model = self._extract_tree_model(model)
        self.explainer = shap.TreeExplainer(self.tree_model)
        self.expected_value = self._extract_expected_value()

    def _extract_tree_model(self, model: Any) -> Any:
        """Extract underlying tree model if model is CalibratedClassifierCV."""
        if hasattr(model, "calibrated_classifiers_") and len(model.calibrated_classifiers_) > 0:
            # CalibratedClassifierCV wrapper
            cal_clf = model.calibrated_classifiers_[0]
            if hasattr(cal_clf, "estimator"):
                return cal_clf.estimator
            elif hasattr(cal_clf, "base_estimator"):
                return cal_clf.base_estimator
        return model

    def _extract_expected_value(self) -> float:
        """Extract scalar expected base value from TreeExplainer."""
        exp_val = self.explainer.expected_value
        if isinstance(exp_val, (list, np.ndarray)):
            if len(exp_val) > 1:
                return float(exp_val[1])  # Positive class for binary classification
            return float(exp_val[0])
        return float(exp_val)

    def compute_shap_values(self, X: Union[np.ndarray, pd.DataFrame]) -> np.ndarray:
        """
        Compute SHAP values for feature matrix X.
        Returns shape (N, D) where N is samples and D is features.
        """
        X_arr = np.asarray(X)
        shap_values = self.explainer.shap_values(X_arr)

        if isinstance(shap_values, list):
            # Binary classification in TreeExplainer often returns [shap_neg, shap_pos]
            if len(shap_values) == 2:
                return np.asarray(shap_values[1])
            return np.asarray(shap_values[0])
        elif isinstance(shap_values, np.ndarray) and shap_values.ndim == 3:
            # Some shap versions return (N, D, 2)
            return shap_values[:, :, 1]
        return np.asarray(shap_values)

    def explain_global(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        top_k: int = 20,
    ) -> pd.DataFrame:
        """
        Compute global feature importance based on mean absolute SHAP values.
        """
        X_arr = np.asarray(X)
        shap_vals = self.compute_shap_values(X_arr)
        mean_abs_shap = np.mean(np.abs(shap_vals), axis=0)

        names = self.feature_names
        if not names or len(names) != X_arr.shape[1]:
            names = [f"feature_{i}" for i in range(X_arr.shape[1])]

        df_importance = pd.DataFrame({
            "feature": names,
            "mean_abs_shap": mean_abs_shap,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)

        return df_importance.head(top_k)

    def explain_local(
        self,
        x: Union[np.ndarray, pd.Series, List[float]],
        top_k: int = 5,
    ) -> Dict[str, Any]:
        """
        Compute local attribution for a single video prediction.

        Returns:
            Dict containing base_value, prediction, and top positive/negative drivers.
        """
        x_arr = np.asarray(x).reshape(1, -1)
        shap_val = self.compute_shap_values(x_arr)[0]

        names = self.feature_names
        if not names or len(names) != len(shap_val):
            names = [f"feature_{i}" for i in range(len(shap_val))]

        contributions = [
            {"feature": name, "shap_value": float(val), "feature_value": float(x_val)}
            for name, val, x_val in zip(names, shap_val, x_arr[0])
        ]

        # Sort into top positive and top negative contributors
        sorted_contribs = sorted(contributions, key=lambda c: c["shap_value"], reverse=True)
        top_pos = [c for c in sorted_contribs if c["shap_value"] > 0][:top_k]
        top_neg = [c for c in sorted_contribs if c["shap_value"] < 0][-top_k:]

        predicted_margin = self.expected_value + float(np.sum(shap_val))

        return {
            "expected_value": self.expected_value,
            "predicted_margin": predicted_margin,
            "sum_shap": float(np.sum(shap_val)),
            "top_positive_drivers": top_pos,
            "top_negative_drivers": top_neg,
            "all_contributions": sorted_contribs,
        }

    def verify_additivity(
        self,
        X: Union[np.ndarray, pd.DataFrame],
        atol: float = 1e-3,
    ) -> bool:
        """
        Verify that sum of SHAP values plus base value equals tree margin output.
        """
        X_arr = np.asarray(X)
        shap_vals = self.compute_shap_values(X_arr)
        tree_preds = self.tree_model.predict(X_arr, raw_score=True)

        if tree_preds.ndim > 1:
            tree_preds = tree_preds[:, 1] if tree_preds.shape[1] > 1 else tree_preds[:, 0]

        reconstructed = self.expected_value + np.sum(shap_vals, axis=1)
        return bool(np.allclose(reconstructed, tree_preds, atol=atol))
