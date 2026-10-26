"""
Population Stability Index (PSI) Drift Monitoring Module.

Quantifies distribution shifts between training baseline distributions and
post-deployment inference distributions for tabular and embedding features.
Alerts when PSI exceeds 0.20 (critical drift) or 0.10 (moderate drift).
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd


@dataclass
class FeatureDriftResult:
    """Drift evaluation result for a single feature."""
    feature_name: str
    psi: float
    status: str  # "stable", "warning", "drift_alert"
    baseline_distribution: List[float]
    target_distribution: List[float]
    bin_edges: Optional[List[float]] = None
    categories: Optional[List[str]] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class DriftReport:
    """Aggregated drift report across all monitored features."""
    overall_status: str  # "stable", "warning", "drift_alert"
    mean_psi: float
    max_psi: float
    drifted_features: List[str]
    warning_features: List[str]
    feature_results: Dict[str, FeatureDriftResult]
    total_features_monitored: int
    baseline_sample_size: int
    target_sample_size: int

    def to_dict(self) -> Dict[str, Any]:
        res = asdict(self)
        res["feature_results"] = {
            k: v.to_dict() if hasattr(v, "to_dict") else v
            for k, v in self.feature_results.items()
        }
        return res

    def to_dataframe(self) -> pd.DataFrame:
        rows = []
        for name, res in self.feature_results.items():
            rows.append({
                "feature": name,
                "psi": round(res.psi, 4),
                "status": res.status,
            })
        df = pd.DataFrame(rows)
        return df.sort_values(by="psi", ascending=False).reset_index(drop=True)


def calculate_psi(
    baseline: Union[pd.Series, np.ndarray, List[float]],
    target: Union[pd.Series, np.ndarray, List[float]],
    num_bins: int = 10,
    epsilon: float = 1e-4,
) -> Tuple[float, List[float], List[float], List[float]]:
    """
    Calculate Population Stability Index (PSI) for continuous numeric distributions.

    PSI Formula:
        PSI = sum((target_prop - base_prop) * ln(target_prop / base_prop))

    Args:
        baseline: Baseline (training) distribution values.
        target: Target (inference/current) distribution values.
        num_bins: Number of quantiles for binning (default: 10).
        epsilon: Smoothing constant to prevent division by zero or log(0).

    Returns:
        Tuple of (psi_value, baseline_proportions, target_proportions, bin_edges).
    """
    base_arr = np.asarray(baseline, dtype=float)
    target_arr = np.asarray(target, dtype=float)

    # Filter out NaNs
    base_clean = base_arr[~np.isnan(base_arr)]
    target_clean = target_arr[~np.isnan(target_arr)]

    if len(base_clean) == 0 or len(target_clean) == 0:
        return 0.0, [], [], []

    # If constant distribution
    if np.all(base_clean == base_clean[0]):
        if np.all(target_clean == base_clean[0]):
            return 0.0, [1.0], [1.0], [float(base_clean[0])]
        else:
            return 1.0, [1.0], [0.0], [float(base_clean[0])]

    # Determine quantile bin edges based strictly on baseline distribution
    percentiles = np.linspace(0, 100, num_bins + 1)
    bin_edges = np.percentile(base_clean, percentiles)
    # Ensure strictly monotonic bin edges
    bin_edges = np.unique(bin_edges)

    if len(bin_edges) < 2:
        # Fall back to equal-width bins if percentiles collapsed
        min_v, max_v = float(np.min(base_clean)), float(np.max(base_clean))
        if min_v == max_v:
            max_v += 1.0
        bin_edges = np.linspace(min_v, max_v, num_bins + 1)

    # Set outer edges to -inf and inf to capture all future out-of-range values
    extended_edges = bin_edges.copy()
    extended_edges[0] = -np.inf
    extended_edges[-1] = np.inf

    # Count occurrences in each bin
    base_counts, _ = np.histogram(base_clean, bins=extended_edges)
    target_counts, _ = np.histogram(target_clean, bins=extended_edges)

    # Normalize to proportions
    base_prop = base_counts / float(len(base_clean))
    target_prop = target_counts / float(len(target_clean))

    # Apply epsilon smoothing
    base_prop = np.where(base_prop == 0, epsilon, base_prop)
    target_prop = np.where(target_prop == 0, epsilon, target_prop)

    # Re-normalize after smoothing
    base_prop = base_prop / np.sum(base_prop)
    target_prop = target_prop / np.sum(target_prop)

    # Compute PSI
    psi_vector = (target_prop - base_prop) * np.log(target_prop / base_prop)
    psi_value = float(np.sum(psi_vector))

    return (
        max(0.0, psi_value),
        base_prop.tolist(),
        target_prop.tolist(),
        bin_edges.tolist(),
    )


def calculate_categorical_psi(
    baseline: Union[pd.Series, List[str]],
    target: Union[pd.Series, List[str]],
    epsilon: float = 1e-4,
) -> Tuple[float, List[float], List[float], List[str]]:
    """Calculate PSI for discrete/categorical feature distributions."""
    base_series = pd.Series(baseline).dropna().astype(str)
    target_series = pd.Series(target).dropna().astype(str)

    all_categories = sorted(list(set(base_series.unique()).union(set(target_series.unique()))))
    if not all_categories:
        return 0.0, [], [], []

    base_counts = base_series.value_counts(normalize=True).to_dict()
    target_counts = target_series.value_counts(normalize=True).to_dict()

    base_prop = np.array([base_counts.get(cat, epsilon) for cat in all_categories], dtype=float)
    target_prop = np.array([target_counts.get(cat, epsilon) for cat in all_categories], dtype=float)

    base_prop = base_prop / np.sum(base_prop)
    target_prop = target_prop / np.sum(target_prop)

    psi_vector = (target_prop - base_prop) * np.log(target_prop / base_prop)
    psi_value = float(np.sum(psi_vector))

    return max(0.0, psi_value), base_prop.tolist(), target_prop.tolist(), all_categories


class FeatureDriftMonitor:
    """
    Production Feature Drift Monitor using Population Stability Index (PSI).

    Thresholds:
      - PSI < 0.10: Stable (normal operational variance)
      - 0.10 <= PSI < 0.20: Warning (moderate shift, flag for inspection)
      - PSI >= 0.20: Drift Alert (significant divergence, retraining trigger)
    """

    PSI_WARNING_THRESHOLD = 0.10
    PSI_ALERT_THRESHOLD = 0.20

    def __init__(self, num_bins: int = 10, epsilon: float = 1e-4):
        self.num_bins = num_bins
        self.epsilon = epsilon
        self.baseline_profile: Dict[str, Dict[str, Any]] = {}
        self.baseline_sample_size: int = 0

    def fit_baseline(
        self,
        df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        categorical_cols: Optional[List[str]] = None,
    ) -> FeatureDriftMonitor:
        """
        Build and store baseline quantile bins and proportions from training data.
        """
        if feature_cols is None:
            feature_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c])]

        categorical_cols = categorical_cols or []
        self.baseline_sample_size = len(df)
        self.baseline_profile = {}

        for col in feature_cols:
            if col in df.columns:
                series = df[col].dropna()
                if len(series) == 0:
                    continue

                if col in categorical_cols or series.nunique() <= 5:
                    # Categorical treatment
                    cats = sorted(series.unique().tolist())
                    counts = series.value_counts(normalize=True).to_dict()
                    self.baseline_profile[col] = {
                        "type": "categorical",
                        "categories": [str(c) for c in cats],
                        "proportions": [float(counts.get(c, self.epsilon)) for c in cats],
                    }
                else:
                    # Continuous quantile binning
                    percentiles = np.linspace(0, 100, self.num_bins + 1)
                    bin_edges = np.percentile(series.values, percentiles)
                    bin_edges = np.unique(bin_edges)
                    if len(bin_edges) < 2:
                        min_v, max_v = float(series.min()), float(series.max())
                        if min_v == max_v:
                            max_v += 1.0
                        bin_edges = np.linspace(min_v, max_v, self.num_bins + 1)

                    extended_edges = bin_edges.copy()
                    extended_edges[0] = -np.inf
                    extended_edges[-1] = np.inf

                    counts, _ = np.histogram(series.values, bins=extended_edges)
                    props = counts / float(len(series))
                    props = np.where(props == 0, self.epsilon, props)
                    props = props / np.sum(props)

                    self.baseline_profile[col] = {
                        "type": "numeric",
                        "bin_edges": bin_edges.tolist(),
                        "proportions": props.tolist(),
                    }

        return self

    def compute_drift(self, current_df: pd.DataFrame) -> DriftReport:
        """
        Compare current inference data against the stored baseline profile.
        """
        if not self.baseline_profile:
            raise ValueError("Baseline profile is empty. Call fit_baseline() or load_baseline_profile() first.")

        feature_results: Dict[str, FeatureDriftResult] = {}
        psi_values: List[float] = []
        drifted_features: List[str] = []
        warning_features: List[str] = []

        for col, prof in self.baseline_profile.items():
            if col not in current_df.columns:
                continue

            target_series = current_df[col].dropna()
            if len(target_series) == 0:
                continue

            if prof["type"] == "categorical":
                cats = prof["categories"]
                base_prop = np.array(prof["proportions"], dtype=float)

                target_counts = target_series.astype(str).value_counts(normalize=True).to_dict()
                target_prop = np.array([target_counts.get(cat, self.epsilon) for cat in cats], dtype=float)
                target_prop = target_prop / np.sum(target_prop)

                psi_vec = (target_prop - base_prop) * np.log(target_prop / base_prop)
                psi_val = float(max(0.0, np.sum(psi_vec)))

                status = (
                    "drift_alert" if psi_val >= self.PSI_ALERT_THRESHOLD
                    else "warning" if psi_val >= self.PSI_WARNING_THRESHOLD
                    else "stable"
                )

                result = FeatureDriftResult(
                    feature_name=col,
                    psi=psi_val,
                    status=status,
                    baseline_distribution=base_prop.tolist(),
                    target_distribution=target_prop.tolist(),
                    categories=cats,
                )
            else:
                bin_edges = np.array(prof["bin_edges"], dtype=float)
                base_prop = np.array(prof["proportions"], dtype=float)

                extended_edges = bin_edges.copy()
                extended_edges[0] = -np.inf
                extended_edges[-1] = np.inf

                target_counts, _ = np.histogram(target_series.values, bins=extended_edges)
                target_prop = target_counts / float(len(target_series))
                target_prop = np.where(target_prop == 0, self.epsilon, target_prop)
                target_prop = target_prop / np.sum(target_prop)

                psi_vec = (target_prop - base_prop) * np.log(target_prop / base_prop)
                psi_val = float(max(0.0, np.sum(psi_vec)))

                status = (
                    "drift_alert" if psi_val >= self.PSI_ALERT_THRESHOLD
                    else "warning" if psi_val >= self.PSI_WARNING_THRESHOLD
                    else "stable"
                )

                result = FeatureDriftResult(
                    feature_name=col,
                    psi=psi_val,
                    status=status,
                    baseline_distribution=base_prop.tolist(),
                    target_distribution=target_prop.tolist(),
                    bin_edges=bin_edges.tolist(),
                )

            feature_results[col] = result
            psi_values.append(psi_val)
            if status == "drift_alert":
                drifted_features.append(col)
            elif status == "warning":
                warning_features.append(col)

        mean_psi = float(np.mean(psi_values)) if psi_values else 0.0
        max_psi = float(np.max(psi_values)) if psi_values else 0.0

        overall_status = (
            "drift_alert" if len(drifted_features) > 0
            else "warning" if len(warning_features) > 0
            else "stable"
        )

        return DriftReport(
            overall_status=overall_status,
            mean_psi=mean_psi,
            max_psi=max_psi,
            drifted_features=drifted_features,
            warning_features=warning_features,
            feature_results=feature_results,
            total_features_monitored=len(feature_results),
            baseline_sample_size=self.baseline_sample_size,
            target_sample_size=len(current_df),
        )

    def save_baseline_profile(self, filepath: Union[str, Path]) -> None:
        """Persist the baseline profile to a JSON file."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "num_bins": self.num_bins,
            "epsilon": self.epsilon,
            "baseline_sample_size": self.baseline_sample_size,
            "profile": self.baseline_profile,
        }
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)

    def load_baseline_profile(self, filepath: Union[str, Path]) -> FeatureDriftMonitor:
        """Load a persisted baseline profile from JSON."""
        filepath = Path(filepath)
        with open(filepath, "r", encoding="utf-8") as f:
            payload = json.load(f)
        self.num_bins = payload["num_bins"]
        self.epsilon = payload["epsilon"]
        self.baseline_sample_size = payload["baseline_sample_size"]
        self.baseline_profile = payload["profile"]
        return self
