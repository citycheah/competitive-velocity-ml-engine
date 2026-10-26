"""Point-in-time feature extraction, transformation, and temporal splitting pipeline.

Enforces strict physical t0 quarantine validation to ensure zero look-ahead maturity bias.
"""

from __future__ import annotations

import logging
from typing import Generator, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.model_selection import TimeSeriesSplit

from src.features.text_pipeline import TextEmbeddingPipeline
from src.ingest.schemas import T0QuarantineValidator

logger = logging.getLogger("feature_pipeline")


# The 14 High-Signal Core Tabular Features Knowable at t0
CORE_NUMERIC_FEATURES = [
    "channel_historical_avg_engagement_30d",
    "channel_historical_std_engagement_30d",
    "channel_subscriber_count",
    "channel_upload_blitz_24h",
    "channel_upload_blitz_7d",
    "hours_since_last_upload",
    "duration_seconds",
    "channel_format_volatility_30d",
    "title_length_chars",
    "title_word_count",
    "publish_hour_of_day",
    "publish_day_of_week",
]

CORE_BINARY_FEATURES = [
    "is_short",
    "is_weekend",
]


class FeaturePipeline(BaseEstimator, TransformerMixin):
    """End-to-end feature preparation pipeline.
    
    Transforms raw mart records into a unified modeling design matrix X by combining:
      1. 14 high-signal tabular features (cadence, momentum, format, temporal).
      2. Domain format-interaction features (v2 enhancement).
      3. 16-component TruncatedSVD dense text embeddings of video titles.
      
    Guarantees strict t0 quarantine enforcement via T0QuarantineValidator.
    """

    def __init__(
        self,
        n_text_components: int = 16,
        include_format_interaction: bool = True,
        random_state: int = 42,
    ):
        self.n_text_components = n_text_components
        self.include_format_interaction = include_format_interaction
        self.random_state = random_state

        self.text_pipeline = TextEmbeddingPipeline(
            n_components=self.n_text_components,
            random_state=self.random_state,
        )
        self.is_fitted = False
        self.feature_names_: List[str] = []

    def _engineer_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Engineers non-linear format and cadence interaction features strictly observable at t0."""
        df_out = df.copy()

        # Format surprise: when a channel known for long-form drops a short, or vice-versa
        # Volatility near 0 means monochromatic channel format; near 1 means mixed.
        # Deviation = |is_short - (1.0 - format_volatility)|
        is_short_float = df_out["is_short"].astype(float)
        volatility = df_out["channel_format_volatility_30d"].astype(float)
        df_out["format_interaction_surprise"] = np.round(np.abs(is_short_float - (1.0 - volatility)), 4)

        # Cadence blitz acceleration: ratio of 24h blitz relative to 7d blitz baseline
        blitz_7d = df_out["channel_upload_blitz_7d"].astype(float)
        blitz_24h = df_out["channel_upload_blitz_24h"].astype(float)
        # Avoid division by zero: blitz_7d / 7.0 vs blitz_24h
        daily_7d_rate = blitz_7d / 7.0
        df_out["cadence_blitz_acceleration"] = np.where(
            daily_7d_rate > 0,
            np.round(blitz_24h / (daily_7d_rate + 1e-4), 4),
            np.round(blitz_24h, 4),
        )

        return df_out

    def fit(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> FeaturePipeline:
        """Fits transformers (e.g. text vectorizer and SVD) strictly on training records."""
        titles = df["title"].fillna("")
        self.text_pipeline.fit(titles)

        # Assemble output feature names
        feature_names = list(CORE_NUMERIC_FEATURES) + list(CORE_BINARY_FEATURES)
        if self.include_format_interaction:
            feature_names.extend(["format_interaction_surprise", "cadence_blitz_acceleration"])
        feature_names.extend(self.text_pipeline.get_feature_names_out())

        # Validate that the assembled feature set contains ZERO mature leakage metrics
        T0QuarantineValidator.validate_feature_columns(feature_names)

        self.feature_names_ = feature_names
        self.is_fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transforms input DataFrame into numeric design matrix X."""
        if not self.is_fitted:
            raise RuntimeError("FeaturePipeline must be fitted before calling transform().")

        df_proc = df.copy()
        if self.include_format_interaction:
            df_proc = self._engineer_interaction_features(df_proc)

        # 1. Extract tabular features and ensure numeric types with defensive zero imputation
        tabular_cols = list(CORE_NUMERIC_FEATURES) + list(CORE_BINARY_FEATURES)
        if self.include_format_interaction:
            tabular_cols.extend(["format_interaction_surprise", "cadence_blitz_acceleration"])

        tabular_matrix = df_proc[tabular_cols].fillna(0.0).astype(float).values

        # 2. Extract text SVD embeddings
        titles = df_proc["title"].fillna("")
        text_matrix = self.text_pipeline.transform(titles)

        # 3. Concatenate horizontally
        X = np.hstack([tabular_matrix, text_matrix])
        return X

    def fit_transform(self, df: pd.DataFrame, y: Optional[np.ndarray] = None) -> np.ndarray:
        """Fits on training DataFrame and returns design matrix X."""
        return self.fit(df, y).transform(df)

    def get_feature_names_out(self) -> List[str]:
        """Returns ordered list of all output feature names."""
        return self.feature_names_


def chronological_train_test_split(
    df: pd.DataFrame,
    test_ratio: float = 0.15,
    maturity_days_filter: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Partitions dataset chronologically into training split and untouched holdout test split.
    
    Args:
        df: Input DataFrame containing published_epoch or published_at.
        test_ratio: Fraction of the most recent chronological records reserved for test (default 0.15).
        maturity_days_filter: Optional days threshold. If provided, training split excludes records
            younger than maturity_days_filter days to eliminate lifetime accumulation bias.
            
    Returns:
        Tuple of (train_df, test_df) sorted chronologically.
    """
    if "published_epoch" in df.columns:
        df_sorted = df.sort_values("published_epoch", ascending=True).reset_index(drop=True)
    else:
        df_sorted = df.sort_values("published_at", ascending=True).reset_index(drop=True)

    n_total = len(df_sorted)
    n_test = max(1, int(n_total * test_ratio))
    n_train = n_total - n_test

    train_df = df_sorted.iloc[:n_train].copy()
    test_df = df_sorted.iloc[n_train:].copy()

    # Optional asymptotic maturity cutoff on training split
    if maturity_days_filter is not None and maturity_days_filter > 0:
        max_epoch = df_sorted["published_epoch"].max()
        cutoff_epoch = max_epoch - (maturity_days_filter * 86400)
        filtered_train = train_df[train_df["published_epoch"] <= cutoff_epoch].copy()
        if len(filtered_train) > 0:
            train_df = filtered_train
            logger.info(
                "Applied %d-day maturity cutoff: retained %d training records (<= epoch %d)",
                maturity_days_filter,
                len(train_df),
                cutoff_epoch,
            )
        else:
            logger.warning(
                "Maturity filter of %d days eliminated all training records (dataset span %.1f days). "
                "Retaining unfiltered chronological training split to preserve pipeline execution.",
                maturity_days_filter,
                (max_epoch - df_sorted["published_epoch"].min()) / 86400,
            )

    logger.info(
        "Chronological split completed: Train=%d records, Test=%d records (Holdout=%.1f%%)",
        len(train_df),
        len(test_df),
        test_ratio * 100,
    )
    return train_df, test_df


def get_time_series_cv(
    train_df: pd.DataFrame,
    n_splits: int = 5,
) -> TimeSeriesSplit:
    """Returns TimeSeriesSplit cross-validator over chronologically ordered training records."""
    return TimeSeriesSplit(n_splits=n_splits)
