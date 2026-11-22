"""Feature engineering and point-in-time assembly subsystem."""

from intel_velocity.features.feature_pipeline import (
    CORE_BINARY_FEATURES,
    CORE_NUMERIC_FEATURES,
    FeaturePipeline,
    chronological_train_test_split,
    get_time_series_cv,
)
from intel_velocity.features.text_pipeline import TextEmbeddingPipeline

__all__ = [
    "TextEmbeddingPipeline",
    "FeaturePipeline",
    "chronological_train_test_split",
    "get_time_series_cv",
    "CORE_NUMERIC_FEATURES",
    "CORE_BINARY_FEATURES",
]
