"""Feature engineering and point-in-time assembly subsystem."""

from src.features.text_pipeline import TextEmbeddingPipeline
from src.features.feature_pipeline import (
    FeaturePipeline,
    chronological_train_test_split,
    get_time_series_cv,
    CORE_NUMERIC_FEATURES,
    CORE_BINARY_FEATURES,
)

__all__ = [
    "TextEmbeddingPipeline",
    "FeaturePipeline",
    "chronological_train_test_split",
    "get_time_series_cv",
    "CORE_NUMERIC_FEATURES",
    "CORE_BINARY_FEATURES",
]
