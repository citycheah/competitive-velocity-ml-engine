"""Text processing and dimensionality reduction pipeline for video titles and descriptions.

Fits strictly on training folds to prevent out-of-fold vocabulary or singular vector leakage.
"""

from __future__ import annotations

import logging
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger("text_pipeline")


class TextEmbeddingPipeline(BaseEstimator, TransformerMixin):
    """Compresses raw video titles into low-dimensional dense vectors using TF-IDF + TruncatedSVD.
    
    Adheres strictly to scikit-learn transformer protocol. Fitting strictly on the training
    split ensures that vocabulary distributions and singular value coordinates do not leak
    into validation folds or out-of-time test sets.
    """

    def __init__(
        self,
        n_components: int = 16,
        max_features: int = 500,
        ngram_range: tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        self.n_components = n_components
        self.max_features = max_features
        self.ngram_range = ngram_range
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            stop_words="english",
            lowercase=True,
            strip_accents="unicode",
        )
        self.svd = TruncatedSVD(
            n_components=self.n_components,
            random_state=self.random_state,
        )
        self.is_fitted = False

    def _sanitize_text(self, text_series: Union[pd.Series, List[str], np.ndarray]) -> List[str]:
        """Sanitizes text inputs, replacing NaNs and non-strings with empty strings."""
        if isinstance(text_series, pd.Series):
            return text_series.fillna("").astype(str).tolist()
        return [str(t) if t is not None and not (isinstance(t, float) and np.isnan(t)) else "" for t in text_series]

    def fit(self, X: Union[pd.Series, List[str], np.ndarray], y: Optional[np.ndarray] = None) -> TextEmbeddingPipeline:
        """Fits TF-IDF vocabulary and TruncatedSVD components strictly on the provided training texts."""
        clean_texts = self._sanitize_text(X)
        tfidf_matrix = self.vectorizer.fit_transform(clean_texts)

        # Guard against vocabularies smaller than n_components
        actual_features = tfidf_matrix.shape[1]
        if actual_features < self.n_components:
            logger.warning(
                "TF-IDF vocabulary size (%d) is smaller than n_components (%d). Adjusting SVD components.",
                actual_features,
                self.n_components,
            )
            self.svd.n_components = max(1, actual_features - 1)

        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
        return self

    def transform(self, X: Union[pd.Series, List[str], np.ndarray]) -> np.ndarray:
        """Transforms input text into dense SVD component coordinates."""
        if not self.is_fitted:
            raise RuntimeError("TextEmbeddingPipeline must be fitted before calling transform().")

        clean_texts = self._sanitize_text(X)
        tfidf_matrix = self.vectorizer.transform(clean_texts)
        embeddings = self.svd.transform(tfidf_matrix)

        # Pad with zeros if n_components was adjusted downwards due to tiny corpus
        if embeddings.shape[1] < self.n_components:
            pad_width = self.n_components - embeddings.shape[1]
            embeddings = np.pad(embeddings, ((0, 0), (0, pad_width)), mode="constant")

        return embeddings

    def fit_transform(
        self, X: Union[pd.Series, List[str], np.ndarray], y: Optional[np.ndarray] = None
    ) -> np.ndarray:
        """Fits on X and returns transformed SVD embeddings."""
        return self.fit(X, y).transform(X)

    def get_feature_names_out(self, input_features: Optional[List[str]] = None) -> List[str]:
        """Returns feature names for the dense SVD components."""
        return [f"text_svd_dim_{i:02d}" for i in range(self.n_components)]
