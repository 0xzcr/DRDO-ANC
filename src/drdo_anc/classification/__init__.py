"""Noise classification analysis primitives."""

from .categories import (
    DEFENCE_NOISE_CATEGORIES,
    NOISE_CLASSES,
    UNKNOWN_CLASS,
)
from .classifier import ClassificationResult, NoiseClassifier
from .features import (
    DEFAULT_HOP_MS,
    DEFAULT_SAMPLE_RATE,
    DEFAULT_WINDOW_MS,
    AggregatedFeatures,
    extract_features,
    feature_vector,
)

__all__ = [
    "AggregatedFeatures",
    "ClassificationResult",
    "DEFENCE_NOISE_CATEGORIES",
    "DEFAULT_HOP_MS",
    "DEFAULT_SAMPLE_RATE",
    "DEFAULT_WINDOW_MS",
    "NOISE_CLASSES",
    "NoiseClassifier",
    "UNKNOWN_CLASS",
    "extract_features",
    "feature_vector",
]
