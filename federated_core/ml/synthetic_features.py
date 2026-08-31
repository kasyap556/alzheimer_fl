# =============================================================================
# federated_core/ml/synthetic_features.py
# ----------------------------------------
# Generates synthetic clinical feature vectors conditioned on Alzheimer's
# diagnosis class labels. Distributions are informed by the EDA notebook
# (alzheimer-features-prediction.ipynb) and medical literature.
#
# This module enables the multimodal architecture to function with an
# image-only dataset by providing realistic tabular features for the
# MLP branch and SHAP explainability.
#
# Author: Kasyap (Final-Year Academic Project)
# =============================================================================

from __future__ import annotations

import numpy as np
from typing import List, Tuple, Dict


# ─────────────────────────────────────────────────────────────────────────────
# Feature schema (9 clinical features)
# ─────────────────────────────────────────────────────────────────────────────
FEATURE_NAMES: List[str] = [
    "MMSE",     # Mini-Mental State Examination (0–30)
    "CDR",      # Clinical Dementia Rating (0–3)
    "Age",      # Patient age in years
    "EDUC",     # Years of education
    "nWBV",     # Normalized Whole-Brain Volume
    "eTIV",     # Estimated Total Intracranial Volume (cm³)
    "ASF",      # Atlas Scaling Factor
    "SES",      # Socioeconomic Status (1–5)
    "Gender",   # 0 = Female, 1 = Male
]

NUM_FEATURES: int = len(FEATURE_NAMES)  # 9


# ─────────────────────────────────────────────────────────────────────────────
# Class-conditional Gaussian parameters
# ─────────────────────────────────────────────────────────────────────────────
# Class indices match torchvision.datasets.ImageFolder alphabetical order:
#   0 = MildDemented, 1 = ModerateDemented, 2 = NonDemented, 3 = VeryMildDemented
#
# Each entry: (mean, std, min_clip, max_clip)
# Distributions are derived from the notebook's EDA (boxplots, histograms)
# and established Alzheimer's clinical correlations.

CLASS_DISTRIBUTIONS: Dict[int, Dict[str, Tuple[float, float, float, float]]] = {
    # ── Class 0: Mild Dementia ────────────────────────────────────────────
    0: {
        "MMSE":   (20.0,  2.0,   10.0, 30.0),
        "CDR":    (1.0,   0.15,   0.5,  2.0),
        "Age":    (76.0,  6.0,   60.0, 95.0),
        "EDUC":   (13.0,  3.0,    4.0, 23.0),
        "nWBV":   (0.70,  0.03,   0.60, 0.85),
        "eTIV":   (1450.0, 125.0, 1100.0, 2000.0),
        "ASF":    (1.24,  0.13,   0.80, 1.60),
        "SES":    (3.0,   1.0,    1.0,  5.0),
        "Gender": (0.55,  0.50,   0.0,  1.0),   # Bernoulli approximated as Gaussian, then rounded
    },
    # ── Class 1: Moderate Dementia ────────────────────────────────────────
    1: {
        "MMSE":   (14.0,  3.0,    0.0, 30.0),
        "CDR":    (2.0,   0.3,    1.0,  3.0),
        "Age":    (79.0,  5.0,   60.0, 95.0),
        "EDUC":   (12.0,  3.0,    4.0, 23.0),
        "nWBV":   (0.66,  0.03,   0.60, 0.85),
        "eTIV":   (1420.0, 130.0, 1100.0, 2000.0),
        "ASF":    (1.26,  0.14,   0.80, 1.60),
        "SES":    (3.2,   1.0,    1.0,  5.0),
        "Gender": (0.60,  0.49,   0.0,  1.0),
    },
    # ── Class 2: Non Demented ─────────────────────────────────────────────
    2: {
        "MMSE":   (29.0,  1.0,   25.0, 30.0),
        "CDR":    (0.0,   0.05,   0.0,  0.5),
        "Age":    (70.0,  6.0,   60.0, 95.0),
        "EDUC":   (16.0,  3.0,    4.0, 23.0),
        "nWBV":   (0.78,  0.03,   0.60, 0.85),
        "eTIV":   (1500.0, 120.0, 1100.0, 2000.0),
        "ASF":    (1.20,  0.12,   0.80, 1.60),
        "SES":    (2.5,   1.0,    1.0,  5.0),
        "Gender": (0.50,  0.50,   0.0,  1.0),
    },
    # ── Class 3: Very Mild Dementia ───────────────────────────────────────
    3: {
        "MMSE":   (25.5,  1.5,   20.0, 30.0),
        "CDR":    (0.5,   0.1,    0.0,  1.0),
        "Age":    (74.0,  7.0,   60.0, 95.0),
        "EDUC":   (14.0,  3.0,    4.0, 23.0),
        "nWBV":   (0.74,  0.03,   0.60, 0.85),
        "eTIV":   (1480.0, 130.0, 1100.0, 2000.0),
        "ASF":    (1.22,  0.13,   0.80, 1.60),
        "SES":    (2.8,   1.0,    1.0,  5.0),
        "Gender": (0.50,  0.50,   0.0,  1.0),
    },
}

# ─────────────────────────────────────────────────────────────────────────────
# Precomputed global normalization statistics (mean/std across all classes)
# Used for z-score normalization:  x' = (x - μ) / σ
# ─────────────────────────────────────────────────────────────────────────────
_GLOBAL_MEAN = np.array([22.125, 0.875, 74.75, 13.75, 0.72, 1462.5, 1.23, 2.875, 0.5375], dtype=np.float32)
_GLOBAL_STD  = np.array([6.0,   0.75,  6.5,   3.0,  0.05, 130.0,  0.13, 1.0,   0.50],   dtype=np.float32)


class SyntheticClinicalGenerator:
    """
    Generates synthetic clinical feature vectors conditioned on class labels.

    Each feature is sampled from a class-conditional Gaussian distribution
    and clipped to medically valid ranges:

        x_i ~ N(μ_{c,i}, σ_{c,i}²)  clipped to [min_i, max_i]

    where c is the class label and i indexes the feature.

    The Gender feature is special-cased: the Gaussian sample is rounded
    to 0 or 1 to produce a binary indicator.
    """

    def __init__(self, seed: int | None = None) -> None:
        """
        Args:
            seed: Optional random seed for reproducibility.
        """
        self.rng = np.random.RandomState(seed)
        self.feature_names: List[str] = FEATURE_NAMES
        self.num_features: int = NUM_FEATURES

    def generate(self, class_label: int) -> np.ndarray:
        """
        Generate a single synthetic clinical feature vector.

        Args:
            class_label: Integer class label (0–3).

        Returns:
            np.ndarray of shape (9,) with raw (unnormalized) feature values.
        """
        if class_label not in CLASS_DISTRIBUTIONS:
            raise ValueError(f"Unknown class label: {class_label}. Expected 0–3.")

        dist = CLASS_DISTRIBUTIONS[class_label]
        features = np.zeros(self.num_features, dtype=np.float32)

        for i, name in enumerate(self.feature_names):
            mean, std, clip_min, clip_max = dist[name]
            value = self.rng.normal(mean, std)
            value = np.clip(value, clip_min, clip_max)

            # Gender is binary — round to 0 or 1
            if name == "Gender":
                value = float(round(value))

            # SES is integer-valued (1–5)
            if name == "SES":
                value = float(round(value))

            features[i] = value

        return features

    def generate_normalized(self, class_label: int) -> np.ndarray:
        """
        Generate a z-score normalized feature vector.

        Args:
            class_label: Integer class label (0–3).

        Returns:
            np.ndarray of shape (9,) with z-score normalized values.
        """
        raw = self.generate(class_label)
        return (raw - _GLOBAL_MEAN) / (_GLOBAL_STD + 1e-8)

    def generate_batch(
        self, class_labels: np.ndarray, normalize: bool = True
    ) -> np.ndarray:
        """
        Generate a batch of feature vectors.

        Args:
            class_labels: Array of integer class labels, shape (N,).
            normalize:    If True, apply z-score normalization.

        Returns:
            np.ndarray of shape (N, 9).
        """
        batch = np.stack([
            self.generate_normalized(int(c)) if normalize
            else self.generate(int(c))
            for c in class_labels
        ])
        return batch

    def get_background_samples(
        self, n: int = 50, normalize: bool = True
    ) -> np.ndarray:
        """
        Generate a mixed-class background dataset for SHAP KernelExplainer.

        Produces roughly equal samples per class.

        Args:
            n:         Total number of background samples.
            normalize: If True, z-score normalize.

        Returns:
            np.ndarray of shape (n, 9).
        """
        num_classes = len(CLASS_DISTRIBUTIONS)
        samples_per_class = n // num_classes
        remainder = n % num_classes

        all_samples = []
        for c in range(num_classes):
            count = samples_per_class + (1 if c < remainder else 0)
            for _ in range(count):
                if normalize:
                    all_samples.append(self.generate_normalized(c))
                else:
                    all_samples.append(self.generate(c))

        return np.stack(all_samples, axis=0)

    @staticmethod
    def get_feature_names() -> List[str]:
        """Return the ordered list of feature names."""
        return FEATURE_NAMES.copy()

    @staticmethod
    def get_normalization_stats() -> Tuple[np.ndarray, np.ndarray]:
        """
        Return (global_mean, global_std) arrays for z-score normalization.

        Returns:
            Tuple of (mean, std), each np.ndarray of shape (9,).
        """
        return _GLOBAL_MEAN.copy(), _GLOBAL_STD.copy()

    @staticmethod
    def normalize(raw_features: np.ndarray) -> np.ndarray:
        """
        Apply z-score normalization to raw feature vectors.

        Args:
            raw_features: Shape (..., 9) array of raw values.

        Returns:
            Normalized array of the same shape.
        """
        return (raw_features - _GLOBAL_MEAN) / (_GLOBAL_STD + 1e-8)
