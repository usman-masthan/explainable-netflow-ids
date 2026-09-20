"""
Isolation Forest Anomaly Detection Model for NetFlow Telemetry.
Trains on baseline enterprise traffic and outputs continuous anomaly scores
with operational threshold calibration for SecOps teams.
"""

import joblib
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Union, Dict, Any, Tuple
from sklearn.ensemble import IsolationForest
from src.config import ModelConfig


class NetFlowAnomalyDetector:
    """
    Unsupervised Anomaly Detection wrapper around scikit-learn's Isolation Forest.
    Provides calibrated operational scoring and thresholding.
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        self.config = config or ModelConfig()
        self.model = IsolationForest(
            n_estimators=self.config.n_estimators,
            max_samples=self.config.max_samples,
            contamination=self.config.contamination,
            random_state=self.config.random_state,
            n_jobs=self.config.n_jobs,
        )
        self.is_fitted: bool = False
        self.calibrated_threshold: float = 0.5
        self.score_min: float = 0.0
        self.score_max: float = 1.0

    def fit(self, X_train: pd.DataFrame) -> "NetFlowAnomalyDetector":
        """Fits the Isolation Forest on benign baseline feature vectors."""
        self.model.fit(X_train)
        self.is_fitted = True

        # Calculate score distribution on training baseline for normalization
        raw_train_scores = -self.model.score_samples(X_train)
        self.score_min = float(np.min(raw_train_scores))
        self.score_max = float(np.max(raw_train_scores))
        if self.score_max <= self.score_min:
            self.score_max = self.score_min + 1e-6

        # Set default threshold to 95th percentile of baseline
        self.calibrated_threshold = float(np.percentile(self.compute_anomaly_scores(X_train), 95.0))
        return self

    def compute_raw_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes raw anomaly scores where higher values indicate greater anomaly.
        (-score_samples(X) in scikit-learn).
        """
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before computing anomaly scores.")
        return -self.model.score_samples(X)

    def compute_anomaly_scores(self, X: pd.DataFrame) -> np.ndarray:
        """
        Returns normalized anomaly scores scaled to [0, 1], where 1.0 represents
        an extreme outlier requiring immediate Fusion SOC triage.
        """
        raw_scores = self.compute_raw_scores(X)
        normalized = (raw_scores - self.score_min) / (self.score_max - self.score_min)
        return np.clip(normalized, 0.0, 1.0)

    def calibrate_threshold_by_fpr(
        self, X_benign_val: pd.DataFrame, target_fpr: float = 0.02
    ) -> float:
        """
        Calibrates the decision threshold on a benign validation sample to strictly
        enforce a maximum operational False Positive Rate (FPR).
        E.g., target_fpr = 0.01 implies at most 1% of benign traffic will alert the SOC.
        """
        scores = self.compute_anomaly_scores(X_benign_val)
        percentile_cutoff = (1.0 - target_fpr) * 100.0
        self.calibrated_threshold = float(np.percentile(scores, percentile_cutoff))
        return self.calibrated_threshold

    def predict(self, X: pd.DataFrame, threshold: Optional[float] = None) -> np.ndarray:
        """
        Predicts binary anomaly status (1 = Anomaly / Alert, 0 = Benign).
        Uses calibrated operational threshold by default.
        """
        thresh = threshold if threshold is not None else self.calibrated_threshold
        scores = self.compute_anomaly_scores(X)
        return (scores >= thresh).astype(int)

    def save(self, file_path: Union[str, Path]) -> Path:
        """Serializes the fitted detector to disk."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "config": self.config,
                "is_fitted": self.is_fitted,
                "calibrated_threshold": self.calibrated_threshold,
                "score_min": self.score_min,
                "score_max": self.score_max,
            },
            path,
        )
        return path

    @classmethod
    def load(cls, file_path: Union[str, Path]) -> "NetFlowAnomalyDetector":
        """Deserializes a fitted detector from disk."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        data = joblib.load(path)
        detector = cls(data["config"])
        detector.model = data["model"]
        detector.is_fitted = data["is_fitted"]
        detector.calibrated_threshold = data["calibrated_threshold"]
        detector.score_min = data["score_min"]
        detector.score_max = data["score_max"]
        return detector

