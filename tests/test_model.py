"""
Tests for Isolation Forest NetFlow anomaly detector.
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.config import ModelConfig
from src.data_generator import NetFlowDataGenerator
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector


@pytest.fixture
def sample_features():
    gen = NetFlowDataGenerator(random_state=42)
    train_df, test_df = gen.generate_train_test_split(n_train=250, n_test=100, test_anomaly_ratio=0.2)
    preprocessor = NetFlowPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    return train_df, test_df, X_train, X_test


def test_model_fit_and_score(sample_features):
    train_df, test_df, X_train, X_test = sample_features

    detector = NetFlowAnomalyDetector(ModelConfig(n_estimators=50, random_state=42))
    assert not detector.is_fitted

    detector.fit(X_train)
    assert detector.is_fitted

    scores = detector.compute_anomaly_scores(X_test)
    assert len(scores) == 100
    assert (scores >= 0.0).all() and (scores <= 1.0).all()


def test_model_calibrate_threshold(sample_features):
    train_df, test_df, X_train, X_test = sample_features

    detector = NetFlowAnomalyDetector(ModelConfig(n_estimators=50, random_state=42))
    detector.fit(X_train)

    target_fpr = 0.04
    calibrated_th = detector.calibrate_threshold_by_fpr(X_train, target_fpr=target_fpr)
    assert 0.0 < calibrated_th < 1.0

    # Empirical false positive rate on X_train should be approximately target_fpr
    train_preds = detector.predict(X_train)
    empirical_fpr = train_preds.mean()
    assert abs(empirical_fpr - target_fpr) < 0.02


def test_model_save_and_load(tmp_path: Path, sample_features):
    train_df, test_df, X_train, X_test = sample_features

    detector = NetFlowAnomalyDetector(ModelConfig(n_estimators=50, random_state=42))
    detector.fit(X_train)
    detector.calibrate_threshold_by_fpr(X_train, target_fpr=0.03)

    model_file = tmp_path / "iso_forest.joblib"
    detector.save(model_file)
    assert model_file.exists()

    loaded_detector = NetFlowAnomalyDetector.load(model_file)
    assert loaded_detector.is_fitted
    assert loaded_detector.calibrated_threshold == detector.calibrated_threshold

    original_scores = detector.compute_anomaly_scores(X_test)
    loaded_scores = loaded_detector.compute_anomaly_scores(X_test)
    np.testing.assert_allclose(original_scores, loaded_scores)

