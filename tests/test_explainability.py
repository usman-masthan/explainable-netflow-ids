"""
Tests for TreeSHAP explainability engine.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
import pandas as pd
from pathlib import Path
from src.config import ModelConfig
from src.data_generator import NetFlowDataGenerator
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector
from src.explainability import NetFlowExplainer


@pytest.fixture
def trained_pipeline():
    gen = NetFlowDataGenerator(random_state=42)
    train_df, test_df = gen.generate_train_test_split(n_train=200, n_test=50, test_anomaly_ratio=0.2)
    preprocessor = NetFlowPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    detector = NetFlowAnomalyDetector(ModelConfig(n_estimators=30, random_state=42))
    detector.fit(X_train)
    detector.calibrate_threshold_by_fpr(X_train, target_fpr=0.05)

    return train_df, test_df, X_train, X_test, detector


def test_explainer_computations(trained_pipeline):
    train_df, test_df, X_train, X_test, detector = trained_pipeline

    explainer = NetFlowExplainer(detector)
    explainer.fit_baseline_reference(X_train, train_df)

    # Compute SHAP values
    anomaly_shap = explainer.compute_anomaly_shap(X_test.iloc[:10])
    assert anomaly_shap.shape == (10, X_test.shape[1])

    # Global importance
    imp_df = explainer.get_feature_importance(X_test.iloc[:20])
    assert len(imp_df) == X_test.shape[1]
    assert "mean_abs_shap" in imp_df.columns
    assert (imp_df["mean_abs_shap"] >= 0).all()


def test_explain_single_flow(trained_pipeline):
    train_df, test_df, X_train, X_test, detector = trained_pipeline

    explainer = NetFlowExplainer(detector)
    explainer.fit_baseline_reference(X_train, train_df)

    explanation = explainer.explain_flow(
        flow_idx=0,
        X=X_test,
        raw_flow=test_df.iloc[0],
        top_k=3,
    )

    assert "flow_index" in explanation
    assert "anomaly_score" in explanation
    assert "top_drivers" in explanation
    assert len(explanation["top_drivers"]) <= 3
    assert "narrative" in explanation
    assert isinstance(explanation["narrative"], str)
    assert len(explanation["narrative"]) > 20


def test_plot_feature_importance_save(tmp_path: Path, trained_pipeline):
    train_df, test_df, X_train, X_test, detector = trained_pipeline

    explainer = NetFlowExplainer(detector)
    plot_file = tmp_path / "test_importance.png"
    fig = explainer.plot_feature_importance(X_test.iloc[:20], top_n=5, save_path=plot_file)

    assert plot_file.exists()
    assert plot_file.stat().st_size > 1000

