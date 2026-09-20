"""
Integration tests for the complete explainable NetFlow IDS pipeline.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
import pandas as pd
from pathlib import Path
from src.config import DataGenConfig, ModelConfig
from src.data_generator import NetFlowDataGenerator
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector
from src.explainability import NetFlowExplainer
from src.evaluation import SecOpsEvaluator
from src.triage_enricher import FusionTriageEnricher


def test_full_pipeline_integration(tmp_path: Path):
    # 1. Generate data
    gen = NetFlowDataGenerator(random_state=42)
    train_df, test_df = gen.generate_train_test_split(n_train=300, n_test=100, test_anomaly_ratio=0.20)

    # 2. Preprocess
    preprocessor = NetFlowPreprocessor()
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)

    # 3. Fit Model & Calibrate Threshold
    detector = NetFlowAnomalyDetector(ModelConfig(n_estimators=30, random_state=42))
    detector.fit(X_train)
    calibrated_th = detector.calibrate_threshold_by_fpr(X_train, target_fpr=0.08)

    # 4. Predict
    scores = detector.compute_anomaly_scores(X_test)
    preds = detector.predict(X_test)
    assert len(scores) == 100
    assert len(preds) == 100

    # 5. Explain
    explainer = NetFlowExplainer(detector)
    explainer.fit_baseline_reference(X_train, train_df)

    anomalous_indices = [i for i, p in enumerate(preds) if p == 1]
    assert len(anomalous_indices) > 0, "Pipeline should detect anomalies"

    first_anomaly_idx = anomalous_indices[0]
    explanation = explainer.explain_flow(
        flow_idx=first_anomaly_idx,
        X=X_test,
        raw_flow=test_df.iloc[first_anomaly_idx],
        top_k=4,
    )
    assert explanation["anomaly_score"] >= calibrated_th
    assert len(explanation["top_drivers"]) > 0

    # 6. Evaluate
    evaluator = SecOpsEvaluator()
    metrics = evaluator.evaluate_detection(
        y_true=test_df["is_anomaly"].values,
        anomaly_scores=scores,
        threshold=calibrated_th,
        attack_labels=test_df["label"],
    )
    assert metrics["auroc"] > 0.85
    assert metrics["recall"] >= 0.50

    # 7. Enrich and Export Alerts
    enricher = FusionTriageEnricher()
    alert = enricher.create_alert(test_df.iloc[first_anomaly_idx], explanation)
    assert "network_traffic" in alert
    assert "soar_remediation" in alert

    alert_file = tmp_path / "sample_alert.json"
    enricher.export_alerts_to_json([alert], alert_file)
    assert alert_file.exists()
