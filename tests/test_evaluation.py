"""
Tests for SecOps evaluation and trade-off analysis.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
from src.evaluation import SecOpsEvaluator


def test_evaluator_metrics():
    evaluator = SecOpsEvaluator(target_daily_flow_volume=10000)
    y_true = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
    scores = np.array([0.1, 0.2, 0.15, 0.3, 0.25, 0.8, 0.85, 0.9, 0.75, 0.95])
    labels = pd.Series(["benign"] * 5 + ["c2_beacon"] * 3 + ["port_scan"] * 2)

    metrics = evaluator.evaluate_detection(
        y_true=y_true,
        anomaly_scores=scores,
        threshold=0.5,
        attack_labels=labels,
    )

    assert metrics["auroc"] == 1.0
    assert metrics["pr_auc"] == 1.0
    assert metrics["recall"] == 1.0
    assert metrics["precision"] == 1.0
    assert metrics["false_positive_rate"] == 0.0
    assert metrics["confusion_matrix"]["true_positives"] == 5
    assert metrics["confusion_matrix"]["true_negatives"] == 5

    assert "c2_beacon" in metrics["per_attack_recall"]
    assert metrics["per_attack_recall"]["c2_beacon"]["recall"] == 1.0


def test_alert_tradeoff_curve():
    evaluator = SecOpsEvaluator(target_daily_flow_volume=100000)
    y_true = np.array([0] * 80 + [1] * 20)
    scores = np.linspace(0.1, 0.9, 100)

    tradeoff_df = evaluator.compute_alert_tradeoff_curve(y_true, scores, steps=10)
    assert isinstance(tradeoff_df, pd.DataFrame)
    assert len(tradeoff_df) == 10
    assert "threshold" in tradeoff_df.columns
    assert "recall" in tradeoff_df.columns
    assert "fpr" in tradeoff_df.columns
    assert "projected_daily_alerts" in tradeoff_df.columns


def test_plot_curves_save(tmp_path: Path):
    evaluator = SecOpsEvaluator()
    y_true = np.array([0] * 50 + [1] * 50)
    scores = np.random.uniform(0, 1, 100)

    roc_path = tmp_path / "roc.png"
    fig = evaluator.plot_roc_and_pr_curves(y_true, scores, save_path=roc_path)
    assert roc_path.exists()
    assert roc_path.stat().st_size > 1000

