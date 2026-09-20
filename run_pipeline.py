"""
End-to-End Orchestrator and CLI for Explainable NetFlow IDS.
Trains Isolation Forest, generates TreeSHAP explanations, evaluates SecOps metrics,
and exports Splunk CIM / SOAR alerts for Enterprise Security Operations.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import argparse
import json
from pathlib import Path
import pandas as pd
import numpy as np

from src.config import (
    DataGenConfig,
    ModelConfig,
    ExplainabilityConfig,
    EvalConfig,
    DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
)
from src.data_generator import NetFlowDataGenerator
from src.ingestion import NetFlowIngestion
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector
from src.explainability import NetFlowExplainer
from src.evaluation import SecOpsEvaluator
from src.triage_enricher import FusionTriageEnricher


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Explainable NetFlow Anomaly Detection and Triage Pipeline."
    )
    parser.add_argument("--n-train", type=int, default=8000, help="Number of baseline training flows.")
    parser.add_argument("--n-test", type=int, default=3000, help="Number of test evaluation flows.")
    parser.add_argument("--anomaly-ratio", type=float, default=0.15, help="Anomaly ratio in test set.")
    parser.add_argument("--target-fpr", type=float, default=0.03, help="Target False Positive Rate for calibration.")
    parser.add_argument("--n-estimators", type=int, default=150, help="Number of isolation trees.")
    parser.add_argument("--seed", type=int, default=42, help="Random state seed.")
    parser.add_argument("--output-dir", type=str, default=str(REPORTS_DIR), help="Output directory.")
    return parser.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.output_dir)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 78)
    print(" EXPLAINABLE NETFLOW INTRUSION DETECTION SYSTEM (X-NetFlow-IDS)")
    print(" Enterprise Security Operations & Cyber Fusion Pipeline")
    print("=" * 78)

    # 1. Telemetry Generation & Ingestion
    print(f"\n[1/7] Ingesting NetFlow v9 Telemetry...")
    gen = NetFlowDataGenerator(random_state=args.seed)
    train_df, test_df = gen.generate_train_test_split(
        n_train=args.n_train, n_test=args.n_test, test_anomaly_ratio=args.anomaly_ratio
    )
    NetFlowIngestion.save_to_csv(train_df, DATA_DIR / "train_baseline.csv")
    NetFlowIngestion.save_to_csv(test_df, DATA_DIR / "test_flows.csv")
    print(f"      - Baseline train flows (100% benign): {len(train_df):,}")
    print(f"      - Evaluation test flows ({args.anomaly_ratio:.0%} attacks): {len(test_df):,}")

    # 2. Leak-Free Feature Engineering
    print(f"\n[2/7] Extracting Domain Behavioral Network Features...")
    preprocessor = NetFlowPreprocessor(scaler_type="robust")
    X_train = preprocessor.fit_transform(train_df)
    X_test = preprocessor.transform(test_df)
    print(f"      - Feature dimension: {X_train.shape[1]} engineered features")
    print(f"      - Preprocessor fitted strictly on benign training split (no leakage).")

    # 3. Model Training & SecOps Threshold Calibration
    print(f"\n[3/7] Training Isolation Forest Baseline & Calibrating Threshold...")
    model_cfg = ModelConfig(
        n_estimators=args.n_estimators,
        random_state=args.seed,
    )
    detector = NetFlowAnomalyDetector(model_cfg)
    detector.fit(X_train)
    calibrated_th = detector.calibrate_threshold_by_fpr(X_train, target_fpr=args.target_fpr)
    model_path = out_dir / "model_checkpoint.joblib"
    detector.save(model_path)
    print(f"      - Isolation Forest fitted with {args.n_estimators} estimators.")
    print(f"      - Calibrated Threshold (target FPR <= {args.target_fpr:.1%}): {calibrated_th:.4f}")
    print(f"      - Checkpoint saved to: {model_path}")

    # 4. Anomaly Scoring & Predictions
    print(f"\n[4/7] Computing Anomaly Scores & Evaluating Flows...")
    scores = detector.compute_anomaly_scores(X_test)
    preds = detector.predict(X_test)
    n_flagged = int(preds.sum())
    print(f"      - Test flows evaluated: {len(scores):,}")
    print(f"      - Flows flagged as suspicious by detector: {n_flagged:,} ({n_flagged/len(scores):.1%})")

    # 5. TreeSHAP Explainability Layer
    print(f"\n[5/7] Generating TreeSHAP Explanations & Feature Attributions...")
    explainer = NetFlowExplainer(detector)
    explainer.fit_baseline_reference(X_train, train_df)

    imp_path = fig_dir / "global_feature_importance.png"
    explainer.plot_feature_importance(X_test.iloc[:500], top_n=10, save_path=imp_path)
    print(f"      - Global feature importance chart saved: {imp_path}")

    # Explain top anomaly
    sorted_anomaly_idxs = np.argsort(-scores)
    top_anomaly_idx = int(sorted_anomaly_idxs[0])
    top_explanation = explainer.explain_flow(
        flow_idx=top_anomaly_idx,
        X=X_test,
        raw_flow=test_df.iloc[top_anomaly_idx],
        top_k=4,
    )
    waterfall_path = fig_dir / "shap_waterfall_top_anomaly.png"
    explainer.plot_waterfall(top_anomaly_idx, X_test, save_path=waterfall_path)
    print(f"      - Local SHAP waterfall saved: {waterfall_path}")
    print(f"\n      [Top Flagged Flow Explanation Sample]:")
    print(f"      {top_explanation['narrative']}")

    # 6. SecOps Evaluation & Trade-off Analysis
    print(f"\n[6/7] Computing SecOps Performance & False Positive Impact...")
    evaluator = SecOpsEvaluator(target_daily_flow_volume=1_000_000)
    metrics = evaluator.evaluate_detection(
        y_true=test_df["is_anomaly"].values,
        anomaly_scores=scores,
        threshold=calibrated_th,
        attack_labels=test_df["label"],
    )

    roc_pr_path = fig_dir / "roc_pr_curves.png"
    evaluator.plot_roc_and_pr_curves(test_df["is_anomaly"].values, scores, save_path=roc_pr_path)

    tradeoff_df = evaluator.compute_alert_tradeoff_curve(test_df["is_anomaly"].values, scores)
    tradeoff_path = fig_dir / "alert_volume_tradeoff.png"
    evaluator.plot_alert_tradeoff(tradeoff_df, save_path=tradeoff_path)

    cm_path = fig_dir / "confusion_matrix.png"
    evaluator.plot_confusion_matrix(metrics["confusion_matrix"], save_path=cm_path)

    summary_file = out_dir / "evaluation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"      - Saved evaluation summary to: {summary_file}")
    print(f"      - AUROC: {metrics['auroc']:.4f} | PR-AUC: {metrics['pr_auc']:.4f}")
    print(f"      - Precision: {metrics['precision']:.4f} | Recall: {metrics['recall']:.4f} | F1: {metrics['f1_score']:.4f}")
    print(f"      - Operational FPR: {metrics['false_positive_rate']:.4%}")

    # 7. Fusion Triage Enrichment & SOAR Alert Export
    print(f"\n[7/7] Generating Splunk CIM & SOAR Enriched Alerts...")
    enricher = FusionTriageEnricher()
    
    # Generate explanations for flagged alerts (up to 50 for sample report)
    flagged_indices = [i for i, p in enumerate(preds) if p == 1][:50]
    sample_explanations = [
        explainer.explain_flow(idx, X_test, raw_flow=test_df.iloc[idx], top_k=4)
        for idx in flagged_indices
    ]
    alerts = [
        enricher.create_alert(test_df.iloc[flagged_indices[i]], sample_explanations[i])
        for i in range(len(flagged_indices))
    ]
    alerts_file = out_dir / "sample_alerts.json"
    enricher.export_alerts_to_json(alerts, alerts_file)
    print(f"      - Exported {len(alerts)} enriched Splunk CIM alerts to: {alerts_file}")

    print("\n" + "=" * 78)
    print(" PIPELINE EXECUTION COMPLETE - ARTIFACTS READY")
    print("=" * 78)
    print(f" • Model:        {model_path}")
    print(f" • Metrics:      {summary_file}")
    print(f" • Alerts JSON:  {alerts_file}")
    print(f" • Figures:      {fig_dir}")
    print("=" * 78)


if __name__ == "__main__":
    main()

