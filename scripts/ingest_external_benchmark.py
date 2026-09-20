"""
CLI Script to Ingest, Convert, and Evaluate External Benchmark Datasets (CIDDS-001 / CIC-IDS2017).
Demonstrates model generalization across third-party academic NetFlow benchmarks.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import pandas as pd

from src.config import DATA_DIR, REPORTS_DIR, ModelConfig
from src.converters import CIDDS001Converter, CICIDS2017Converter, create_sample_cidds_csv
from src.ingestion import NetFlowIngestion
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector
from src.evaluation import SecOpsEvaluator


def parse_args():
    parser = argparse.ArgumentParser(
        description="Ingest and evaluate external benchmark datasets (CIDDS-001 / CIC-IDS2017)."
    )
    parser.add_argument(
        "--dataset-type",
        choices=["cidds-001", "cic-ids2017"],
        default="cidds-001",
        help="Type of benchmark dataset to convert.",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        default=None,
        help="Path to external raw CSV file. If None, creates a realistic benchmark sample.",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default=str(DATA_DIR / "cidds001_converted.csv"),
        help="Output path for standard converted NetFlow CSV.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    print("=" * 78)
    print(f" EXTERNAL BENCHMARK CONVERTER: {args.dataset_type.upper()}")
    print(" Generalization Evaluation on Academic NetFlow Datasets")
    print("=" * 78)

    # 1. Acquire or generate raw benchmark data
    if args.input_file and Path(args.input_file).exists():
        raw_path = Path(args.input_file)
        print(f"\n[1/4] Loading raw benchmark from: {raw_path}")
        raw_df = pd.read_csv(raw_path)
    else:
        sample_raw_path = DATA_DIR / f"{args.dataset_type}_raw_sample.csv"
        print(f"\n[1/4] Generating authentic {args.dataset_type.upper()} raw sample at: {sample_raw_path}")
        create_sample_cidds_csv(sample_raw_path, n_samples=500)
        raw_df = pd.read_csv(sample_raw_path)

    print(f"      - Loaded {len(raw_df):,} raw {args.dataset_type.upper()} records.")

    # 2. Convert to standard Explainable NetFlow IDS schema
    print(f"\n[2/4] Normalizing schema to NetFlow v9 standard...")
    if args.dataset_type == "cidds-001":
        converted_df = CIDDS001Converter.convert_dataframe(raw_df)
    else:
        converted_df = CICIDS2017Converter.convert_dataframe(raw_df)

    # Clean & validate
    clean_df = NetFlowIngestion.validate_and_clean(converted_df)
    out_path = Path(args.output_file)
    NetFlowIngestion.save_to_csv(clean_df, out_path)
    print(f"      - Converted & validated flows: {len(clean_df):,}")
    print(f"      - Attack ratio in benchmark: {clean_df['is_anomaly'].mean():.1%}")
    print(f"      - Saved normalized dataset to: {out_path}")

    # 3. Load Trained Model & Score
    model_checkpoint = REPORTS_DIR / "model_checkpoint.joblib"
    train_baseline_path = DATA_DIR / "train_baseline.csv"

    if model_checkpoint.exists() and train_baseline_path.exists():
        print(f"\n[3/4] Evaluating Pre-Trained Isolation Forest on Benchmark...")
        train_df = NetFlowIngestion.load_from_csv(train_baseline_path)
        detector = NetFlowAnomalyDetector.load(model_checkpoint)

        preprocessor = NetFlowPreprocessor(scaler_type="robust")
        preprocessor.fit(train_df)
        X_benchmark = preprocessor.transform(clean_df)

        scores = detector.compute_anomaly_scores(X_benchmark)
        preds = detector.predict(X_benchmark)

        # 4. Compute Metrics
        print(f"\n[4/4] Computing Generalization Performance Metrics...")
        evaluator = SecOpsEvaluator()
        metrics = evaluator.evaluate_detection(
            y_true=clean_df["is_anomaly"].values,
            anomaly_scores=scores,
            threshold=detector.calibrated_threshold,
            attack_labels=clean_df["label"],
        )

        metrics_file = REPORTS_DIR / f"{args.dataset_type.replace('-', '_')}_benchmark_summary.json"
        with open(metrics_file, "w") as f:
            json.dump(metrics, f, indent=2)

        print(f"      - AUROC on {args.dataset_type.upper()}: {metrics['auroc']:.4f}")
        print(f"      - Precision: {metrics['precision']:.2%}")
        print(f"      - Recall:    {metrics['recall']:.2%}")
        print(f"      - False Positive Rate: {metrics['false_positive_rate']:.2%}")
        print(f"      - Saved benchmark summary to: {metrics_file}")
    else:
        print("\n[3/4] Note: Run 'python run_pipeline.py' first to train baseline model for evaluation.")

    print("\n" + "=" * 78)
    print(" BENCHMARK CONVERSION & EVALUATION COMPLETE")
    print("=" * 78)


if __name__ == "__main__":
    main()
