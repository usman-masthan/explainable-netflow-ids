"""
Tests for NetFlow telemetry generator.
"""

import pytest
import pandas as pd
from src.config import DataGenConfig, RAW_FLOW_COLUMNS
from src.data_generator import NetFlowDataGenerator


def test_generator_column_completeness():
    generator = NetFlowDataGenerator(DataGenConfig(n_samples=50, anomaly_ratio=0.2))
    df = generator.generate_dataset()

    assert isinstance(df, pd.DataFrame)
    assert len(df) == 50
    for col in RAW_FLOW_COLUMNS:
        assert col in df.columns


def test_generator_benign_traffic():
    generator = NetFlowDataGenerator(DataGenConfig(n_samples=100, anomaly_ratio=0.0))
    df = generator.generate_dataset()

    assert (df["is_anomaly"] == 0).all()
    assert (df["label"] == "benign").all()
    assert (df["byte_count"] > 0).all()
    assert (df["packet_count"] >= 1).all()
    assert (df["flow_duration"] >= 0.0).all()


def test_generator_attack_types():
    generator = NetFlowDataGenerator(DataGenConfig(n_samples=200, anomaly_ratio=0.5))
    df = generator.generate_dataset()

    anomalies = df[df["is_anomaly"] == 1]
    assert len(anomalies) == 100
    expected_attacks = {"dns_exfiltration", "c2_beacon", "port_scan", "syn_flood", "ssh_brute_force"}
    found_attacks = set(anomalies["label"].unique())
    assert expected_attacks == found_attacks


def test_generator_train_test_split():
    generator = NetFlowDataGenerator(DataGenConfig(random_state=42))
    train_df, test_df = generator.generate_train_test_split(n_train=200, n_test=100, test_anomaly_ratio=0.20)

    assert len(train_df) == 200
    assert (train_df["is_anomaly"] == 0).all()

    assert len(test_df) == 100
    assert test_df["is_anomaly"].sum() == 20

