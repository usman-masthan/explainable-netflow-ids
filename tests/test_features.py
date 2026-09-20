"""
Tests for NetFlow feature extraction and preprocessing pipeline.
"""

import pytest
import numpy as np
import pandas as pd
from src.data_generator import NetFlowDataGenerator
from src.features import NetFlowFeatureExtractor, NetFlowPreprocessor
from src.config import MODEL_FEATURE_COLUMNS, ENGINEERED_NUMERICAL_FEATURES


def test_feature_extraction_columns():
    gen = NetFlowDataGenerator()
    df = gen.generate_dataset(n_samples=25)

    feats = NetFlowFeatureExtractor.extract_raw_features(df)
    assert isinstance(feats, pd.DataFrame)
    assert list(feats.columns) == MODEL_FEATURE_COLUMNS
    assert len(feats) == 25
    # No NaN or Inf values
    assert not feats.isna().any().any()
    assert not np.isinf(feats.values).any()


def test_numerical_stability_zero_duration():
    # Edge case: 0 duration and 0 bytes
    edge_df = pd.DataFrame([{
        "flow_id": "FLW-TEST-1",
        "timestamp": "2026-09-20T08:00:00",
        "src_ip": "10.0.0.1",
        "dst_ip": "10.0.0.2",
        "src_port": 80,
        "dst_port": 80,
        "protocol": "TCP",
        "flow_duration": 0.0,
        "packet_count": 1,
        "byte_count": 0,
        "tcp_flags": "SYN",
        "tos": 0,
        "label": "benign",
        "is_anomaly": 0,
    }])

    feats = NetFlowFeatureExtractor.extract_raw_features(edge_df)
    assert not np.isnan(feats["bytes_per_packet"].iloc[0])
    assert not np.isinf(feats["packets_per_second"].iloc[0])
    assert feats["is_syn_only"].iloc[0] == 1
    assert feats["src_port_well_known"].iloc[0] == 1


def test_preprocessor_leak_free_fit_transform():
    gen = NetFlowDataGenerator()
    train_df, test_df = gen.generate_train_test_split(n_train=100, n_test=50)

    preprocessor = NetFlowPreprocessor(scaler_type="robust")
    assert not preprocessor.is_fitted

    X_train = preprocessor.fit_transform(train_df)
    assert preprocessor.is_fitted
    assert X_train.shape == (100, len(MODEL_FEATURE_COLUMNS))

    X_test = preprocessor.transform(test_df)
    assert X_test.shape == (50, len(MODEL_FEATURE_COLUMNS))
    assert list(X_test.columns) == MODEL_FEATURE_COLUMNS


def test_preprocessor_not_fitted_error():
    gen = NetFlowDataGenerator()
    df = gen.generate_dataset(n_samples=10)
    preprocessor = NetFlowPreprocessor()

    with pytest.raises(RuntimeError, match="must be fitted"):
        preprocessor.transform(df)

