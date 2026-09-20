"""
Tests for NetFlow ingestion and validation.
"""

import pytest
import pandas as pd
from pathlib import Path
from src.ingestion import NetFlowIngestion, FlowDataValidationError
from src.data_generator import NetFlowDataGenerator


def test_schema_validation_success():
    gen = NetFlowDataGenerator()
    df = gen.generate_dataset(n_samples=20)
    # Should not raise
    NetFlowIngestion.validate_schema(df)


def test_schema_validation_failure():
    df = pd.DataFrame({"src_ip": ["10.0.0.1"], "dst_ip": ["10.0.0.2"]})
    with pytest.raises(FlowDataValidationError):
        NetFlowIngestion.validate_schema(df)


def test_validate_and_clean_filters_corrupt_data():
    gen = NetFlowDataGenerator()
    df = gen.generate_dataset(n_samples=20)

    # Inject corrupt rows
    corrupt_row = df.iloc[0].copy()
    corrupt_row["src_port"] = -99  # Invalid port
    corrupt_df = pd.concat([df, pd.DataFrame([corrupt_row])], ignore_index=True)

    cleaned = NetFlowIngestion.validate_and_clean(corrupt_df)
    assert len(cleaned) == 20
    assert (cleaned["src_port"] >= 0).all()


def test_csv_roundtrip(tmp_path: Path):
    gen = NetFlowDataGenerator()
    df = gen.generate_dataset(n_samples=30)
    file_path = tmp_path / "test_flows.csv"

    saved_path = NetFlowIngestion.save_to_csv(df, file_path)
    assert saved_path.exists()

    loaded = NetFlowIngestion.load_from_csv(saved_path)
    assert len(loaded) == 30
    assert loaded["flow_id"].tolist() == df["flow_id"].tolist()

