"""
Unit tests for external benchmark dataset converters (CIDDS-001 and CIC-IDS2017).
"""

import pytest
import pandas as pd
from pathlib import Path
from src.config import RAW_FLOW_COLUMNS
from src.converters import CIDDS001Converter, CICIDS2017Converter, create_sample_cidds_csv
from src.ingestion import NetFlowIngestion
from src.features import NetFlowPreprocessor


def test_cidds_flag_parsing():
    assert CIDDS001Converter.parse_cidds_flags("....S.") == "SYN"
    assert CIDDS001Converter.parse_cidds_flags(".A....") == "ACK"
    assert "SYN" in CIDDS001Converter.parse_cidds_flags(".AP.SF")
    assert "ACK" in CIDDS001Converter.parse_cidds_flags(".AP.SF")
    assert "FIN" in CIDDS001Converter.parse_cidds_flags(".AP.SF")
    assert CIDDS001Converter.parse_cidds_flags("") == "NONE"


def test_cidds_byte_parsing():
    assert CIDDS001Converter.parse_cidds_bytes(1024) == 1024
    assert CIDDS001Converter.parse_cidds_bytes("1.2 M") == 1200000
    assert CIDDS001Converter.parse_cidds_bytes("450 K") == 450000
    assert CIDDS001Converter.parse_cidds_bytes("54") == 54


def test_cidds_conversion_schema():
    raw_cidds = pd.DataFrame([{
        "Date first seen": "2026-09-20 08:15:30.123",
        "Duration": 0.05,
        "Proto": "TCP",
        "Src IP Addr": "192.168.1.50",
        "Src Pt": 52140,
        "Dst IP Addr": "10.0.0.5",
        "Dst Pt": 443,
        "Packets": 10,
        "Bytes": "8.5 K",
        "Flags": ".AP.SF",
        "Tos": 0,
        "class": "normal",
        "attackType": "---",
    }])

    converted = CIDDS001Converter.convert_dataframe(raw_cidds)
    assert list(converted.columns) == RAW_FLOW_COLUMNS
    assert converted["protocol"].iloc[0] == "TCP"
    assert converted["byte_count"].iloc[0] == 8500
    assert converted["is_anomaly"].iloc[0] == 0
    assert "SYN" in converted["tcp_flags"].iloc[0]

    # Validate against NetFlowIngestion
    clean = NetFlowIngestion.validate_and_clean(converted)
    assert len(clean) == 1


def test_cicids_conversion_schema():
    raw_cic = pd.DataFrame([{
        "Timestamp": "2026-09-20 08:30:00",
        "Flow Duration": 50000,  # 50,000 microseconds = 0.05s
        "Source IP": "10.0.0.15",
        "Source Port": 55120,
        "Destination IP": "192.168.1.1",
        "Destination Port": 80,
        "Protocol": 6,  # TCP
        "Total Fwd Packets": 4,
        "Total Backward Packets": 2,
        "Total Length of Fwd Packets": 400,
        "Total Length of Bwd Packets": 600,
        "FIN Flag Count": 0,
        "SYN Flag Count": 1,
        "RST Flag Count": 0,
        "PSH Flag Count": 0,
        "ACK Flag Count": 1,
        "URG Flag Count": 0,
        "Label": "PortScan",
    }])

    converted = CICIDS2017Converter.convert_dataframe(raw_cic)
    assert list(converted.columns) == RAW_FLOW_COLUMNS
    assert converted["protocol"].iloc[0] == "TCP"
    assert converted["flow_duration"].iloc[0] == 0.05
    assert converted["packet_count"].iloc[0] == 6
    assert converted["byte_count"].iloc[0] == 1000
    assert converted["label"].iloc[0] == "port_scan"
    assert converted["is_anomaly"].iloc[0] == 1

    clean = NetFlowIngestion.validate_and_clean(converted)
    assert len(clean) == 1


def test_converted_data_preprocessor_pipeline(tmp_path: Path):
    sample_file = tmp_path / "cidds_sample.csv"
    create_sample_cidds_csv(sample_file, n_samples=30)
    raw_df = pd.read_csv(sample_file)

    converted = CIDDS001Converter.convert_dataframe(raw_df)
    clean = NetFlowIngestion.validate_and_clean(converted)

    preprocessor = NetFlowPreprocessor()
    X = preprocessor.fit_transform(clean)
    assert X.shape == (30, len(preprocessor.feature_names))
    assert not X.isna().any().any()
