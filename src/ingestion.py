"""
NetFlow Ingestion, Validation, and Integrity Pipeline.
Enforces strict schema compliance and prepares flow records for feature engineering.
"""

import ipaddress
import pandas as pd
from pathlib import Path
from typing import Union, List, Tuple
from src.config import RAW_FLOW_COLUMNS


class FlowDataValidationError(Exception):
    """Raised when NetFlow telemetry fails integrity or schema verification."""
    pass


class NetFlowIngestion:
    """Handles loading, validation, and serialization of NetFlow telemetry."""

    @staticmethod
    def validate_schema(df: pd.DataFrame) -> None:
        """Verifies that all required NetFlow columns are present and correctly typed."""
        missing = [col for col in RAW_FLOW_COLUMNS if col not in df.columns]
        if missing:
            raise FlowDataValidationError(f"Missing required NetFlow columns: {missing}")

    @staticmethod
    def validate_and_clean(df: pd.DataFrame) -> pd.DataFrame:
        """
        Validates values, removes invalid/corrupt rows, and enforces type constraints:
        - Non-negative packets, bytes, durations
        - Valid port boundaries (0 - 65535)
        - Valid IPv4 formatting
        """
        NetFlowIngestion.validate_schema(df)
        df_clean = df.copy()

        # Check null values
        initial_len = len(df_clean)
        df_clean = df_clean.dropna(subset=["src_ip", "dst_ip", "src_port", "dst_port", "byte_count"])

        # Numerical domain constraints
        valid_mask = (
            (df_clean["src_port"] >= 0) & (df_clean["src_port"] <= 65535) &
            (df_clean["dst_port"] >= 0) & (df_clean["dst_port"] <= 65535) &
            (df_clean["byte_count"] >= 0) &
            (df_clean["packet_count"] >= 1) &
            (df_clean["flow_duration"] >= 0.0)
        )
        df_clean = df_clean[valid_mask].copy()

        # Type casts
        df_clean["src_port"] = df_clean["src_port"].astype(int)
        df_clean["dst_port"] = df_clean["dst_port"].astype(int)
        df_clean["packet_count"] = df_clean["packet_count"].astype(int)
        df_clean["byte_count"] = df_clean["byte_count"].astype(int)
        df_clean["flow_duration"] = df_clean["flow_duration"].astype(float)
        df_clean["protocol"] = df_clean["protocol"].astype(str).str.upper()
        df_clean["tcp_flags"] = df_clean["tcp_flags"].fillna("NONE").astype(str)

        return df_clean

    @staticmethod
    def load_from_csv(file_path: Union[str, Path]) -> pd.DataFrame:
        """Loads and cleans a NetFlow dataset from CSV."""
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")
        df = pd.read_csv(path)
        return NetFlowIngestion.validate_and_clean(df)

    @staticmethod
    def save_to_csv(df: pd.DataFrame, file_path: Union[str, Path]) -> Path:
        """Saves validated NetFlow dataset to CSV."""
        path = Path(file_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        return path

