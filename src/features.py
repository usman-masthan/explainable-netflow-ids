"""
Feature Engineering and Preprocessing Pipeline for NetFlow Anomaly Detection.
Enforces strict leak-free transformation ordering and extracts behavioral security metrics.
"""

import numpy as np
import pandas as pd
from typing import Tuple, List, Optional
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import RobustScaler, StandardScaler
from src.config import (
    RAW_FLOW_COLUMNS,
    ENGINEERED_NUMERICAL_FEATURES,
    ENGINEERED_FLAG_FEATURES,
    MODEL_FEATURE_COLUMNS,
)


class NetFlowFeatureExtractor:
    """
    Computes domain-informed behavioral network telemetry features:
    - Volumetric rates (bytes/packet, packets/sec, bytes/sec)
    - Logarithmic normalizations for heavy-tailed distributions
    - Port classification (well-known, registered, dynamic)
    - TCP flag breakdown and SYN-flood / scan indicators
    """

    @staticmethod
    def extract_raw_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derives unscaled feature columns from raw NetFlow records."""
        feat_df = pd.DataFrame(index=df.index)

        # 1. Ratios and rates (with numerical stability against zero-duration/packets)
        packets = np.maximum(df["packet_count"].values.astype(float), 1.0)
        bytes_cnt = np.maximum(df["byte_count"].values.astype(float), 0.0)
        duration = np.maximum(df["flow_duration"].values.astype(float), 0.0001)

        feat_df["bytes_per_packet"] = bytes_cnt / packets
        feat_df["packets_per_second"] = packets / duration
        feat_df["bytes_per_second"] = bytes_cnt / duration

        # 2. Log1p transforms for long-tailed volume variables
        feat_df["log_bytes"] = np.log1p(bytes_cnt)
        feat_df["log_packets"] = np.log1p(packets)
        feat_df["log_duration"] = np.log1p(duration)
        feat_df["log_bps"] = np.log1p(feat_df["bytes_per_second"])
        feat_df["log_pps"] = np.log1p(feat_df["packets_per_second"])

        # 3. Flag breakdown
        flags = df["tcp_flags"].fillna("NONE").astype(str)
        feat_df["flag_SYN"] = flags.str.contains("SYN", case=False).astype(int)
        feat_df["flag_ACK"] = flags.str.contains("ACK", case=False).astype(int)
        feat_df["flag_FIN"] = flags.str.contains("FIN", case=False).astype(int)
        feat_df["flag_RST"] = flags.str.contains("RST", case=False).astype(int)
        feat_df["flag_PSH"] = flags.str.contains("PSH", case=False).astype(int)
        feat_df["flag_URG"] = flags.str.contains("URG", case=False).astype(int)
        feat_df["is_syn_only"] = (
            (feat_df["flag_SYN"] == 1) & (feat_df["flag_ACK"] == 0) & (feat_df["flag_FIN"] == 0)
        ).astype(int)

        # 4. Port ranges (RFC 6335)
        src_p = df["src_port"].values
        dst_p = df["dst_port"].values

        feat_df["src_port_well_known"] = ((src_p >= 0) & (src_p <= 1023)).astype(int)
        feat_df["src_port_registered"] = ((src_p >= 1024) & (src_p <= 49151)).astype(int)
        feat_df["src_port_ephemeral"] = ((src_p >= 49152) & (src_p <= 65535)).astype(int)

        feat_df["dst_port_well_known"] = ((dst_p >= 0) & (dst_p <= 1023)).astype(int)
        feat_df["dst_port_registered"] = ((dst_p >= 1024) & (dst_p <= 49151)).astype(int)
        feat_df["dst_port_ephemeral"] = ((dst_p >= 49152) & (dst_p <= 65535)).astype(int)

        # 5. Protocol one-hot encoding
        proto = df["protocol"].str.upper()
        feat_df["proto_TCP"] = (proto == "TCP").astype(int)
        feat_df["proto_UDP"] = (proto == "UDP").astype(int)
        feat_df["proto_ICMP"] = (proto == "ICMP").astype(int)

        # Ensure column ordering matches configuration
        feat_df = feat_df[MODEL_FEATURE_COLUMNS]
        return feat_df


class NetFlowPreprocessor(BaseEstimator, TransformerMixin):
    """
    Leak-free preprocessor that fits scalers strictly on baseline training flows
    and applies standard transformations to evaluation flows.
    """

    def __init__(self, scaler_type: str = "robust"):
        self.scaler_type = scaler_type
        if scaler_type == "robust":
            self.scaler = RobustScaler()
        elif scaler_type == "standard":
            self.scaler = StandardScaler()
        else:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}")

        self.numerical_cols = ENGINEERED_NUMERICAL_FEATURES
        self.flag_cols = ENGINEERED_FLAG_FEATURES
        self.feature_names = MODEL_FEATURE_COLUMNS
        self.is_fitted = False

    def fit(self, df_raw: pd.DataFrame, y=None):
        """Fits numerical scalers strictly on baseline flows."""
        extracted = NetFlowFeatureExtractor.extract_raw_features(df_raw)
        self.scaler.fit(extracted[self.numerical_cols])
        self.is_fitted = True
        return self

    def transform(self, df_raw: pd.DataFrame) -> pd.DataFrame:
        """Transforms raw NetFlow records into scaled model features."""
        if not self.is_fitted:
            raise RuntimeError("Preprocessor must be fitted on baseline data before transform.")

        extracted = NetFlowFeatureExtractor.extract_raw_features(df_raw)
        scaled_num = self.scaler.transform(extracted[self.numerical_cols])

        transformed_df = pd.DataFrame(index=df_raw.index)
        for i, col in enumerate(self.numerical_cols):
            transformed_df[col] = scaled_num[:, i]

        for col in self.flag_cols:
            transformed_df[col] = extracted[col].values

        return transformed_df[self.feature_names]

    def fit_transform(self, df_raw: pd.DataFrame, y=None) -> pd.DataFrame:
        """Fits on training raw dataframe and returns transformed features."""
        return self.fit(df_raw, y).transform(df_raw)

