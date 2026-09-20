"""
Explainability Layer (XAI) for NetFlow Intrusion Detection.
Utilizes TreeSHAP to calculate exact local feature attributions and transforms
mathematical Shapley values into actionable natural-language triage narratives for SOC analysts.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import shap
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Any, Optional, Union
from src.config import ExplainabilityConfig, MODEL_FEATURE_COLUMNS
from src.model import NetFlowAnomalyDetector


class NetFlowExplainer:
    """
    TreeSHAP explanation engine for Isolation Forest network flow anomaly detection.
    Translates tree path reductions into positive anomaly attributions.
    """

    def __init__(
        self,
        detector: NetFlowAnomalyDetector,
        feature_names: Optional[List[str]] = None,
        config: Optional[ExplainabilityConfig] = None,
    ):
        if not detector.is_fitted:
            raise RuntimeError("Anomaly detector must be fitted before initializing explainer.")

        self.detector = detector
        self.config = config or ExplainabilityConfig()
        self.feature_names = feature_names or MODEL_FEATURE_COLUMNS
        self.explainer = shap.TreeExplainer(self.detector.model)
        self.baseline_stats: Dict[str, Dict[str, float]] = {}

    def fit_baseline_reference(self, X_train: pd.DataFrame, df_train_raw: Optional[pd.DataFrame] = None) -> None:
        """
        Stores baseline summary statistics (mean, std, median) of benign traffic
        to ground natural-language comparisons in human-interpretable numbers.
        """
        self.baseline_stats = {}
        target_df = df_train_raw if df_train_raw is not None else X_train

        for col in target_df.columns:
            if pd.api.types.is_numeric_dtype(target_df[col]):
                vals = target_df[col].values.astype(float)
                self.baseline_stats[col] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)) if np.std(vals) > 0 else 1.0,
                    "median": float(np.median(vals)),
                }

    def compute_anomaly_shap(self, X: pd.DataFrame) -> np.ndarray:
        """
        Computes SHAP values aligned so that POSITIVE values indicate features
        driving the flow TOWARDS isolation (i.e. increasing anomaly likelihood).
        """
        raw_shap = self.explainer.shap_values(X)
        if isinstance(raw_shap, list):
            raw_shap = raw_shap[0]
        # In TreeSHAP on Isolation Forest, lower score = anomaly (shorter path).
        # We invert the sign so positive SHAP = contributes to anomaly.
        return -np.array(raw_shap)

    def get_feature_importance(self, X: pd.DataFrame) -> pd.DataFrame:
        """Calculates global feature importance as mean absolute anomaly SHAP value."""
        anomaly_shap = self.compute_anomaly_shap(X)
        mean_abs = np.mean(np.abs(anomaly_shap), axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
        return importance_df

    def explain_flow(
        self,
        flow_idx: int,
        X: pd.DataFrame,
        raw_flow: Optional[pd.Series] = None,
        top_k: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Generates local explanation for a single flow record.
        Returns top anomalous features, their attribution values, and a natural language narrative.
        """
        k = top_k or self.config.top_k_features
        row_features = X.iloc[[flow_idx]]
        anomaly_shap = self.compute_anomaly_shap(row_features)[0]

        # Rank features by positive contribution to anomaly
        sorted_indices = np.argsort(-anomaly_shap)
        top_indices = [idx for idx in sorted_indices if anomaly_shap[idx] > 0][:k]
        if not top_indices:
            top_indices = list(sorted_indices[:k])

        top_drivers = []
        for idx in top_indices:
            feat_name = self.feature_names[idx]
            contribution = float(anomaly_shap[idx])
            val = float(row_features.iloc[0, idx])
            raw_val = None
            if raw_flow is not None and feat_name in raw_flow:
                raw_val = raw_flow[feat_name]

            top_drivers.append({
                "feature": feat_name,
                "shap_contribution": round(contribution, 4),
                "scaled_value": round(val, 4),
                "raw_value": raw_val,
            })

        # Build natural-language explanation
        score = float(self.detector.compute_anomaly_scores(row_features)[0])
        nl_text = self._build_narrative(score, top_drivers, raw_flow)

        return {
            "flow_index": flow_idx,
            "anomaly_score": round(score, 4),
            "is_anomaly": int(score >= self.detector.calibrated_threshold),
            "top_drivers": top_drivers,
            "narrative": nl_text,
        }

    def _build_narrative(
        self, score: float, top_drivers: List[Dict[str, Any]], raw_flow: Optional[pd.Series]
    ) -> str:
        """Constructs human-readable narrative summarizing why the flow was flagged."""
        status = "ANOMALOUS" if score >= self.detector.calibrated_threshold else "BENIGN"
        reasons = []

        for d in top_drivers:
            feat = d["feature"]
            contrib = d["shap_contribution"]
            val = d["scaled_value"]

            if feat == "bytes_per_packet":
                raw_bpp = f"{raw_flow['byte_count'] / max(raw_flow['packet_count'], 1):.1f} B/pkt" if raw_flow is not None else f"scaled={val}"
                reasons.append(f"Payload density `{feat}` ({raw_bpp}, SHAP: +{contrib}) significantly exceeds baseline")
            elif feat == "packets_per_second":
                reasons.append(f"Transmission rate `{feat}` (SHAP: +{contrib}) indicates volumetric burst")
            elif feat == "is_syn_only" and val > 0.5:
                reasons.append(f"TCP flag profile is strictly `SYN-only` without handshake completion (SHAP: +{contrib}), typical of scans or SYN floods")
            elif feat == "log_duration" and val < 0.0:
                reasons.append(f"Flow duration was anomalously brief for the requested port (SHAP: +{contrib})")
            elif feat == "proto_UDP" and val > 0.5:
                reasons.append(f"Protocol is UDP on atypical service port (SHAP: +{contrib})")
            elif feat.startswith("dst_port"):
                reasons.append(f"Destination port profile `{feat}` deviates from baseline enterprise behavior (SHAP: +{contrib})")
            else:
                reasons.append(f"`{feat}` deviated strongly from baseline distribution (SHAP: +{contrib})")

        driver_str = "; ".join(reasons) if reasons else "no single dominant outlier attribute"
        narrative = (
            f"Flow flagged as {status} (calibrated anomaly score: {score:.3f}, threshold: {self.detector.calibrated_threshold:.3f}). "
            f"Primary operational drivers: {driver_str}."
        )
        return narrative

    def plot_feature_importance(
        self, X: pd.DataFrame, top_n: int = 10, save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Generates and optionally saves a bar chart of global feature importances."""
        df_imp = self.get_feature_importance(X).head(top_n)

        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.barh(df_imp["feature"][::-1], df_imp["mean_abs_shap"][::-1], color="#1f77b4", edgecolor="black")
        ax.set_xlabel("Mean |Anomaly SHAP Value|", fontsize=12)
        ax.set_title("Global Feature Importance (TreeSHAP Isolation Forest)", fontsize=14, fontweight="bold")
        ax.grid(axis="x", linestyle="--", alpha=0.6)
        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300)

        return fig

    def plot_waterfall(
        self, flow_idx: int, X: pd.DataFrame, save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Generates local SHAP waterfall plot for a specific flow."""
        row = X.iloc[[flow_idx]]
        anomaly_shap = self.compute_anomaly_shap(row)[0]
        base_val = -float(self.explainer.expected_value)

        # Create shap Explanation object
        exp = shap.Explanation(
            values=anomaly_shap,
            base_values=base_val,
            data=row.iloc[0].values,
            feature_names=self.feature_names,
        )

        fig = plt.figure(figsize=(10, 6))
        shap.plots.waterfall(exp, max_display=8, show=False)
        plt.title(f"Local Anomaly Attribution (Flow #{flow_idx})", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            plt.savefig(path, dpi=300, bbox_inches="tight")

        return fig

