"""
SecOps Performance Evaluation and Alert Fatigue Trade-off Analysis.
Calculates AUROC, PR-AUC, False Positive Rates, and operational alert fatigue metrics
for Fusion Security Operations Centers.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    roc_curve,
    precision_recall_curve,
    confusion_matrix,
    classification_report,
)


class SecOpsEvaluator:
    """
    Evaluates anomaly detection performance with a focus on operational SecOps
    realities: False Positive Rate (FPR) budgets, analyst fatigue, and per-attack recall.
    """

    def __init__(self, target_daily_flow_volume: int = 1_000_000):
        self.target_daily_flow_volume = target_daily_flow_volume

    def evaluate_detection(
        self,
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        threshold: float,
        attack_labels: Optional[pd.Series] = None,
    ) -> Dict[str, Any]:
        """
        Computes standard and operational SecOps metrics for a given decision threshold.
        """
        y_pred = (anomaly_scores >= threshold).astype(int)

        auroc = float(roc_auc_score(y_true, anomaly_scores))
        pr_auc = float(average_precision_score(y_true, anomaly_scores))

        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        tn, fp, fn, tp = cm.ravel()

        precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
        recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
        f1 = float(2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
        fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

        daily_false_alarms = int(fpr * self.target_daily_flow_volume)
        total_daily_alerts = int((y_pred.mean()) * self.target_daily_flow_volume)

        # Per-attack detection breakdown
        attack_breakdown = {}
        if attack_labels is not None:
            df_eval = pd.DataFrame({"label": attack_labels, "y_pred": y_pred})
            for atk in df_eval["label"].unique():
                if atk != "benign":
                    sub = df_eval[df_eval["label"] == atk]
                    attack_breakdown[atk] = {
                        "count": int(len(sub)),
                        "detected": int(sub["y_pred"].sum()),
                        "recall": round(float(sub["y_pred"].mean()), 4),
                    }

        return {
            "auroc": round(auroc, 4),
            "pr_auc": round(pr_auc, 4),
            "threshold": round(threshold, 4),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1_score": round(f1, 4),
            "false_positive_rate": round(fpr, 5),
            "confusion_matrix": {
                "true_negatives": int(tn),
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
            },
            "operational_impact": {
                "assumed_daily_flows": self.target_daily_flow_volume,
                "projected_daily_alerts": total_daily_alerts,
                "projected_daily_false_alarms": daily_false_alarms,
            },
            "per_attack_recall": attack_breakdown,
        }

    def compute_alert_tradeoff_curve(
        self, y_true: np.ndarray, anomaly_scores: np.ndarray, steps: int = 50
    ) -> pd.DataFrame:
        """
        Calculates trade-offs between Alert Volume, Detection Rate (Recall),
        and False Positive Rate across 50 threshold cutoffs.
        """
        thresholds = np.linspace(0.1, 0.95, steps)
        records = []

        for th in thresholds:
            y_pred = (anomaly_scores >= th).astype(int)
            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            tn, fp, fn, tp = cm.ravel()

            rec = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            alert_pct = y_pred.mean() * 100.0
            daily_alerts = int((alert_pct / 100.0) * self.target_daily_flow_volume)

            records.append({
                "threshold": round(float(th), 3),
                "recall": round(float(rec), 4),
                "precision": round(float(prec), 4),
                "fpr": round(float(fpr), 5),
                "alert_percentage": round(float(alert_pct), 3),
                "projected_daily_alerts": daily_alerts,
            })

        return pd.DataFrame(records)

    def plot_roc_and_pr_curves(
        self,
        y_true: np.ndarray,
        anomaly_scores: np.ndarray,
        save_path: Optional[Union[str, Path]] = None,
    ) -> plt.Figure:
        """Generates ROC and Precision-Recall plots side by side."""
        fpr, tpr, _ = roc_curve(y_true, anomaly_scores)
        auroc = roc_auc_score(y_true, anomaly_scores)

        precision, recall, _ = precision_recall_curve(y_true, anomaly_scores)
        pr_auc = average_precision_score(y_true, anomaly_scores)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # ROC Curve
        ax1.plot(fpr, tpr, color="#0052cc", lw=2, label=f"Isolation Forest (AUROC = {auroc:.3f})")
        ax1.plot([0, 1], [0, 1], color="gray", lw=1, linestyle="--", label="Random Chance")
        ax1.set_xlim([0.0, 1.0])
        ax1.set_ylim([0.0, 1.05])
        ax1.set_xlabel("False Positive Rate (FPR)", fontsize=11)
        ax1.set_ylabel("True Positive Rate (Recall)", fontsize=11)
        ax1.set_title("Receiver Operating Characteristic (ROC)", fontsize=13, fontweight="bold")
        ax1.grid(True, linestyle="--", alpha=0.5)
        ax1.legend(loc="lower right")

        # PR Curve
        ax2.plot(recall, precision, color="#00875a", lw=2, label=f"Precision-Recall (PR-AUC = {pr_auc:.3f})")
        baseline_rate = y_true.mean()
        ax2.plot([0, 1], [baseline_rate, baseline_rate], color="gray", lw=1, linestyle="--", label=f"No-skill ({baseline_rate:.2f})")
        ax2.set_xlim([0.0, 1.0])
        ax2.set_ylim([0.0, 1.05])
        ax2.set_xlabel("Recall (Detection Rate)", fontsize=11)
        ax2.set_ylabel("Precision", fontsize=11)
        ax2.set_title("Precision-Recall Curve (PR-AUC)", fontsize=13, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.5)
        ax2.legend(loc="lower left")

        plt.tight_layout()
        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300)

        return fig

    def plot_alert_tradeoff(
        self, tradeoff_df: pd.DataFrame, save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Visualizes Alert Volume vs Detection Rate to identify the optimal SOC operating point."""
        fig, ax1 = plt.subplots(figsize=(10, 6))

        color1 = "#0052cc"
        color2 = "#de350b"

        ax1.set_xlabel("Anomaly Decision Threshold", fontsize=11)
        ax1.set_ylabel("Recall / Detection Rate", color=color1, fontsize=11)
        line1 = ax1.plot(tradeoff_df["threshold"], tradeoff_df["recall"], color=color1, lw=2.5, label="Detection Rate (Recall)")
        ax1.tick_params(axis="y", labelcolor=color1)
        ax1.grid(True, linestyle="--", alpha=0.5)

        ax2 = ax1.twinx()
        ax2.set_ylabel("False Positive Rate (FPR)", color=color2, fontsize=11)
        line2 = ax2.plot(tradeoff_df["threshold"], tradeoff_df["fpr"], color=color2, lw=2.5, linestyle="--", label="False Positive Rate")
        ax2.tick_params(axis="y", labelcolor=color2)

        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc="center right")

        plt.title("SecOps Alert Volume vs Detection Trade-off", fontsize=13, fontweight="bold")
        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300)

        return fig

    def plot_confusion_matrix(
        self, cm_dict: Dict[str, int], save_path: Optional[Union[str, Path]] = None
    ) -> plt.Figure:
        """Plots an annotated confusion matrix with SecOps context."""
        cm = np.array([
            [cm_dict["true_negatives"], cm_dict["false_positives"]],
            [cm_dict["false_negatives"], cm_dict["true_positives"]],
        ])

        fig, ax = plt.subplots(figsize=(7, 5))
        cax = ax.matshow(cm, cmap="Blues", alpha=0.8)

        for (i, j), z in np.ndenumerate(cm):
            label = f"{z:,}\n"
            if i == 0 and j == 0:
                label += "(Benign Ignored)"
            elif i == 0 and j == 1:
                label += "(False Alarms)"
            elif i == 1 and j == 0:
                label += "(Missed Attacks)"
            else:
                label += "(True Alerts)"
            ax.text(j, i, label, ha="center", va="center", fontsize=11, fontweight="bold")

        fig.colorbar(cax)
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["Predicted Benign", "Predicted Anomaly"], fontsize=11)
        ax.set_yticklabels(["Actual Benign", "Actual Attack"], fontsize=11)
        ax.set_title("SecOps Confusion Matrix", fontsize=13, fontweight="bold", pad=20)
        plt.tight_layout()

        if save_path:
            path = Path(save_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=300)

        return fig

