"""
Fusion SecOps Triage Enrichment and SOAR Alert Dispatcher.
Transforms model anomaly scores and SHAP explanations into Splunk CIM-compliant
security events enriched with MITRE ATT&CK tags and automated SOAR playbooks.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import pandas as pd
from src.config import ATTACK_METADATA


class FusionTriageEnricher:
    """
    Enriches detected NetFlow anomalies into production-ready SOC alert records.
    Maps findings to Splunk CIM Network Traffic data model and MITRE ATT&CK.
    """

    def __init__(self, vendor_product: str = "Explainable-NetFlow-IDS v1.0"):
        self.vendor_product = vendor_product

    def _infer_attack_context(self, raw_flow: pd.Series, top_drivers: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Infers the most likely threat archetype based on flow telemetry and top SHAP drivers,
        matching against MITRE ATT&CK intelligence.
        """
        # If true ground truth label is present in evaluation data
        label = raw_flow.get("label", "unknown")
        if label in ATTACK_METADATA:
            return ATTACK_METADATA[label]

        # Heuristic inference if evaluating live unlabelled telemetry
        dst_p = int(raw_flow.get("dst_port", 0))
        proto = str(raw_flow.get("protocol", "")).upper()
        flags = str(raw_flow.get("tcp_flags", ""))
        bpp = float(raw_flow.get("byte_count", 0)) / max(float(raw_flow.get("packet_count", 1)), 1.0)

        if dst_p == 53 and bpp > 400:
            return ATTACK_METADATA["dns_exfiltration"]
        elif flags == "SYN" and "ACK" not in flags and float(raw_flow.get("packet_count", 0)) > 500:
            return ATTACK_METADATA["syn_flood"]
        elif dst_p == 22:
            return ATTACK_METADATA["ssh_brute_force"]
        elif flags == "SYN" and float(raw_flow.get("flow_duration", 0)) < 0.01:
            return ATTACK_METADATA["port_scan"]
        elif dst_p in [8443, 4444, 9001]:
            return ATTACK_METADATA["c2_beacon"]

        # Default fallback
        return {
            "technique_id": "T1046",
            "technique_name": "Network Anomaly / Behavioral Outlier",
            "tactic": "Initial Access / Command and Control",
            "severity": "HIGH",
            "recommended_playbook": "PB-ANOM-001: Inspect source endpoint network socket history; query proxy logs for destination IP reputation.",
        }

    def _calculate_severity(self, anomaly_score: float, threat_meta: Dict[str, Any]) -> str:
        """Determines operational alert priority combining score and inherent threat impact."""
        base_sev = threat_meta.get("severity", "MEDIUM")
        if anomaly_score >= 0.85:
            return "CRITICAL" if base_sev in ["CRITICAL", "HIGH"] else "HIGH"
        elif anomaly_score >= 0.65:
            return "HIGH" if base_sev == "CRITICAL" else base_sev
        else:
            return "MEDIUM"

    def create_alert(
        self,
        raw_flow: pd.Series,
        explanation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Creates a structured, Splunk CIM Network Traffic compliant security alert.
        """
        flow_id = str(raw_flow.get("flow_id", f"FLW-{uuid.uuid4().hex[:8]}"))
        timestamp = raw_flow.get("timestamp", datetime.utcnow().isoformat())
        score = float(explanation["anomaly_score"])
        top_drivers = explanation.get("top_drivers", [])
        narrative = explanation.get("narrative", "")

        threat_meta = self._infer_attack_context(raw_flow, top_drivers)
        severity = self._calculate_severity(score, threat_meta)

        # Splunk CIM Network Traffic schema mapping
        alert_record = {
            "alert_id": f"ALT-{datetime.utcnow().strftime('%Y%m%d')}-{uuid.uuid4().hex[:8]}",
            "timestamp": timestamp,
            "vendor_product": self.vendor_product,
            "flow_id": flow_id,
            "severity": severity,
            "action": "flagged_for_triage",
            "network_traffic": {
                "src": str(raw_flow.get("src_ip")),
                "src_ip": str(raw_flow.get("src_ip")),
                "src_port": int(raw_flow.get("src_port", 0)),
                "dest": str(raw_flow.get("dst_ip")),
                "dest_ip": str(raw_flow.get("dst_ip")),
                "dest_port": int(raw_flow.get("dst_port", 0)),
                "transport": str(raw_flow.get("protocol", "TCP")).lower(),
                "duration": float(raw_flow.get("flow_duration", 0.0)),
                "packets": int(raw_flow.get("packet_count", 0)),
                "bytes": int(raw_flow.get("byte_count", 0)),
                "tcp_flags": str(raw_flow.get("tcp_flags", "NONE")),
            },
            "anomaly_analysis": {
                "calibrated_anomaly_score": score,
                "top_driving_features": top_drivers,
                "soc_justification": narrative,
            },
            "fusion_threat_intel": {
                "mitre_technique_id": threat_meta["technique_id"],
                "mitre_technique_name": threat_meta["technique_name"],
                "mitre_tactic": threat_meta["tactic"],
            },
            "soar_remediation": {
                "recommended_playbook": threat_meta["recommended_playbook"],
                "suggested_containment": f"Block source IP {raw_flow.get('src_ip')} at perimeter firewall or isolate endpoint.",
            },
        }
        return alert_record

    def generate_alerts_batch(
        self,
        df_raw: pd.DataFrame,
        explanations: List[Dict[str, Any]],
        filter_anomalies_only: bool = True,
    ) -> List[Dict[str, Any]]:
        """Generates enriched alert records for a batch of flows."""
        alerts = []
        for i, exp in enumerate(explanations):
            if filter_anomalies_only and not exp.get("is_anomaly", 0):
                continue
            raw_row = df_raw.iloc[i]
            alert = self.create_alert(raw_row, exp)
            alerts.append(alert)
        return alerts

    def export_alerts_to_json(
        self, alerts: List[Dict[str, Any]], output_path: Union[str, Path]
    ) -> Path:
        """Serializes alert batch to formatted JSON file ready for SOAR integration."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(alerts, f, indent=2)
        return path

