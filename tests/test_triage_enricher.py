"""
Tests for Fusion triage enrichment and SOAR alert formatting.
"""

import json
import pytest
import pandas as pd
from pathlib import Path
from src.triage_enricher import FusionTriageEnricher


def test_create_alert_splunk_cim():
    enricher = FusionTriageEnricher()
    raw_flow = pd.Series({
        "flow_id": "FLW-12345",
        "timestamp": "2026-09-20T08:30:00",
        "src_ip": "10.100.1.15",
        "dst_ip": "185.220.101.5",
        "src_port": 51234,
        "dst_port": 53,
        "protocol": "UDP",
        "flow_duration": 4.5,
        "packet_count": 120,
        "byte_count": 85000,
        "tcp_flags": "NONE",
        "label": "dns_exfiltration",
        "is_anomaly": 1,
    })

    explanation = {
        "flow_index": 0,
        "anomaly_score": 0.89,
        "is_anomaly": 1,
        "top_drivers": [
            {"feature": "bytes_per_packet", "shap_contribution": 0.45, "scaled_value": 3.8}
        ],
        "narrative": "Flow flagged as ANOMALOUS due to extreme payload density.",
    }

    alert = enricher.create_alert(raw_flow, explanation)

    assert "alert_id" in alert
    assert alert["severity"] == "CRITICAL"
    assert "network_traffic" in alert
    assert alert["network_traffic"]["src_ip"] == "10.100.1.15"
    assert alert["network_traffic"]["dest_port"] == 53
    assert alert["fusion_threat_intel"]["mitre_technique_id"] == "T1071.004"
    assert "soar_remediation" in alert
    assert "PB-EXFIL-002" in alert["soar_remediation"]["recommended_playbook"]


def test_export_alerts_json(tmp_path: Path):
    enricher = FusionTriageEnricher()
    alerts = [{
        "alert_id": "ALT-TEST-1",
        "severity": "HIGH",
        "narrative": "Suspicious scan detected",
    }]
    output_path = tmp_path / "alerts.json"
    saved = enricher.export_alerts_to_json(alerts, output_path)

    assert saved.exists()
    with open(saved, "r") as f:
        loaded = json.load(f)
    assert len(loaded) == 1
    assert loaded[0]["alert_id"] == "ALT-TEST-1"

