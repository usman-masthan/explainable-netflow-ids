"""
Configuration and Schema Definitions for Explainable NetFlow IDS.
Designed for LSEG Security Operations & Fusion Management.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Any


# Root paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"


# Standard NetFlow v9 Schema
RAW_FLOW_COLUMNS: List[str] = [
    "flow_id",
    "timestamp",
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",
    "protocol",
    "flow_duration",
    "packet_count",
    "byte_count",
    "tcp_flags",
    "tos",
    "label",
    "is_anomaly",
]

# Engineered behavioral numerical features for anomaly detection
ENGINEERED_NUMERICAL_FEATURES: List[str] = [
    "bytes_per_packet",
    "packets_per_second",
    "bytes_per_second",
    "log_bytes",
    "log_packets",
    "log_duration",
    "log_bps",
    "log_pps",
]

# Engineered categorical / binary flags
ENGINEERED_FLAG_FEATURES: List[str] = [
    "flag_SYN",
    "flag_ACK",
    "flag_FIN",
    "flag_RST",
    "flag_PSH",
    "flag_URG",
    "is_syn_only",
    "src_port_well_known",
    "src_port_registered",
    "src_port_ephemeral",
    "dst_port_well_known",
    "dst_port_registered",
    "dst_port_ephemeral",
    "proto_TCP",
    "proto_UDP",
    "proto_ICMP",
]

MODEL_FEATURE_COLUMNS: List[str] = (
    ENGINEERED_NUMERICAL_FEATURES + ENGINEERED_FLAG_FEATURES
)

# Attack Types & MITRE ATT&CK Mapping for Fusion Operations
ATTACK_METADATA: Dict[str, Dict[str, Any]] = {
    "dns_exfiltration": {
        "technique_id": "T1071.004",
        "technique_name": "Application Layer Protocol: DNS",
        "tactic": "Exfiltration",
        "severity": "CRITICAL",
        "recommended_playbook": "PB-EXFIL-002: Isolate host endpoint, inspect DNS query payload length, sinkhole external resolver.",
    },
    "c2_beacon": {
        "technique_id": "T1071.001",
        "technique_name": "Application Layer Protocol: Web Protocols",
        "tactic": "Command and Control",
        "severity": "HIGH",
        "recommended_playbook": "PB-C2-004: Quarantine external IP at border firewall, pull process tree from source host, analyze EDR telemetry.",
    },
    "port_scan": {
        "technique_id": "T1046",
        "technique_name": "Network Service Discovery",
        "tactic": "Discovery",
        "severity": "MEDIUM",
        "recommended_playbook": "PB-RECON-001: Rate-limit or block source IP, verify external perimeter exposure on scanned ports.",
    },
    "syn_flood": {
        "technique_id": "T1498.001",
        "technique_name": "Network Denial of Service: Direct Network Flood",
        "tactic": "Impact",
        "severity": "CRITICAL",
        "recommended_playbook": "PB-DDOS-001: Enable SYN cookies on perimeter routers, trigger upstream scrubbing center mitigation.",
    },
    "ssh_brute_force": {
        "technique_id": "T1110.001",
        "technique_name": "Brute Force: Password Guessing",
        "tactic": "Credential Access",
        "severity": "HIGH",
        "recommended_playbook": "PB-AUTH-003: Temporarily lock compromised accounts, enforce MFA / public-key auth, block source IP.",
    },
}


@dataclass
class DataGenConfig:
    """Configuration for synthetic NetFlow generator."""
    n_samples: int = 10000
    anomaly_ratio: float = 0.10  # 10% anomalies in test evaluation
    random_state: int = 42
    benign_ratio_https: float = 0.55
    benign_ratio_dns: float = 0.20
    benign_ratio_db: float = 0.15
    benign_ratio_mgmt: float = 0.10


@dataclass
class ModelConfig:
    """Configuration for Isolation Forest Model."""
    n_estimators: int = 150
    max_samples: str = "auto"
    contamination: float = 0.05  # Initial contamination assumption
    random_state: int = 42
    n_jobs: int = -1


@dataclass
class ExplainabilityConfig:
    """Configuration for TreeSHAP Explainability Layer."""
    background_samples: int = 100
    top_k_features: int = 4
    random_state: int = 42


@dataclass
class EvalConfig:
    """Configuration for SecOps Evaluation & Threshold Tuning."""
    target_fpr: float = 0.02  # Target maximum 2% False Positive Rate in SOC
    n_threshold_steps: int = 100

