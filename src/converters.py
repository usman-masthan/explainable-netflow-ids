"""
External Benchmark Dataset Converters for CIDDS-001 and CIC-IDS2017.
Normalizes public benchmark datasets into standard NetFlow v9 schema for
cross-dataset generalization in Fusion SOC evaluation.
"""

import uuid
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Union, Dict, Any, Optional
from src.config import RAW_FLOW_COLUMNS


class ExternalDatasetConverter:
    """Base class for benchmark dataset converters."""

    @staticmethod
    def _generate_flow_id(prefix: str = "EXT") -> str:
        return f"{prefix}-{uuid.uuid4().hex[:12]}"


class CIDDS001Converter(ExternalDatasetConverter):
    """
    Converts Coburg Intrusion Detection Data Set (CIDDS-001) NetFlow v9 format
    into the standard Explainable NetFlow IDS schema.
    """

    # CIDDS-001 Flag character mapping
    FLAG_MAP = {
        0: "URG",
        1: "ACK",
        2: "PSH",
        3: "RST",
        4: "SYN",
        5: "FIN",
    }

    ATTACK_TYPE_MAP = {
        "---": "benign",
        "portscan": "port_scan",
        "dos": "syn_flood",
        "pingScan": "port_scan",
        "bruteForce": "ssh_brute_force",
        "unknown": "benign",
    }

    @classmethod
    def parse_cidds_flags(cls, flag_str: str) -> str:
        """
        Parses CIDDS TCP flag notation (e.g. '....S.', '.AP.SF') into
        standard comma-separated flag string.
        """
        if not isinstance(flag_str, str) or flag_str.strip() == "":
            return "NONE"

        clean = flag_str.strip()
        active_flags = []
        # CIDDS-001 flag order: U, A, P, R, S, F
        if len(clean) >= 6:
            if clean[0] in ["U", "1"]:
                active_flags.append("URG")
            if clean[1] in ["A", "1"]:
                active_flags.append("ACK")
            if clean[2] in ["P", "1"]:
                active_flags.append("PSH")
            if clean[3] in ["R", "1"]:
                active_flags.append("RST")
            if clean[4] in ["S", "1"]:
                active_flags.append("SYN")
            if clean[5] in ["F", "1"]:
                active_flags.append("FIN")
        else:
            # Fallback for standard letters
            for flag in ["URG", "ACK", "PSH", "RST", "SYN", "FIN"]:
                if flag[0] in clean.upper():
                    active_flags.append(flag)

        return ",".join(active_flags) if active_flags else "NONE"

    @classmethod
    def parse_cidds_bytes(cls, byte_val: Any) -> int:
        """Handles numeric or formatted byte strings (e.g. '1.5 M', '240 K')."""
        if pd.isna(byte_val):
            return 0
        if isinstance(byte_val, (int, float)):
            return int(byte_val)

        val_str = str(byte_val).strip().upper()
        if val_str.endswith("M"):
            return int(float(val_str[:-1].strip()) * 1_000_000)
        elif val_str.endswith("K"):
            return int(float(val_str[:-1].strip()) * 1_000)
        try:
            return int(float(val_str))
        except ValueError:
            return 0

    @classmethod
    def convert_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a CIDDS-001 raw DataFrame into the standard schema."""
        converted = pd.DataFrame()

        # Handle column naming variations in CIDDS-001
        col_map = {c.lower().strip(): c for c in df.columns}

        date_col = col_map.get("date first seen", col_map.get("date", None))
        converted["timestamp"] = df[date_col].astype(str) if date_col else "2026-09-20T08:00:00"

        dur_col = col_map.get("duration", "duration")
        converted["flow_duration"] = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0)

        proto_col = col_map.get("proto", "proto")
        converted["protocol"] = df[proto_col].astype(str).str.upper()

        src_ip_col = col_map.get("src ip addr", col_map.get("src_ip", "src_ip"))
        converted["src_ip"] = df[src_ip_col].astype(str)

        src_pt_col = col_map.get("src pt", col_map.get("src_port", "src_port"))
        converted["src_port"] = pd.to_numeric(df[src_pt_col], errors="coerce").fillna(0).astype(int)

        dst_ip_col = col_map.get("dst ip addr", col_map.get("dst_ip", "dst_ip"))
        converted["dst_ip"] = df[dst_ip_col].astype(str)

        dst_pt_col = col_map.get("dst pt", col_map.get("dst_port", "dst_port"))
        converted["dst_port"] = pd.to_numeric(df[dst_pt_col], errors="coerce").fillna(0).astype(int)

        pkt_col = col_map.get("packets", "packets")
        converted["packet_count"] = pd.to_numeric(df[pkt_col], errors="coerce").fillna(1).astype(int)

        byte_col = col_map.get("bytes", "bytes")
        converted["byte_count"] = df[byte_col].apply(cls.parse_cidds_bytes)

        flag_col = col_map.get("flags", "flags")
        if flag_col in df.columns:
            converted["tcp_flags"] = df[flag_col].apply(cls.parse_cidds_flags)
        else:
            converted["tcp_flags"] = "NONE"

        tos_col = col_map.get("tos", "tos")
        converted["tos"] = pd.to_numeric(df[tos_col], errors="coerce").fillna(0).astype(int) if tos_col in df.columns else 0

        # Label mapping
        atk_col = col_map.get("attacktype", col_map.get("class", None))
        if atk_col:
            raw_labels = df[atk_col].astype(str).str.strip()
            converted["label"] = raw_labels.map(lambda x: cls.ATTACK_TYPE_MAP.get(x.lower(), cls.ATTACK_TYPE_MAP.get(x, "benign")))
        else:
            converted["label"] = "benign"

        converted["is_anomaly"] = (converted["label"] != "benign").astype(int)
        converted["flow_id"] = [cls._generate_flow_id("CIDDS") for _ in range(len(converted))]

        return converted[RAW_FLOW_COLUMNS]


class CICIDS2017Converter(ExternalDatasetConverter):
    """
    Converts Canadian Institute for Cybersecurity (CIC-IDS2017) flow telemetry
    into the standard Explainable NetFlow IDS schema.
    """

    ATTACK_MAP = {
        "benign": "benign",
        "portscan": "port_scan",
        "ddos": "syn_flood",
        "dos hulk": "syn_flood",
        "dos goldeneye": "syn_flood",
        "dos slowloris": "syn_flood",
        "dos slowhttptest": "syn_flood",
        "ssh-patator": "ssh_brute_force",
        "ftp-patator": "ssh_brute_force",
        "bot": "c2_beacon",
        "infiltration": "c2_beacon",
    }

    @classmethod
    def convert_dataframe(cls, df: pd.DataFrame) -> pd.DataFrame:
        """Transforms a CIC-IDS2017 flow DataFrame into the standard schema."""
        converted = pd.DataFrame()
        col_clean = {c.strip().lower(): c for c in df.columns}

        # Timestamp
        time_col = col_clean.get("timestamp", None)
        converted["timestamp"] = df[time_col].astype(str) if time_col else "2026-09-20T08:00:00"

        # Duration (CICFlowMeter records duration in microseconds)
        dur_col = col_clean.get("flow duration", None)
        if dur_col:
            converted["flow_duration"] = pd.to_numeric(df[dur_col], errors="coerce").fillna(0.0) / 1_000_000.0
        else:
            converted["flow_duration"] = 0.0

        # IPs and Ports
        src_ip = col_clean.get("source ip", None)
        converted["src_ip"] = df[src_ip].astype(str) if src_ip else "10.100.1.10"

        src_pt = col_clean.get("source port", None)
        converted["src_port"] = pd.to_numeric(df[src_pt], errors="coerce").fillna(0).astype(int) if src_pt else 49152

        dst_ip = col_clean.get("destination ip", None)
        converted["dst_ip"] = df[dst_ip].astype(str) if dst_ip else "10.200.1.53"

        dst_pt = col_clean.get("destination port", None)
        converted["dst_port"] = pd.to_numeric(df[dst_pt], errors="coerce").fillna(0).astype(int) if dst_pt else 443

        # Protocol (6 = TCP, 17 = UDP, 1 = ICMP)
        proto_col = col_clean.get("protocol", None)
        if proto_col:
            proto_nums = pd.to_numeric(df[proto_col], errors="coerce").fillna(6).astype(int)
            proto_map = {6: "TCP", 17: "UDP", 1: "ICMP"}
            converted["protocol"] = proto_nums.map(lambda p: proto_map.get(p, "TCP"))
        else:
            converted["protocol"] = "TCP"

        # Packets (Fwd + Bwd)
        fwd_p = col_clean.get("total fwd packets", None)
        bwd_p = col_clean.get("total backward packets", None)
        fwd_pkts = pd.to_numeric(df[fwd_p], errors="coerce").fillna(0) if fwd_p else 0
        bwd_pkts = pd.to_numeric(df[bwd_p], errors="coerce").fillna(0) if bwd_p else 0
        converted["packet_count"] = np.maximum((fwd_pkts + bwd_pkts).astype(int), 1)

        # Bytes (Fwd + Bwd length)
        fwd_b = col_clean.get("total length of fwd packets", None)
        bwd_b = col_clean.get("total length of bwd packets", None)
        fwd_bytes = pd.to_numeric(df[fwd_b], errors="coerce").fillna(0) if fwd_b else 0
        bwd_bytes = pd.to_numeric(df[bwd_b], errors="coerce").fillna(0) if bwd_b else 0
        converted["byte_count"] = np.maximum((fwd_bytes + bwd_bytes).astype(int), 0)

        # Reconstruct TCP flags string from individual flag counts
        def build_flags(row):
            flags = []
            if row.get(col_clean.get("fin flag count", "")) == 1:
                flags.append("FIN")
            if row.get(col_clean.get("syn flag count", "")) == 1:
                flags.append("SYN")
            if row.get(col_clean.get("rst flag count", "")) == 1:
                flags.append("RST")
            if row.get(col_clean.get("psh flag count", "")) == 1:
                flags.append("PSH")
            if row.get(col_clean.get("ack flag count", "")) == 1:
                flags.append("ACK")
            if row.get(col_clean.get("urg flag count", "")) == 1:
                flags.append("URG")
            return ",".join(flags) if flags else "NONE"

        converted["tcp_flags"] = df.apply(build_flags, axis=1)
        converted["tos"] = 0

        # Label mapping
        label_col = col_clean.get("label", None)
        if label_col:
            raw_labels = df[label_col].astype(str).str.strip().str.lower()
            converted["label"] = raw_labels.map(lambda l: cls.ATTACK_MAP.get(l, "port_scan" if "port" in l else "syn_flood" if "dos" in l else "benign"))
        else:
            converted["label"] = "benign"

        converted["is_anomaly"] = (converted["label"] != "benign").astype(int)
        converted["flow_id"] = [cls._generate_flow_id("CIC") for _ in range(len(converted))]

        return converted[RAW_FLOW_COLUMNS]


def create_sample_cidds_csv(output_path: Union[str, Path], n_samples: int = 200) -> Path:
    """
    Generates a realistic sample CIDDS-001 raw formatted CSV file for testing
    and offline evaluation without needing to download gigabytes of raw data.
    """
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n_samples):
        is_attack = rng.random() < 0.20
        if is_attack:
            attack_type = rng.choice(["portScan", "dos", "bruteForce"])
            flags = "....S." if attack_type == "portScan" else "....S." if attack_type == "dos" else ".AP.SF"
            pkts = rng.integers(1, 4) if attack_type == "portScan" else rng.integers(2000, 10000) if attack_type == "dos" else 24
            bytes_str = f"{pkts * 44}" if attack_type in ["portScan", "dos"] else f"{pkts * 90}"
            dur = rng.uniform(0.001, 0.05) if attack_type == "portScan" else rng.uniform(0.1, 1.5)
            src_ip = f"194.168.10.{rng.integers(1, 255)}"
            dst_pt = rng.choice([22, 80, 443, 8080])
        else:
            attack_type = "---"
            flags = ".AP.SF"
            pkts = int(rng.integers(8, 150))
            bytes_val = int(pkts * rng.uniform(200, 900))
            bytes_str = f"{bytes_val / 1000:.1f} K" if bytes_val > 10000 else str(bytes_val)
            dur = rng.uniform(0.1, 12.0)
            src_ip = f"10.100.1.{rng.integers(10, 50)}"
            dst_pt = int(rng.choice([53, 443, 80, 1433]))

        rows.append({
            "Date first seen": f"2026-09-20 08:{rng.integers(10, 59):02d}:{rng.integers(10, 59):02d}.000",
            "Duration": round(dur, 3),
            "Proto": "TCP" if dst_pt != 53 else "UDP",
            "Src IP Addr": src_ip,
            "Src Pt": int(rng.integers(49152, 65535)),
            "Dst IP Addr": f"10.200.1.{rng.integers(10, 30)}",
            "Dst Pt": dst_pt,
            "Packets": pkts,
            "Bytes": bytes_str,
            "Flags": flags,
            "Tos": 0,
            "class": "attacker" if is_attack else "normal",
            "attackType": attack_type,
            "attackID": "---" if not is_attack else f"ATK-{rng.integers(100, 999)}",
        })

    df = pd.DataFrame(rows)
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    return path
