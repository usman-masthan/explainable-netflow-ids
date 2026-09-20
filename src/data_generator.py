"""
Realistic NetFlow v9 Telemetry Generator for Fusion SOC Anomaly Detection.
Simulates high-fidelity benign enterprise baselines and realistic cyber threat archetypes.
"""

import uuid
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from typing import Tuple, List, Optional
from src.config import DataGenConfig, RAW_FLOW_COLUMNS


class NetFlowDataGenerator:
    """
    Generates realistic NetFlow v9 records modeled after corporate and financial
    market infrastructure networks.
    """

    def __init__(self, config: Optional[DataGenConfig] = None, random_state: Optional[int] = None):
        if config is not None:
            self.config = config
        else:
            self.config = DataGenConfig()
            if random_state is not None:
                self.config.random_state = random_state
        self.rng = np.random.default_rng(self.config.random_state)

        # Internal subnet pools
        self.internal_client_ips = [
            f"10.100.{subnet}.{host}"
            for subnet in range(1, 10)
            for host in range(10, 50)
        ]
        self.internal_server_ips = [
            f"10.200.{subnet}.{host}"
            for subnet in range(1, 4)
            for host in range(10, 30)
        ]
        self.dns_server_ips = ["10.200.1.53", "10.200.2.53"]
        self.db_server_ips = ["10.200.3.10", "10.200.3.20", "10.200.3.30"]

        # External IP pools
        self.external_legit_ips = [
            f"{self.rng.choice([151, 157, 104, 13, 20])}.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}"
            for _ in range(100)
        ]
        self.external_threat_ips = [
            f"{self.rng.choice([185, 194, 91, 45, 193])}.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}.{self.rng.integers(1, 255)}"
            for _ in range(20)
        ]

    def _sample_ip(self, ip_list: List[str]) -> str:
        return str(self.rng.choice(ip_list))

    def _generate_flow_id(self) -> str:
        rand_suffix = self.rng.integers(100000, 999999)
        return f"FLW-{rand_suffix}-{uuid.uuid4().hex[:8]}"

    def generate_benign_flow(self, base_time: datetime) -> dict:
        """Generates a single benign network flow according to enterprise baselines."""
        category_choice = self.rng.choice(
            ["https", "dns", "db", "mgmt"],
            p=[
                self.config.benign_ratio_https,
                self.config.benign_ratio_dns,
                self.config.benign_ratio_db,
                self.config.benign_ratio_mgmt,
            ],
        )

        flow_time = base_time + timedelta(seconds=float(self.rng.uniform(0, 3600)))
        src_ip = self._sample_ip(self.internal_client_ips)
        src_port = int(self.rng.integers(49152, 65535))

        if category_choice == "https":
            dst_ip = self._sample_ip(self.external_legit_ips)
            dst_port = 443
            protocol = "TCP"
            duration = float(np.clip(self.rng.exponential(scale=2.5), 0.05, 30.0))
            packets = int(np.clip(self.rng.negative_binomial(10, 0.15), 6, 800))
            # Typical TLS payload byte distribution (500 - 1400 bytes/packet)
            avg_bpp = float(self.rng.normal(loc=750, scale=180))
            bytes_count = int(max(packets * 54, packets * avg_bpp))
            flags = "SYN,ACK,PSH,FIN"
            tos = 0

        elif category_choice == "dns":
            dst_ip = self._sample_ip(self.dns_server_ips)
            dst_port = 53
            protocol = "UDP"
            duration = float(self.rng.uniform(0.002, 0.060))
            packets = int(self.rng.choice([2, 4], p=[0.85, 0.15]))
            # Standard query/response DNS payload (70 - 280 bytes)
            bytes_count = int(self.rng.integers(120, 520))
            flags = "NONE"
            tos = 0

        elif category_choice == "db":
            src_ip = self._sample_ip(self.internal_server_ips)
            dst_ip = self._sample_ip(self.db_server_ips)
            dst_port = int(self.rng.choice([5432, 1433]))
            protocol = "TCP"
            duration = float(np.clip(self.rng.gamma(shape=2.0, scale=3.0), 0.2, 45.0))
            packets = int(np.clip(self.rng.negative_binomial(15, 0.1), 10, 1500))
            avg_bpp = float(self.rng.normal(loc=900, scale=150))
            bytes_count = int(max(packets * 54, packets * avg_bpp))
            flags = "SYN,ACK,PSH"
            tos = 16  # Low delay QoS

        else:  # mgmt (NTP/SNMP/internal ICMP)
            dst_ip = self._sample_ip(self.internal_server_ips)
            protocol_choice = self.rng.choice(["UDP", "ICMP"], p=[0.8, 0.2])
            if protocol_choice == "UDP":
                dst_port = 123
                protocol = "UDP"
                duration = float(self.rng.uniform(0.001, 0.020))
                packets = 2
                bytes_count = 96
                flags = "NONE"
            else:
                dst_port = 0
                protocol = "ICMP"
                duration = float(self.rng.uniform(0.001, 0.010))
                packets = 2
                bytes_count = 128
                flags = "NONE"
            tos = 0

        return {
            "flow_id": self._generate_flow_id(),
            "timestamp": flow_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "flow_duration": round(duration, 5),
            "packet_count": packets,
            "byte_count": bytes_count,
            "tcp_flags": flags,
            "tos": tos,
            "label": "benign",
            "is_anomaly": 0,
        }

    def generate_attack_flow(self, attack_type: str, base_time: datetime) -> dict:
        """Generates an anomalous flow belonging to a specific cyber threat archetype."""
        flow_time = base_time + timedelta(seconds=float(self.rng.uniform(0, 3600)))
        src_port = int(self.rng.integers(49152, 65535))

        if attack_type == "dns_exfiltration":
            # DNS Tunneling: excessive byte count, high packet count, unexpected payload density
            src_ip = self._sample_ip(self.internal_client_ips)
            dst_ip = self._sample_ip(self.external_threat_ips)
            dst_port = 53
            protocol = "UDP"
            duration = float(self.rng.uniform(1.5, 12.0))
            packets = int(self.rng.integers(60, 450))
            # Inflated payload: 600 - 1200 bytes per packet in DNS is anomalous
            bytes_count = int(packets * self.rng.uniform(550, 1100))
            flags = "NONE"
            tos = 0

        elif attack_type == "c2_beacon":
            # Highly consistent, low-entropy heartbeat to malicious external infrastructure
            src_ip = self._sample_ip(self.internal_client_ips)
            dst_ip = self._sample_ip(self.external_threat_ips)
            dst_port = int(self.rng.choice([8443, 4444, 9001]))
            protocol = "TCP"
            duration = float(self.rng.uniform(0.04, 0.15))
            packets = int(self.rng.choice([4, 6, 8]))
            # Fixed payload size across beacon pulses
            bytes_count = int(packets * 64)
            flags = "SYN,ACK,PSH,FIN"
            tos = 0

        elif attack_type == "port_scan":
            # Fast reconnaissance sweep: SYN-only or RST, near-zero duration, tiny packet
            src_ip = self._sample_ip(self.external_threat_ips)
            dst_ip = self._sample_ip(self.internal_server_ips)
            dst_port = int(self.rng.choice([21, 22, 23, 25, 80, 443, 445, 1433, 3389, 8080]))
            protocol = "TCP"
            duration = float(self.rng.uniform(0.0001, 0.004))
            packets = int(self.rng.choice([1, 2]))
            bytes_count = int(packets * 44)  # TCP SYN packet header size
            flags = "SYN" if self.rng.random() > 0.3 else "RST"
            tos = 0

        elif attack_type == "syn_flood":
            # Volumetric denial of service: extreme packet rate, SYN only, no payload
            src_ip = self._sample_ip(self.external_threat_ips)
            dst_ip = self._sample_ip(self.internal_server_ips)
            dst_port = 443
            protocol = "TCP"
            duration = float(self.rng.uniform(0.05, 1.2))
            packets = int(self.rng.integers(3000, 25000))
            bytes_count = int(packets * 54)
            flags = "SYN"
            tos = 0

        elif attack_type == "ssh_brute_force":
            # High-frequency auth attempts against SSH port 22
            src_ip = self._sample_ip(self.external_threat_ips)
            dst_ip = self._sample_ip(self.internal_server_ips)
            dst_port = 22
            protocol = "TCP"
            duration = float(self.rng.uniform(0.8, 2.2))
            packets = int(self.rng.integers(22, 34))
            bytes_count = int(packets * self.rng.uniform(90, 130))
            flags = "SYN,ACK,PSH,FIN"
            tos = 0

        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        return {
            "flow_id": self._generate_flow_id(),
            "timestamp": flow_time.isoformat(),
            "src_ip": src_ip,
            "dst_ip": dst_ip,
            "src_port": src_port,
            "dst_port": dst_port,
            "protocol": protocol,
            "flow_duration": round(duration, 5),
            "packet_count": packets,
            "byte_count": bytes_count,
            "tcp_flags": flags,
            "tos": tos,
            "label": attack_type,
            "is_anomaly": 1,
        }

    def generate_dataset(
        self, n_samples: Optional[int] = None, anomaly_ratio: Optional[float] = None
    ) -> pd.DataFrame:
        """
        Generates a complete dataset of flows.
        If anomaly_ratio == 0, returns a 100% benign baseline (ideal for unsupervised fitting).
        """
        n = n_samples if n_samples is not None else self.config.n_samples
        ratio = anomaly_ratio if anomaly_ratio is not None else self.config.anomaly_ratio

        n_anomalies = int(n * ratio)
        n_benign = n - n_anomalies

        base_time = datetime(2026, 9, 20, 8, 0, 0)
        flows = []

        # Generate benign
        for _ in range(n_benign):
            flows.append(self.generate_benign_flow(base_time))

        # Generate anomalies distributed across threat archetypes
        if n_anomalies > 0:
            attack_types = [
                "dns_exfiltration",
                "c2_beacon",
                "port_scan",
                "syn_flood",
                "ssh_brute_force",
            ]
            for i in range(n_anomalies):
                attack = attack_types[i % len(attack_types)]
                flows.append(self.generate_attack_flow(attack, base_time))

        df = pd.DataFrame(flows)
        # Reorder columns to standard schema
        df = df[RAW_FLOW_COLUMNS]
        # Shuffle
        df = df.sample(frac=1.0, random_state=self.config.random_state).reset_index(drop=True)
        return df

    def generate_train_test_split(
        self, n_train: int = 8000, n_test: int = 4000, test_anomaly_ratio: float = 0.15
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Produces:
        - train_df: 100% benign traffic for unsupervised baseline model training.
        - test_df: realistic blend of benign + attacks for evaluation.
        """
        train_df = self.generate_dataset(n_samples=n_train, anomaly_ratio=0.0)
        test_df = self.generate_dataset(n_samples=n_test, anomaly_ratio=test_anomaly_ratio)
        return train_df, test_df
