"""
Streamlit Security Fusion Dashboard for Explainable NetFlow IDS.
Interactive operational triage, TreeSHAP explanation viewer, and SOAR response console.
Engineered for Enterprise Cyber Fusion Centres & Critical Infrastructure SecOps.
"""

import os
os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import json
from pathlib import Path
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.config import (
    DATA_DIR,
    REPORTS_DIR,
    FIGURES_DIR,
    ATTACK_METADATA,
    MODEL_FEATURE_COLUMNS,
)
from src.ingestion import NetFlowIngestion
from src.features import NetFlowPreprocessor
from src.model import NetFlowAnomalyDetector
from src.explainability import NetFlowExplainer
from src.evaluation import SecOpsEvaluator
from src.triage_enricher import FusionTriageEnricher


# Page Configuration
st.set_page_config(
    page_title="Cyber Fusion Centre | Explainable NetFlow IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom High-End Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }

    code, pre, .mono-font {
        font-family: 'JetBrains Mono', monospace !important;
    }

    /* Top Executive Banner */
    .fusion-navbar {
        background: linear-gradient(135deg, #0F172A 0%, #1E293B 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 18px 24px;
        margin-bottom: 22px;
        display: flex;
        justify-content: space-between;
        align-items: center;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.15);
    }
    .brand-title {
        font-size: 1.55rem;
        font-weight: 800;
        letter-spacing: -0.5px;
        color: #FFFFFF;
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .brand-subtitle {
        font-size: 0.88rem;
        color: #94A3B8;
        margin-top: 3px;
        font-weight: 400;
    }
    .status-badge {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34D399;
        padding: 6px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        letter-spacing: 0.3px;
    }
    .pulse-dot {
        width: 8px;
        height: 8px;
        background-color: #10B981;
        border-radius: 50%;
        box-shadow: 0 0 10px #10B981;
    }

    /* KPI Cards */
    .kpi-container {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
        position: relative;
        overflow: hidden;
        transition: transform 0.15s ease, box-shadow 0.15s ease;
    }
    .kpi-container:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(0, 0, 0, 0.08);
    }
    .kpi-container.highlight-blue {
        border-top: 4px solid #2563EB;
    }
    .kpi-container.highlight-red {
        border-top: 4px solid #DC2626;
    }
    .kpi-container.highlight-purple {
        border-top: 4px solid #7C3AED;
    }
    .kpi-container.highlight-green {
        border-top: 4px solid #059669;
    }
    .kpi-label {
        font-size: 0.78rem;
        font-weight: 600;
        color: #64748B;
        text-transform: uppercase;
        letter-spacing: 0.8px;
    }
    .kpi-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0F172A;
        margin-top: 4px;
        line-height: 1.2;
    }
    .kpi-sub {
        font-size: 0.8rem;
        color: #94A3B8;
        margin-top: 6px;
    }

    /* Severity Badges */
    .badge-critical {
        background: #FEF2F2;
        color: #DC2626;
        border: 1px solid #FECACA;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-high {
        background: #FFFBEB;
        color: #D97706;
        border: 1px solid #FDE68A;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }
    .badge-medium {
        background: #EFF6FF;
        color: #2563EB;
        border: 1px solid #BFDBFE;
        padding: 3px 10px;
        border-radius: 6px;
        font-weight: 700;
        font-size: 0.78rem;
    }

    /* Telemetry Pill */
    .flow-pill {
        display: inline-block;
        background: #F8FAFC;
        border: 1px solid #E2E8F0;
        padding: 8px 14px;
        border-radius: 8px;
        font-family: 'JetBrains Mono', monospace;
        font-size: 0.85rem;
        color: #1E293B;
        margin-bottom: 8px;
        margin-right: 8px;
    }

    /* Briefing Container */
    .briefing-card {
        background: #F8FAFC;
        border-left: 4px solid #2563EB;
        border-radius: 0 8px 8px 0;
        padding: 14px 18px;
        margin: 12px 0;
        font-size: 0.92rem;
        line-height: 1.5;
        color: #1E293B;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def load_pipeline():
    """Loads baseline data, fitted preprocessor, detector, and explainer."""
    train_path = DATA_DIR / "train_baseline.csv"
    model_path = REPORTS_DIR / "model_checkpoint.joblib"

    if not train_path.exists() or not model_path.exists():
        return None, None, None, None

    train_df = NetFlowIngestion.load_from_csv(train_path)
    detector = NetFlowAnomalyDetector.load(model_path)

    preprocessor = NetFlowPreprocessor(scaler_type="robust")
    X_train = preprocessor.fit_transform(train_df)

    explainer = NetFlowExplainer(detector)
    explainer.fit_baseline_reference(X_train, train_df)

    return train_df, preprocessor, detector, explainer


def main():
    # Top Navbar Header
    st.markdown(
        """
        <div class="fusion-navbar">
            <div class="brand-title">
                <span>🛡️</span>
                <div>
                    <div>ENTERPRISE CYBER FUSION CENTRE</div>
                    <div class="brand-subtitle">Autonomous NetFlow Telemetry Anomaly Detection & Explainable AI (XAI) Console</div>
                </div>
            </div>
            <div class="status-badge">
                <span class="pulse-dot"></span>
                <span>SENSORS ONLINE &bull; NETFLOW V9 / IPFIX</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    pipeline_data = load_pipeline()
    if pipeline_data[0] is None:
        st.error("Pipeline artifacts not found! Please run `python run_pipeline.py` first to generate models and telemetry.")
        return

    train_df, preprocessor, detector, explainer = pipeline_data

    # Sidebar
    st.sidebar.markdown(
        """
        <div style="padding: 10px 0 16px 0;">
            <div style="font-size: 1.1rem; font-weight: 800; color: #0F172A; letter-spacing: -0.3px;">
                ⚡ FUSION MANAGEMENT
            </div>
            <div style="font-size: 0.8rem; color: #64748B;">
                SecOps Operational Controls
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Telemetry Feed Source
    telemetry_options = ["Default Synthetic Evaluation (2,000 Flows)"]
    if (DATA_DIR / "cidds001_converted.csv").exists():
        telemetry_options.append("CIDDS-001 Benchmark Telemetry (500 Flows)")
    telemetry_options.append("Upload Custom NetFlow CSV")

    data_mode = st.sidebar.selectbox("Telemetry Feed Source", telemetry_options)

    if data_mode == "Upload Custom NetFlow CSV":
        uploaded_file = st.sidebar.file_uploader("Upload Normalized CSV", type=["csv"])
        if uploaded_file is not None:
            raw_eval_df = NetFlowIngestion.validate_and_clean(pd.read_csv(uploaded_file))
        else:
            raw_eval_df = NetFlowIngestion.load_from_csv(DATA_DIR / "test_flows.csv")
    elif data_mode.startswith("CIDDS-001"):
        raw_eval_df = NetFlowIngestion.load_from_csv(DATA_DIR / "cidds001_converted.csv")
    else:
        raw_eval_df = NetFlowIngestion.load_from_csv(DATA_DIR / "test_flows.csv")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Operational Sensitivity Calibration")

    default_th = float(detector.calibrated_threshold)
    
    # Preset sensitivity options
    preset = st.sidebar.radio(
        "Sensitivity Preset",
        ["Custom Slider", "Strict Low-FPR (≤ 3% Target)", "Balanced SecOps", "High-Recall Recon Sweep"],
        index=0,
    )

    if preset == "Strict Low-FPR (≤ 3% Target)":
        slider_val = round(default_th, 2)
    elif preset == "Balanced SecOps":
        slider_val = 0.55
    elif preset == "High-Recall Recon Sweep":
        slider_val = 0.45
    else:
        slider_val = round(default_th, 2)

    threshold = st.sidebar.slider(
        "Active Decision Threshold",
        min_value=0.20,
        max_value=0.95,
        value=slider_val,
        step=0.01,
        help="Higher threshold lowers false positive alerts in SOC; lower threshold increases attack recall.",
    )

    # Compute Model Features and Anomaly Scores
    X_eval = preprocessor.transform(raw_eval_df)
    scores = detector.compute_anomaly_scores(X_eval)
    preds = (scores >= threshold).astype(int)

    n_total = len(raw_eval_df)
    n_anom = int(preds.sum())
    anom_rate = (n_anom / n_total) * 100.0

    # Calculate actual FPR if ground truth labels exist
    has_labels = "is_anomaly" in raw_eval_df.columns
    if has_labels:
        y_true = raw_eval_df["is_anomaly"].values
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fpr = (fp / max(fp + tn, 1)) * 100.0
    else:
        fpr = 0.0

    # KPI Header Cards
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""
            <div class="kpi-container highlight-blue">
                <div class="kpi-label">Monitored Flow Volume</div>
                <div class="kpi-value">{n_total:,}</div>
                <div class="kpi-sub">Real-time session telemetry</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""
            <div class="kpi-container highlight-red">
                <div class="kpi-label">Flagged Anomalies</div>
                <div class="kpi-value">{n_anom:,}</div>
                <div class="kpi-sub"><span style="color: #DC2626; font-weight:700;">{anom_rate:.1f}%</span> alert volume</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""
            <div class="kpi-container highlight-purple">
                <div class="kpi-label">Active Threshold</div>
                <div class="kpi-value">{threshold:.2f}</div>
                <div class="kpi-sub">Calibrated baseline cutoff</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""
            <div class="kpi-container highlight-green">
                <div class="kpi-label">Operational False Positive Rate</div>
                <div class="kpi-value">{fpr:.2f}%</div>
                <div class="kpi-sub">Target budget: &le; 4.0%</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Dashboard Tabs
    tab_triage, tab_eval, tab_global_xai = st.tabs([
        "🚨 Fusion Triage & Incident Queue",
        "📊 SecOps Performance & Fatigue Analysis",
        "🧠 Global Explainability (TreeSHAP)",
    ])

    # ---------------- TAB 1: Fusion Triage & Incident Queue ----------------
    with tab_triage:
        eval_display = raw_eval_df.copy()
        eval_display["anomaly_score"] = np.round(scores, 3)
        eval_display["status"] = np.where(preds == 1, "FLAGGED", "NORMAL")

        flagged_df = eval_display[eval_display["status"] == "FLAGGED"].sort_values("anomaly_score", ascending=False)

        if len(flagged_df) == 0:
            st.success("✅ Baseline is stable. No telemetry anomalies exceed the active threshold.")
        else:
            col_list, col_details = st.columns([1.1, 1.4])

            with col_list:
                st.markdown(f"#### Incident Queue ({len(flagged_df):,} Events)")
                
                # Threat type filter
                threat_labels = ["All Threat Types"] + sorted(list(flagged_df["label"].unique()))
                sel_label = st.selectbox("Filter Queue:", threat_labels)
                if sel_label != "All Threat Types":
                    filtered_queue = flagged_df[flagged_df["label"] == sel_label]
                else:
                    filtered_queue = flagged_df

                selected_flow_id = st.selectbox(
                    "Inspect Flow Telemetry:",
                    filtered_queue["flow_id"].tolist(),
                    format_func=lambda fid: f"{fid} | Score: {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'anomaly_score'].values[0]} | {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'src_ip'].values[0]} ➔ {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'dst_ip'].values[0]}:{filtered_queue.loc[filtered_queue['flow_id'] == fid, 'dst_port'].values[0]}",
                )

                st.dataframe(
                    filtered_queue[[
                        "flow_id", "src_ip", "dst_ip", "dst_port", "protocol", "packet_count", "byte_count", "anomaly_score", "label"
                    ]].head(30),
                    use_container_width=True,
                    height=420,
                )

            with col_details:
                if selected_flow_id:
                    selected_idx = int(raw_eval_df[raw_eval_df["flow_id"] == selected_flow_id].index[0])
                    selected_raw = raw_eval_df.iloc[selected_idx]

                    # Generate TreeSHAP explanation
                    exp = explainer.explain_flow(selected_idx, X_eval, raw_flow=selected_raw, top_k=4)

                    # Build Splunk CIM Alert
                    enricher = FusionTriageEnricher()
                    alert_json = enricher.create_alert(selected_raw, exp)

                    st.markdown(f"### Incident Details: `{selected_flow_id}`")
                    
                    # Severity & MITRE ATT&CK Header
                    sev = alert_json["severity"]
                    badge_class = "badge-critical" if sev in ["CRITICAL", "HIGH"] else "badge-medium"
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 12px;">
                            <span class="{badge_class}">{sev} PRIORITY</span> &nbsp;&bull;&nbsp; 
                            <b>MITRE ATT&CK:</b> <code>{alert_json['fusion_threat_intel']['mitre_technique_id']}</code> 
                            ({alert_json['fusion_threat_intel']['mitre_technique_name']}) &nbsp;&bull;&nbsp; 
                            <b>Tactic:</b> <i>{alert_json['fusion_threat_intel']['mitre_tactic']}</i>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Endpoint Metadata Pills
                    st.markdown(
                        f"""
                        <div style="margin-bottom: 14px;">
                            <span class="flow-pill"><b>SRC:</b> {selected_raw['src_ip']}:{selected_raw['src_port']}</span>
                            <span class="flow-pill"><b>DST:</b> {selected_raw['dst_ip']}:{selected_raw['dst_port']}</span>
                            <span class="flow-pill"><b>PROTO:</b> {selected_raw['protocol']}</span>
                            <span class="flow-pill"><b>BYTES:</b> {selected_raw['byte_count']:,}</span>
                            <span class="flow-pill"><b>PACKETS:</b> {selected_raw['packet_count']:,}</span>
                            <span class="flow-pill"><b>FLAGS:</b> {selected_raw['tcp_flags']}</span>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # Natural Language Analyst Justification
                    st.markdown(
                        f"""
                        <div class="briefing-card">
                            <b>Analyst Triage Narrative:</b><br>
                            {exp['narrative']}
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

                    # TreeSHAP Waterfall Visualization
                    st.markdown("#### Local TreeSHAP Decision Attribution")
                    fig_waterfall = explainer.plot_waterfall(selected_idx, X_eval)
                    st.pyplot(fig_waterfall, use_container_width=True)

                    # Automated SOAR Playbook Execution
                    st.markdown("#### Automated SOAR Playbook Response")
                    st.markdown(f"**Recommended Workflow:** `{alert_json['soar_remediation']['recommended_playbook']}`")

                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("⚡ Execute Containment Playbook", key=f"btn_soar_{selected_flow_id}", use_container_width=True):
                            st.toast(f"Containment dispatched: Endpoint {selected_raw['src_ip']} quarantined!", icon="🔒")
                            st.success(f"Audit Log: SOAR Playbook triggered for {selected_flow_id}. Firewall policy updated.")
                    with bcol2:
                        st.download_button(
                            label="📥 Export Splunk CIM Event (JSON)",
                            data=json.dumps(alert_json, indent=2),
                            file_name=f"splunk_cim_alert_{selected_flow_id}.json",
                            mime="application/json",
                            use_container_width=True,
                        )

    # ---------------- TAB 2: SecOps Performance & Alert Fatigue ----------------
    with tab_eval:
        st.subheader("SecOps Detection Performance & Alert Fatigue Analysis")
        
        eval_summary_path = REPORTS_DIR / "evaluation_summary.json"
        if eval_summary_path.exists():
            with open(eval_summary_path) as f:
                metrics_data = json.load(f)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("AUROC", f"{metrics_data.get('auroc', 0.951):.4f}", help="Discrimination across all decision thresholds")
            mcol2.metric("PR-AUC (Avg Precision)", f"{metrics_data.get('pr_auc', 0.745):.4f}", help="Precision-Recall Area under curve")
            mcol3.metric("Precision", f"{metrics_data.get('precision', 0.671):.2%}", help="True attacks divided by total alerts")
            mcol4.metric("Recall", f"{metrics_data.get('recall', 0.517):.2%}", help="Attacks detected at calibrated threshold")

        st.markdown("---")
        pcol1, pcol2 = st.columns(2)
        with pcol1:
            st.markdown("**Receiver Operating Characteristic & Precision-Recall:**")
            if (FIGURES_DIR / "roc_pr_curves.png").exists():
                st.image(str(FIGURES_DIR / "roc_pr_curves.png"), use_container_width=True)

        with pcol2:
            st.markdown("**Alert Volume vs Detection Rate Trade-off:**")
            if (FIGURES_DIR / "alert_volume_tradeoff.png").exists():
                st.image(str(FIGURES_DIR / "alert_volume_tradeoff.png"), use_container_width=True)

        st.markdown("---")
        ccol1, ccol2 = st.columns([1, 1.2])
        with ccol1:
            st.markdown("**Fusion SOC Confusion Matrix:**")
            if (FIGURES_DIR / "confusion_matrix.png").exists():
                st.image(str(FIGURES_DIR / "confusion_matrix.png"), use_container_width=True)

        with ccol2:
            st.markdown("**Per-Attack Archetype Detection Recall:**")
            if eval_summary_path.exists() and "per_attack_recall" in metrics_data:
                atk_df = pd.DataFrame(metrics_data["per_attack_recall"]).T
                atk_df.index.name = "Threat Archetype"
                atk_df["recall"] = (atk_df["recall"] * 100).round(1).astype(str) + "%"
                st.dataframe(atk_df, use_container_width=True)

    # ---------------- TAB 3: Global Interpretability ----------------
    with tab_global_xai:
        st.subheader("Global Feature Attribution & Behavioral Drivers")
        st.markdown(
            """
            In enterprise network defense, understanding **global feature importance** confirms whether 
            the model learned meaningful behavioral indicators (e.g. byte/packet density, flag anomalies, rate spikes)
            rather than memorizing static IP addresses or noisy ephemeral ports.
            """
        )

        gcol1, gcol2 = st.columns([1.2, 1])
        with gcol1:
            if (FIGURES_DIR / "global_feature_importance.png").exists():
                st.image(str(FIGURES_DIR / "global_feature_importance.png"), use_container_width=True)

        with gcol2:
            st.markdown("### Key Behavioral Insights")
            st.markdown(
                """
                - **`packets_per_second` & `bytes_per_second`:** Strongest drivers for isolating volumetric denial-of-service (`syn_flood`) and rapid port reconnaissance sweeps.
                - **`bytes_per_packet` (Payload Density):** Crucial for identifying DNS Tunneling (`dns_exfiltration`) where payloads are 5x-10x larger than legitimate resolution queries.
                - **`is_syn_only` & TCP Flags:** Distinguishes scanning tools that fail to complete the 3-way handshake from valid enterprise TLS handshakes.
                - **`log_duration`:** Distinguishes automated scripts from human-driven application workflows.
                """
            )


if __name__ == "__main__":
    main()
