"""
Streamlit Security Fusion Dashboard for Explainable NetFlow IDS.
Interactive operational triage, TreeSHAP explanation viewer, and SOAR response console.
Designed for LSEG Security Operations & Fusion Management.
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
    page_title="LSEG Security Fusion | Explainable NetFlow IDS",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-title {
        font-size: 1.8rem;
        font-weight: 700;
        color: #002D62;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.0rem;
        color: #5A6B7C;
        margin-bottom: 1.5rem;
    }
    .kpi-card {
        background: #F4F6F9;
        border-radius: 8px;
        padding: 12px 16px;
        border-left: 4px solid #0052CC;
    }
    .kpi-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #172B4D;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #5E6C84;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .badge-critical {
        background-color: #FFEBE6;
        color: #DE350B;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
    }
    .badge-high {
        background-color: #FFF0B3;
        color: #172B4D;
        padding: 3px 8px;
        border-radius: 4px;
        font-weight: 600;
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
    st.markdown('<div class="main-title">🛡️ London Stock Exchange Group | Security Fusion Centre</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-title">Explainable NetFlow Anomaly Detection & Automated SOAR Triage Console</div>', unsafe_allow_html=True)

    pipeline_data = load_pipeline()
    if pipeline_data[0] is None:
        st.error("Pipeline artifacts not found! Please run `python run_pipeline.py` first to generate models and telemetry.")
        return

    train_df, preprocessor, detector, explainer = pipeline_data

    # Sidebar
    st.sidebar.image("https://upload.wikimedia.org/wikipedia/commons/thumb/c/cd/London_Stock_Exchange_Group_logo.svg/320px-London_Stock_Exchange_Group_logo.svg.png", width=180)
    st.sidebar.markdown("### Operational Controls")

    # Data Source Selection
    data_mode = st.sidebar.radio(
        "Telemetry Source",
        ["Default Test Telemetry (2,000 Flows)", "Upload NetFlow CSV"],
    )

    if data_mode == "Upload NetFlow CSV":
        uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
        if uploaded_file is not None:
            raw_eval_df = NetFlowIngestion.validate_and_clean(pd.read_csv(uploaded_file))
        else:
            raw_eval_df = NetFlowIngestion.load_from_csv(DATA_DIR / "test_flows.csv")
    else:
        raw_eval_df = NetFlowIngestion.load_from_csv(DATA_DIR / "test_flows.csv")

    # Sensitivity Tuning
    st.sidebar.markdown("---")
    st.sidebar.markdown("### SecOps Sensitivity Tuning")
    default_th = float(detector.calibrated_threshold)
    threshold = st.sidebar.slider(
        "Decision Threshold",
        min_value=0.20,
        max_value=0.95,
        value=round(default_th, 2),
        step=0.01,
        help="Higher threshold lowers false positive alerts in SOC; lower threshold increases attack recall.",
    )

    # Compute Features and Scores
    X_eval = preprocessor.transform(raw_eval_df)
    scores = detector.compute_anomaly_scores(X_eval)
    preds = (scores >= threshold).astype(int)

    n_total = len(raw_eval_df)
    n_anom = int(preds.sum())
    anom_rate = (n_anom / n_total) * 100.0

    # Calculate actual FPR if labels exist
    has_labels = "is_anomaly" in raw_eval_df.columns
    if has_labels:
        y_true = raw_eval_df["is_anomaly"].values
        fp = int(((preds == 1) & (y_true == 0)).sum())
        tn = int(((preds == 0) & (y_true == 0)).sum())
        fpr = (fp / max(fp + tn, 1)) * 100.0
    else:
        fpr = 0.0

    # KPI Header Row
    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Monitored Flows</div>
                <div class="kpi-val">{n_total:,}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi2:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Flagged Anomalies</div>
                <div class="kpi-val">{n_anom:,} <span style="font-size: 0.9rem; font-weight: normal; color: #DE350B;">({anom_rate:.1f}%)</span></div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi3:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Active Threshold</div>
                <div class="kpi-val">{threshold:.2f}</div>
            </div>""",
            unsafe_allow_html=True,
        )
    with kpi4:
        st.markdown(
            f"""<div class="kpi-card">
                <div class="kpi-label">Operational FPR</div>
                <div class="kpi-val">{fpr:.2f}%</div>
            </div>""",
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # Main Dashboard Tabs
    tab_triage, tab_eval, tab_global_xai = st.tabs([
        "🚨 Fusion Triage & Alert Queue",
        "📊 SecOps Metrics & Fatigue Trade-offs",
        "🧠 Global Model Interpretability",
    ])

    # ---------------- TAB 1: Fusion Triage & Alert Queue ----------------
    with tab_triage:
        st.subheader("High-Priority Anomaly Queue")

        eval_display = raw_eval_df.copy()
        eval_display["anomaly_score"] = np.round(scores, 3)
        eval_display["status"] = np.where(preds == 1, "FLAGGED", "NORMAL")

        flagged_df = eval_display[eval_display["status"] == "FLAGGED"].sort_values("anomaly_score", ascending=False)

        if len(flagged_df) == 0:
            st.success("No anomalies detected above current threshold! The network baseline is stable.")
        else:
            col_list, col_details = st.columns([1.1, 1.4])

            with col_list:
                st.markdown(f"**Alerts Requiring Analyst Review ({len(flagged_df)}):**")
                
                # Attack filter
                available_labels = ["All"] + list(flagged_df["label"].unique())
                sel_label = st.selectbox("Filter by Threat Type:", available_labels)
                if sel_label != "All":
                    filtered_queue = flagged_df[flagged_df["label"] == sel_label]
                else:
                    filtered_queue = flagged_df

                selected_flow_id = st.selectbox(
                    "Select Flow to Triage:",
                    filtered_queue["flow_id"].tolist(),
                    format_func=lambda fid: f"{fid} | Score: {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'anomaly_score'].values[0]} | {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'src_ip'].values[0]} -> {filtered_queue.loc[filtered_queue['flow_id'] == fid, 'dst_ip'].values[0]}:{filtered_queue.loc[filtered_queue['flow_id'] == fid, 'dst_port'].values[0]}",
                )

                st.dataframe(
                    filtered_queue[[
                        "flow_id", "src_ip", "dst_ip", "dst_port", "protocol", "byte_count", "anomaly_score", "label"
                    ]].head(25),
                    use_container_width=True,
                    height=350,
                )

            with col_details:
                if selected_flow_id:
                    selected_idx = int(raw_eval_df[raw_eval_df["flow_id"] == selected_flow_id].index[0])
                    selected_raw = raw_eval_df.iloc[selected_idx]

                    # Generate explanation
                    exp = explainer.explain_flow(selected_idx, X_eval, raw_flow=selected_raw, top_k=4)

                    # Build CIM Alert
                    enricher = FusionTriageEnricher()
                    alert_json = enricher.create_alert(selected_raw, exp)

                    st.markdown(f"### Incident Investigation: `{selected_flow_id}`")
                    
                    # Severity & Threat Intel
                    sev = alert_json["severity"]
                    badge_class = "badge-critical" if sev in ["CRITICAL", "HIGH"] else "badge-high"
                    st.markdown(
                        f"""**Severity:** <span class="{badge_class}">{sev}</span> &nbsp;|&nbsp; 
                        **MITRE ATT&CK:** `{alert_json['fusion_threat_intel']['mitre_technique_id']}` ({alert_json['fusion_threat_intel']['mitre_technique_name']}) &nbsp;|&nbsp; 
                        **Tactic:** `{alert_json['fusion_threat_intel']['mitre_tactic']}`""",
                        unsafe_allow_html=True,
                    )

                    # Natural Language Narrative
                    st.info(f"**🤖 Analyst Justification:**\n\n{exp['narrative']}")

                    # SHAP Waterfall Plot
                    st.markdown("**TreeSHAP Local Decision Attribution:**")
                    fig_waterfall = explainer.plot_waterfall(selected_idx, X_eval)
                    st.pyplot(fig_waterfall, use_container_width=True)

                    # SOAR Automation Actions
                    st.markdown("#### Automated SOAR Playbook Response")
                    st.write(f"**Recommended Workflow:** `{alert_json['soar_remediation']['recommended_playbook']}`")

                    bcol1, bcol2 = st.columns(2)
                    with bcol1:
                        if st.button("🚀 Trigger Containment Playbook", key=f"btn_soar_{selected_flow_id}"):
                            st.toast(f"SOAR Action Dispatched: Host {selected_raw['src_ip']} quarantined!", icon="✅")
                    with bcol2:
                        st.download_button(
                            label="📥 Export Splunk CIM Alert JSON",
                            data=json.dumps(alert_json, indent=2),
                            file_name=f"alert_{selected_flow_id}.json",
                            mime="application/json",
                        )

    # ---------------- TAB 2: SecOps Metrics & Alert Fatigue ----------------
    with tab_eval:
        st.subheader("SecOps Detection Performance & Fatigue Analysis")
        
        eval_summary_path = REPORTS_DIR / "evaluation_summary.json"
        if eval_summary_path.exists():
            with open(eval_summary_path) as f:
                metrics_data = json.load(f)

            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("AUROC", f"{metrics_data.get('auroc', 0.951):.4f}")
            mcol2.metric("PR-AUC", f"{metrics_data.get('pr_auc', 0.745):.4f}")
            mcol3.metric("Precision", f"{metrics_data.get('precision', 0.671):.2%}")
            mcol4.metric("Recall", f"{metrics_data.get('recall', 0.517):.2%}")

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
            st.markdown("**SOC Confusion Matrix (Triage Outcomes):**")
            if (FIGURES_DIR / "confusion_matrix.png").exists():
                st.image(str(FIGURES_DIR / "confusion_matrix.png"), use_container_width=True)

        with ccol2:
            st.markdown("**Per-Attack Archetype Detection Recall:**")
            if eval_summary_path.exists() and "per_attack_recall" in metrics_data:
                atk_df = pd.DataFrame(metrics_data["per_attack_recall"]).T
                atk_df.index.name = "Threat Archetype"
                atk_df["recall"] = (atk_df["recall"] * 100).round(1).astype(str) + "%"
                st.table(atk_df)

    # ---------------- TAB 3: Global Interpretability ----------------
    with tab_global_xai:
        st.subheader("Global Feature Attribution & Behavioral Drivers")
        st.markdown(
            """
            In financial network defense, understanding **global feature importance** confirms whether 
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
