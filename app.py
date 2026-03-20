"""
app.py — EarlyMind Streamlit UI
Run: /opt/anaconda3/envs/infant_id/bin/streamlit run app.py
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import torch

# ─────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────
st.set_page_config(
    layout="wide",
    page_title="EarlyMind 🧠",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from src.config import cfg
from src.utils.age_norms import compute_corrected_age, dq_to_label


# ─────────────────────────────────────────────────────────────────
# Sidebar navigation
# ─────────────────────────────────────────────────────────────────
st.sidebar.image("https://raw.githubusercontent.com/Rickykapoor/earlyMind/main/docs/logo.png",
                 use_column_width=True) if False else st.sidebar.markdown("# 🧠 EarlyMind")
st.sidebar.markdown("**Multimodal Infant ID Risk Detection**")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "📈 Training Monitor", "🔍 Predict Infant"],
    index=0,
)

# ─────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────

@st.cache_resource
def load_fusion_model():
    """Load trained fusion model from checkpoint."""
    ckpt = cfg.paths.checkpoints / "fusion_model.pt"
    if not ckpt.exists():
        return None
    from src.models.fusion_model import build_fusion_model
    n_hpo = cfg.model.hpo_n_features
    model = build_fusion_model(n_hpo=n_hpo)
    model.load_state_dict(torch.load(str(ckpt), map_location="cpu"))
    model.eval()
    return model


def dq_badge(dq: float) -> str:
    label = cfg.dq_label(dq)
    colors = {
        "Typical":         "#22c55e",
        "Borderline":      "#f59e0b",
        "Mild ID Risk":    "#f97316",
        "Moderate ID Risk":"#ef4444",
        "Severe ID Risk":  "#dc2626",
        "Profound ID Risk":"#991b1b",
    }
    color = colors.get(label, "#6b7280")
    return f'<span style="background:{color};color:white;padding:3px 8px;border-radius:6px;font-size:0.85em">{label}</span>'


def gauge_chart(prob_pct: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=prob_pct,
        title={"text": "ID Risk Probability (%)"},
        gauge={
            "axis": {"range": [0, 100]},
            "bar":  {"color": "#ef4444" if prob_pct >= 50 else "#22c55e"},
            "steps": [
                {"range": [0, 30],   "color": "#dcfce7"},
                {"range": [30, 60],  "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": 50},
        },
        number={"suffix": "%"},
    ))
    fig.update_layout(height=250, margin=dict(l=20, r=20, t=40, b=20))
    return fig


# ═══════════════════════════════════════════════════════════════════
# PAGE 1: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════

if page == "📊 Data Overview":
    st.title("📊 Dataset Overview")

    tab_eeg, tab_mri, tab_hpo = st.tabs(["🧠 EEG", "🫁 MRI", "🧬 HPO Phenotype"])

    # ── EEG ──────────────────────────────────────────────────────
    with tab_eeg:
        st.subheader("Helsinki Neonatal EEG Dataset")
        st.markdown("**3 subjects** | 19-channel neonatal EEG | 256Hz | EDF format")

        clin_path = cfg.paths.eeg_raw / "clinical_information.csv"
        if clin_path.exists():
            df = pd.read_csv(clin_path)
            st.dataframe(df.style.highlight_max(axis=0), use_container_width=True)
        else:
            st.warning("clinical_information.csv not found. Run DVC pull first.")

        # Sample waveform
        st.markdown("#### Sample EEG Waveform (Subject 1, first 10 seconds)")
        edf_path = cfg.paths.eeg_raw / "1.edf"
        if edf_path.exists():
            try:
                import mne
                mne.set_log_level("WARNING")
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                sfreq = raw.info["sfreq"]
                data  = raw.get_data()[:3, :int(10 * sfreq)]  # first 3 channels, 10s
                t = np.linspace(0, 10, data.shape[1])
                ch_names = raw.ch_names[:3]
                fig = go.Figure()
                for i, ch in enumerate(ch_names):
                    fig.add_trace(go.Scatter(
                        x=t, y=data[i] * 1e6,   # → µV
                        name=ch, mode="lines",
                        line=dict(width=0.9),
                    ))
                fig.update_layout(
                    xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
                    title="Neonatal EEG — First 3 Channels",
                    height=350, legend=dict(x=0.01, y=0.99),
                )
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load EDF: {e}")
        else:
            st.info("EDF file not available locally. Use DVC pull.")

        # Feature distributions
        proc_dir = cfg.paths.eeg_processed
        feat_files = list(proc_dir.glob("*_features.npy"))
        if feat_files:
            st.markdown("#### Extracted Feature Distributions")
            feat_names = ["Delta", "Theta", "Alpha", "Beta", "Total Power",
                          "BSR", "IBI Mean", "IBI Std", "SEF95", "Amp Mean", "Amp Std"]
            all_f = np.stack([np.load(f) for f in feat_files])
            sids  = [f.stem.replace("_features", "") for f in feat_files]
            df_feat = pd.DataFrame(all_f, index=sids, columns=feat_names)
            st.dataframe(df_feat.round(4), use_container_width=True)

    # ── MRI ──────────────────────────────────────────────────────
    with tab_mri:
        st.subheader("Baby Open Brains (OpenNeuro ds004797)")
        st.markdown("**10 subjects** | T1w + T2w | BIDS format | 0–36 months")

        tsv_path = cfg.paths.mri_raw / "participants.tsv"
        if tsv_path.exists():
            df_part = pd.read_csv(tsv_path, sep="\t")
            st.dataframe(df_part, use_container_width=True)

            # Age histogram
            age_col = None
            for c in df_part.columns:
                if "age" in c.lower():
                    age_col = c
                    break
            if age_col:
                ages = pd.to_numeric(df_part[age_col], errors="coerce").dropna()
                if ages.max() < 10:
                    ages = ages * 12  # years → months
                fig_age = px.histogram(ages, nbins=10,
                                       title="Age Distribution (months)",
                                       labels={"value": "Age (months)"})
                st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.warning("participants.tsv not found.")

        # MRI slices for sub-01
        sub01_path = cfg.paths.mri_processed / "sub-01.npy"
        if sub01_path.exists():
            slices = np.load(str(sub01_path))
            st.markdown("#### Axial · Coronal · Sagittal slices — sub-01")
            cols = st.columns(3)
            titles = ["Axial", "Coronal", "Sagittal"]
            for col, slc, title in zip(cols, slices, titles):
                col.image((slc * 255).astype(np.uint8),
                          caption=title, clamp=True, use_column_width=True)
        else:
            st.info("Run notebook 02_mri_preprocess.ipynb to generate slices.")

    # ── HPO ──────────────────────────────────────────────────────
    with tab_hpo:
        st.subheader("Human Phenotype Ontology (HPO) Annotations")

        hpoa_path = cfg.paths.hpo_raw / "phenotype.hpoa"
        if hpoa_path.exists():
            df_hpo = pd.read_csv(hpoa_path, sep="\t")
            df_hpo.columns = [c.lstrip("#").strip() for c in df_hpo.columns]
            col_name = df_hpo.columns[1]
            total_diseases = df_hpo[col_name].nunique()
            from src.utils.label_utils import ID_KEYWORDS
            id_mask = df_hpo[col_name].str.lower().apply(
                lambda n: any(kw in n for kw in ID_KEYWORDS)
            )
            n_id = df_hpo[id_mask][col_name].nunique()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Diseases", total_diseases)
            c2.metric("ID-Relevant Diseases", n_id)
            c3.metric("Total HPO Terms", df_hpo.iloc[:, 3].nunique())

            # Feature matrix stats
            matrix_path = cfg.paths.hpo_processed / "hpo_matrix.npy"
            if matrix_path.exists():
                X = np.load(str(matrix_path))
                y = np.load(str(cfg.paths.hpo_processed / "hpo_labels.npy"))
                st.info(f"HPO Feature Matrix: **{X.shape[0]} diseases × {X.shape[1]} features** | "
                        f"ID-positive: {int(y.sum())} | Non-ID: {int((y==0).sum())}")
        else:
            st.warning("phenotype.hpoa not found.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 2: TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════════

elif page == "📈 Training Monitor":
    st.title("📈 Training Monitor")

    ckpt = cfg.paths.checkpoints / "fusion_model.pt"
    hist_path  = cfg.paths.reports / "training_history.json"
    report_path = cfg.paths.reports / "benchmark_report.md"

    if not ckpt.exists():
        st.info("🏋️ Train the model first using the Colab notebooks. See README.md for instructions.")
    else:
        st.success("✅ Trained model found!")

        # Training history
        if hist_path.exists():
            with open(hist_path) as f:
                hist = json.load(f)

            tab_loss, tab_auc = st.tabs(["Loss Curves", "AUC Curves"])
            with tab_loss:
                fig = go.Figure()
                fig.add_trace(go.Scatter(y=hist["train_loss"], name="Train Loss"))
                fig.add_trace(go.Scatter(y=hist["val_loss"],   name="Val Loss"))
                fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss",
                                  title="Fusion Model Loss During Training")
                st.plotly_chart(fig, use_container_width=True)
            with tab_auc:
                fig2 = go.Figure()
                fig2.add_trace(go.Scatter(y=hist.get("train_auc", []), name="Train AUC"))
                fig2.add_trace(go.Scatter(y=hist.get("val_auc", []),   name="Val AUC"))
                fig2.add_hline(y=0.85, line_dash="dash", line_color="red",
                               annotation_text="Target AUC=0.85")
                fig2.update_layout(xaxis_title="Epoch", yaxis_title="AUC",
                                   title="Fusion Model AUC")
                st.plotly_chart(fig2, use_container_width=True)
        else:
            st.warning("No training history found yet.")

        # Benchmark report
        if report_path.exists():
            with open(report_path) as f:
                report_text = f.read()
            st.markdown("### 📋 Benchmark Report")
            st.markdown(report_text)
        else:
            st.info("Run notebook 06_evaluate.ipynb to generate the benchmark report.")


# ═══════════════════════════════════════════════════════════════════
# PAGE 3: PREDICT INFANT
# ═══════════════════════════════════════════════════════════════════

elif page == "🔍 Predict Infant":
    st.title("🔍 Predict Infant — ID Risk Screening")

    model = load_fusion_model()
    if model is None:
        st.warning("⚠️ No trained model found at `checkpoints/fusion_model.pt`. "
                   "Complete training in Colab first, then `git pull && dvc pull`.")
        st.stop()

    st.success("✅ Model loaded and ready for inference.")
    tab_manual, tab_file = st.tabs(["✍️ Manual Input", "📂 Load from File"])

    # ── Shared inference fn ──────────────────────────────────────

    def run_prediction(eeg_vec=None, mri_slices=None, hpo_vec=None):
        """
        Build batch dict and run fusion model.
        Returns: prob (float 0–1), dq (float), importance (3-vec), confidence (str)
        """
        with torch.no_grad():
            batch = {}
            missing = []

            if eeg_vec is not None:
                eeg_t = torch.tensor(eeg_vec, dtype=torch.float32)
                # pad to (19, 7680)
                C, T = cfg.model.eeg_channels, cfg.model.eeg_timesteps
                if eeg_t.ndim == 1:
                    # tabular EEG features → skip EEG modality (no raw epochs)
                    missing.append("eeg")
                else:
                    if eeg_t.shape[0] < C:
                        pad = torch.zeros(C - eeg_t.shape[0], eeg_t.shape[1])
                        eeg_t = torch.cat([eeg_t, pad], dim=0)
                    eeg_t = eeg_t[:C, :T]
                    batch["eeg"] = eeg_t.unsqueeze(0)
            else:
                missing.append("eeg")

            if mri_slices is not None:
                # (3, 64, 64) → (1, 3, 1, 64, 64)
                mri_t = torch.tensor(mri_slices, dtype=torch.float32).unsqueeze(1).unsqueeze(0)
                batch["mri"] = mri_t
            else:
                missing.append("mri")

            if hpo_vec is not None:
                hpo_t = torch.tensor(hpo_vec, dtype=torch.float32).unsqueeze(0)
                # pad to n_hpo
                n = cfg.model.hpo_n_features
                if hpo_t.shape[1] < n:
                    pad = torch.zeros(1, n - hpo_t.shape[1])
                    hpo_t = torch.cat([hpo_t, pad], dim=1)
                batch["hpo"] = hpo_t[:, :n]
            else:
                missing.append("hpo")

            if len(batch) == 0:
                return None, None, None, None

            out = model(batch, missing_modalities=[missing])
            prob = float(torch.softmax(out["logits"], dim=-1)[0, 1])
            dq   = float(out["severity"][0, 0])
            importance = out["modality_importance"].numpy()

            dist_from_half = abs(prob - 0.5)
            if dist_from_half > 0.3:  confidence = "High"
            elif dist_from_half > 0.15: confidence = "Moderate"
            else:                       confidence = "Low"

            return prob, dq, importance, confidence

    def show_results_card(prob, dq, importance, confidence, mods_available):
        """Render the prediction results card."""
        prob_pct = prob * 100
        st.markdown("---")
        st.subheader("📋 Prediction Results")

        if prob < 0.5:
            st.markdown(
                '<div style="background:#22c55e;color:white;padding:20px;'
                'border-radius:12px;text-align:center;font-size:1.5em">'
                '🟢 TYPICAL DEVELOPMENT</div>', unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="background:#ef4444;color:white;padding:20px;'
                'border-radius:12px;text-align:center;font-size:1.5em">'
                '🔴 ID RISK DETECTED</div>', unsafe_allow_html=True,
            )

        st.markdown("")
        c1, c2, c3 = st.columns(3)
        c1.metric("Risk Probability", f"{prob_pct:.1f}%")
        dq_label = cfg.dq_label(dq)
        c2.metric("DQ Estimate", f"{dq:.0f}")
        c2.markdown(dq_badge(dq), unsafe_allow_html=True)
        c3.metric("Confidence", confidence)

        # Gauge
        st.plotly_chart(gauge_chart(prob_pct), use_container_width=True)

        # Modality importance bar chart
        mod_names = ["EEG", "MRI", "HPO"]
        colors = [
            "#22c55e" if m.lower() in [x.lower() for x in mods_available] else "#d1d5db"
            for m in mod_names
        ]
        fig_imp = go.Figure(go.Bar(
            x=importance, y=mod_names,
            orientation="h",
            marker_color=colors,
            text=[f"{v:.1%}" for v in importance],
            textposition="auto",
        ))
        fig_imp.update_layout(
            title="Which signals drove this prediction",
            xaxis_title="Importance Weight",
            height=220,
            margin=dict(l=50, r=20, t=40, b=20),
        )
        st.plotly_chart(fig_imp, use_container_width=True)

        # Clinical interpretation
        with st.expander("🔬 What this means clinically"):
            if prob < 0.5:
                st.markdown(
                    "This screening result does not indicate elevated ID risk. "
                    "Continue routine developmental surveillance. "
                    "Recommended monitoring: age-appropriate milestone checklist "
                    "(ASQ-3 at 4, 9, 18, 24, 30 months)."
                )
            else:
                st.markdown("""
This screening result suggests further evaluation is recommended.

> **This is NOT a diagnosis — it is a screening flag.**

**Recommended next steps:**
1. Referral to developmental pediatrician
2. Formal Bayley-4 developmental assessment
3. Audiology evaluation (hearing is critical for language development)
4. Ophthalmology evaluation
5. Genetic panel if dysmorphic features present
6. Early Intervention referral (IDEA Part C in US, equivalent programs elsewhere)
7. Brain MRI if not already obtained
                """)

        st.info(
            "⚕️ **EarlyMind is a research screening tool only.** It is not FDA cleared "
            "and does not provide a clinical diagnosis. All results must be interpreted "
            "by qualified healthcare professionals. Developmental assessment of infants "
            "requires in-person evaluation using validated instruments (Bayley-4, Vineland-3, ASQ-3)."
        )

    # ── TAB A: Manual Input ──────────────────────────────────────

    with tab_manual:
        st.markdown("#### Enter infant clinical data below")

        with st.expander("👶 Basic Info", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                age_months = st.slider("Age at assessment (months)", 0.0, 36.0, 6.0, step=0.5, key="age")
                ga_weeks   = st.slider("Gestational age at birth (weeks)", 24, 42, 38, key="ga")
            with col_b:
                nicu_admission = st.checkbox("Admitted to NICU after birth", key="nicu")
                nicu_days = 0
                if nicu_admission:
                    nicu_days = st.number_input("NICU stay (days)", 0, 180, 0, key="nicu_days")
                hearing_screen = st.selectbox("Newborn hearing screen", ["Pass", "Refer", "Not done"], key="hearing")
                vision_screen  = st.selectbox("Vision screen", ["Pass", "Refer", "Not done"], key="vision")

            corrected = compute_corrected_age(age_months, ga_weeks)
            st.caption(f"✏️ Corrected age: **{corrected:.1f} months**")

        # EEG
        with st.expander("🧠 EEG (if available)"):
            eeg_available = st.checkbox("EEG recording available", key="eeg_avail")
            eeg_raw_data  = None
            eeg_feats     = None
            if eeg_available:
                cols_e = st.columns(2)
                with cols_e[0]:
                    bsr       = st.slider("Burst-suppression ratio", 0.0, 1.0, 0.1, key="bsr")
                    ibi_mean  = st.slider("Mean inter-burst interval (s)", 0.0, 60.0, 5.0, key="ibi")
                with cols_e[1]:
                    delta_pw  = st.number_input("Delta band power (µV²/Hz)", value=100.0, key="delta")
                    sef95     = st.slider("SEF95 (Hz)", 0.0, 30.0, 15.0, key="sef")

                edf_up = st.file_uploader("Or upload EDF file", type=["edf"], key="edf_up")
                if edf_up is not None:
                    import tempfile, mne
                    with tempfile.NamedTemporaryFile(suffix=".edf", delete=False) as tmp:
                        tmp.write(edf_up.read())
                        tmp_path = tmp.name
                    try:
                        from src.data.eeg_loader import load_edf, preprocess_raw, epoch_raw
                        raw = load_edf(tmp_path)
                        raw = preprocess_raw(raw)
                        eeg_raw_data = epoch_raw(raw)
                        st.success(f"EDF loaded: {eeg_raw_data.shape[0]} epochs")
                    except Exception as e:
                        st.error(f"EDF error: {e}")
                    os.unlink(tmp_path)

        # MRI
        with st.expander("🫁 MRI (if available)"):
            mri_available = st.checkbox("Brain MRI available", key="mri_avail")
            mri_slices_in = None
            if mri_available:
                cols_m = st.columns(2)
                with cols_m[0]:
                    myelin  = st.selectbox("Myelination status",
                                           ["Normal for age", "Mildly delayed",
                                            "Moderately delayed", "Severely delayed"], key="myelin")
                    cc_z    = st.slider("Corpus callosum z-score", -4.0, 2.0, 0.0, key="cc_z")
                with cols_m[1]:
                    bvol_z  = st.slider("Brain volume z-score", -4.0, 2.0, 0.0, key="bvol")

                nii_up = st.file_uploader("Or upload NIfTI file", type=["gz", "nii"], key="nii_up")
                if nii_up is not None:
                    import tempfile
                    suffix = ".nii.gz" if nii_up.name.endswith(".gz") else ".nii"
                    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
                        tmp.write(nii_up.read())
                        tmp_path = tmp.name
                    try:
                        from src.data.mri_loader import load_nifti, extract_slices
                        vol = load_nifti(tmp_path)
                        mri_slices_in = extract_slices(vol)
                        st.success(f"NIfTI loaded: slices shape {mri_slices_in.shape}")
                    except Exception as e:
                        st.error(f"NIfTI error: {e}")
                    os.unlink(tmp_path)

        # HPO
        with st.expander("🧬 HPO Phenotype (if assessed)"):
            hpo_available = st.checkbox("HPO phenotype assessment available", key="hpo_avail")
            hpo_vec_in    = None
            if hpo_available:
                HPO_OPTIONS = [
                    "HP:0000252 — Microcephaly (small head circumference)",
                    "HP:0000316 — Hypertelorism (wide-set eyes)",
                    "HP:0000431 — Wide nasal bridge",
                    "HP:0000322 — Short philtrum",
                    "HP:0001263 — Global developmental delay",
                    "HP:0001249 — Intellectual disability",
                    "HP:0000750 — Delayed speech and language development",
                    "HP:0001290 — Hypotonia (low muscle tone)",
                    "HP:0001256 — Mild intellectual disability",
                    "HP:0002342 — Intellectual disability, moderate",
                ]
                selected_hpo = st.multiselect(
                    "Select observed HPO terms:", HPO_OPTIONS, key="hpo_sel"
                )
                if selected_hpo:
                    # Build simple binary vector (load feature names if available)
                    fn_path = cfg.paths.hpo_processed / "hpo_feature_names.npy"
                    if fn_path.exists():
                        feat_names = np.load(str(fn_path), allow_pickle=True)
                        n = len(feat_names)
                        hpo_vec_in = np.zeros(n, dtype=np.float32)
                        for term_str in selected_hpo:
                            term_id = term_str.split(" — ")[0].strip()
                            matches = np.where(np.char.startswith(feat_names.astype(str), term_id))[0]
                            if len(matches) > 0:
                                hpo_vec_in[matches[0]] = 1.0
                    else:
                        n = cfg.model.hpo_n_features
                        hpo_vec_in = np.zeros(n, dtype=np.float32)

        # Predict Button
        if st.button("🔍 Run ID Risk Assessment", type="primary", use_container_width=True):
            eeg_in = eeg_raw_data[0] if eeg_raw_data is not None else None
            if not eeg_available:
                eeg_in = None

            mri_in = mri_slices_in if mri_available else None
            hpo_in = hpo_vec_in if hpo_available else None

            available_mods = []
            if eeg_in is not None: available_mods.append("EEG")
            if mri_in is not None: available_mods.append("MRI")
            if hpo_in is not None: available_mods.append("HPO")

            if len(available_mods) == 0:
                st.error("Please enable at least one modality (EEG, MRI, or HPO) to run prediction.")
            else:
                with st.spinner("Running inference …"):
                    prob, dq, importance, confidence = run_prediction(eeg_in, mri_in, hpo_in)
                if prob is None:
                    st.error("Inference failed. Check that at least one modality has data.")
                else:
                    show_results_card(prob, dq, importance, confidence, available_mods)

    # ── TAB B: Load from File ────────────────────────────────────

    with tab_file:
        st.markdown("#### Upload a subject JSON file")
        st.code("""{
  "age_months": 6.0,
  "ga_weeks": 38,
  "eeg_epoch_path": "datasets/processed/eeg/1_epochs.npy",
  "mri_slice_path": "datasets/processed/mri/sub-01.npy",
  "hpo_features_path": "datasets/processed/facial/hpo_matrix.npy",
  "hpo_index": 0
}""", language="json")

        json_file = st.file_uploader("Upload subject JSON", type=["json"], key="json_up")
        if json_file is not None:
            data = json.load(json_file)

            eeg_in = mri_in = hpo_in = None
            available_mods = []

            if "eeg_epoch_path" in data:
                ep = Path(data["eeg_epoch_path"])
                if ep.exists():
                    epochs = np.load(str(ep))
                    eeg_in = epochs[0] if len(epochs) > 0 else None
                    if eeg_in is not None: available_mods.append("EEG")

            if "mri_slice_path" in data:
                sp = Path(data["mri_slice_path"])
                if sp.exists():
                    mri_in = np.load(str(sp))
                    available_mods.append("MRI")

            if "hpo_features_path" in data:
                hp = Path(data["hpo_features_path"])
                if hp.exists():
                    X = np.load(str(hp))
                    idx = int(data.get("hpo_index", 0))
                    hpo_in = X[idx] if idx < len(X) else X[0]
                    available_mods.append("HPO")

            st.info(f"Loaded modalities: {available_mods}")

            if st.button("🔍 Run Assessment (from file)", type="primary"):
                with st.spinner("Running inference …"):
                    prob, dq, importance, confidence = run_prediction(eeg_in, mri_in, hpo_in)
                if prob is None:
                    st.error("Inference failed.")
                else:
                    show_results_card(prob, dq, importance, confidence, available_mods)
