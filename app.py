"""
app.py — EarlyMind Streamlit UI
Refactored to call FastAPI backend instead of loading the model directly.
Run locally: streamlit run app.py
Run with Docker: docker compose up (starts both FastAPI + Streamlit)
HF Spaces: This file is the entrypoint served by the container.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import requests
import streamlit as st

API_BASE = os.environ.get("EARLYMIND_API_URL", "http://localhost:8000")

st.set_page_config(
    layout="wide",
    page_title="EarlyMind",
    page_icon="🧠",
    initial_sidebar_state="expanded",
)

sys.path.insert(0, str(Path(__file__).resolve().parent))


def _api_get(path: str, default=None):
    try:
        r = requests.get(f"{API_BASE}{path}", timeout=5)
        r.raise_for_status()
        return r.json()
    except Exception:
        return default


def _api_post(path: str, json=None, default=None):
    try:
        r = requests.post(f"{API_BASE}{path}", json=json, timeout=30)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            err = e.response.json()
            st.error(f"API error: {err.get('detail', str(e))}")
        except Exception:
            st.error(f"API error: {e}")
        return default
    except Exception as e:
        st.error(f"Connection error: {e}")
        return default


def _api_upload_file(path: str, file_name: str, file_bytes: bytes, default=None):
    try:
        files = {"file": (file_name, file_bytes, "application/octet-stream")}
        r = requests.post(f"{API_BASE}{path}", files=files, timeout=60)
        r.raise_for_status()
        return r.json()
    except requests.HTTPError as e:
        try:
            err = e.response.json()
            st.error(f"Upload error: {err.get('detail', str(e))}")
        except Exception:
            st.error(f"Upload error: {e}")
        return default
    except Exception as e:
        st.error(f"Connection error: {e}")
        return default


def _health_check() -> tuple[bool, bool, bool]:
    info = _api_get("/health")
    if info is None:
        return False, False, False
    model_ok = info.get("model_loaded", False)
    api_ok = info.get("status") != "unhealthy"
    return True, api_ok, model_ok


def _model_info():
    return _api_get("/model/info")


st.session_state.setdefault("_api_online", False)


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("# 🧠 EarlyMind")
st.sidebar.markdown("**Multimodal Infant ID Risk Detection**")

api_ok, _, model_ok = _health_check()
st.session_state["_api_online"] = api_ok

if not api_ok:
    st.sidebar.error("API offline — ensure FastAPI is running on :8000")
elif not model_ok:
    st.sidebar.warning("Model not loaded — prediction disabled")
else:
    st.sidebar.success("✅ API + Model online")

st.sidebar.markdown("---")
st.sidebar.markdown(f"**API:** `{API_BASE}`")

page = st.sidebar.radio(
    "Navigate",
    ["📊 Data Overview", "📈 Training Monitor", "🔍 Predict Infant"],
    index=0,
)


# ── Helpers ────────────────────────────────────────────────────────────────

def dq_badge(dq: float) -> str:
    label = _get_dq_label(dq)
    colors = {
        "Typical":           "#22c55e",
        "Borderline":        "#f59e0b",
        "Mild ID Risk":      "#f97316",
        "Moderate ID Risk":  "#ef4444",
        "Severe ID Risk":    "#dc2626",
        "Profound ID Risk":  "#991b1b",
    }
    color = colors.get(label, "#6b7280")
    return f'<span style="background:{color};color:white;padding:3px 8px;border-radius:6px;font-size:0.85em">{label}</span>'


def _get_dq_label(dq: float) -> str:
    info = _model_info()
    bands = info.get("severity_bands", []) if info else []
    if not bands:
        bands = [
            {"label": "Typical", "range": [85, 100]},
            {"label": "Borderline", "range": [70, 85]},
            {"label": "Mild ID Risk", "range": [55, 70]},
            {"label": "Moderate ID Risk", "range": [35, 55]},
            {"label": "Severe ID Risk", "range": [20, 35]},
            {"label": "Profound ID Risk", "range": [0, 20]},
        ]
    for b in bands:
        lo, hi = b["range"]
        if lo <= dq <= hi:
            return b["label"]
    return "Profound ID Risk"


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


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 1: DATA OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════

if page == "📊 Data Overview":
    st.title("📊 Dataset Overview")

    tab_eeg, tab_mri, tab_hpo = st.tabs(["🧠 EEG", "🫁 MRI", "🧬 HPO Phenotype"])

    with tab_eeg:
        st.subheader("Helsinki Neonatal EEG Dataset")
        st.markdown("**3 subjects** | 19-channel neonatal EEG | 256Hz | EDF format")

        clin_path = Path("datasets/eeg/helsinki_neonatal/clinical_information.csv")
        if clin_path.exists():
            df = pd.read_csv(clin_path)
            st.dataframe(df, use_container_width=True)
        else:
            st.warning("clinical_information.csv not found. Run DVC pull first.")

        edf_path = Path("datasets/eeg/helsinki_neonatal/1.edf")
        if edf_path.exists():
            try:
                import mne
                mne.set_log_level("WARNING")
                raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
                sfreq = raw.info["sfreq"]
                data  = raw.get_data()[:3, :int(10 * sfreq)]
                t = np.linspace(0, 10, data.shape[1])
                ch_names = raw.ch_names[:3]
                fig = go.Figure()
                for i, ch in enumerate(ch_names):
                    fig.add_trace(go.Scatter(x=t, y=data[i] * 1e6, name=ch, mode="lines", line=dict(width=0.9)))
                fig.update_layout(xaxis_title="Time (s)", yaxis_title="Amplitude (µV)",
                                  title="Neonatal EEG — First 3 Channels",
                                  height=350, legend=dict(x=0.01, y=0.99))
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:
                st.error(f"Could not load EDF: {e}")
        else:
            st.info("EDF file not available locally. Use DVC pull.")

    with tab_mri:
        st.subheader("Baby Open Brains (OpenNeuro ds004797)")
        st.markdown("**10 subjects** | T1w + T2w | BIDS format | 0–36 months")

        tsv_path = Path("datasets/mri/baby_open_brains/participants.tsv")
        if tsv_path.exists():
            df_part = pd.read_csv(tsv_path, sep="\t")
            st.dataframe(df_part, use_container_width=True)
            age_col = next((c for c in df_part.columns if "age" in c.lower()), None)
            if age_col:
                ages = pd.to_numeric(df_part[age_col], errors="coerce").dropna()
                if ages.max() < 10:
                    ages = ages * 12
                fig_age = px.histogram(ages, nbins=10, title="Age Distribution (months)",
                                        labels={"value": "Age (months)"})
                st.plotly_chart(fig_age, use_container_width=True)
        else:
            st.warning("participants.tsv not found.")

        sub01_path = Path("datasets/processed/mri/sub-01.npy")
        if sub01_path.exists():
            slices = np.load(str(sub01_path))
            st.markdown("#### Axial · Coronal · Sagittal slices — sub-01")
            cols = st.columns(3)
            titles = ["Axial", "Coronal", "Sagittal"]
            for col, slc, title in zip(cols, slices, titles):
                col.image((slc * 255).astype(np.uint8), caption=title, clamp=True, use_column_width=True)
        else:
            st.info("Run notebook 02_mri_preprocess.ipynb to generate slices.")

    with tab_hpo:
        st.subheader("Human Phenotype Ontology (HPO) Annotations")

        hpoa_path = Path("datasets/facial/hpo/phenotype.hpoa")
        if hpoa_path.exists():
            df_hpo = pd.read_csv(hpoa_path, sep="\t", skiprows=4, low_memory=False)
            df_hpo.columns = [c.lstrip("#").strip() for c in df_hpo.columns]
            col_name = df_hpo.columns[1]
            total_diseases = df_hpo[col_name].nunique()

            ID_KEYWORDS = ["intellectual disability", "mental retardation", "developmental delay",
                           "global developmental delay", "cognitive impairment"]
            id_mask = df_hpo[col_name].str.lower().apply(
                lambda n: any(kw in n for kw in ID_KEYWORDS)
            )
            n_id = df_hpo[id_mask][col_name].nunique()

            c1, c2, c3 = st.columns(3)
            c1.metric("Total Diseases", total_diseases)
            c2.metric("ID-Relevant Diseases", n_id)
            c3.metric("Total HPO Terms", df_hpo.iloc[:, 3].nunique())

            matrix_path = Path("datasets/processed/facial/hpo_matrix.npy")
            if matrix_path.exists():
                X = np.load(str(matrix_path))
                y_path = Path("datasets/processed/facial/hpo_labels.npy")
                y = np.load(str(y_path)) if y_path.exists() else np.zeros(len(X))
                st.info(f"HPO Feature Matrix: **{X.shape[0]} diseases × {X.shape[1]} features** | "
                        f"ID-positive: {int(y.sum())} | Non-ID: {int((y==0).sum())}")
        else:
            st.warning("phenotype.hpoa not found.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 2: TRAINING MONITOR
# ═══════════════════════════════════════════════════════════════════════════

elif page == "📈 Training Monitor":
    st.title("📈 Training Monitor")

    info = _model_info()
    hist_path = Path("reports/training_history.json")
    report_path = Path("reports/benchmark_report.md")

    if info is None:
        st.error("Cannot connect to FastAPI. Is the backend running?")
        st.info(f"API URL: {API_BASE}")
    else:
        st.success("✅ API connected")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Model", info.get("model_name", "Unknown"))
        col2.metric("Version", info.get("version", "N/A"))
        col3.metric("Embed Dim", info.get("embed_dim", "N/A"))
        col4.metric("HPO Features", info.get("hpo_n_features", "N/A"))

        st.markdown("#### DQ Severity Bands")
        bands = info.get("severity_bands", [])
        if bands:
            rows = []
            for b in bands:
                lo, hi = b.get("range", [0, 0])
                rows.append({"Label": b["label"], "DQ Range": f"{lo}–{hi}"})
            st.table(pd.DataFrame(rows))

    st.markdown("---")

    if hist_path.exists():
        with open(hist_path) as f:
            hist = json.load(f)

        tab_loss, tab_auc = st.tabs(["Loss Curves", "AUC Curves"])
        with tab_loss:
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=hist.get("train_loss", []), name="Train Loss"))
            fig.add_trace(go.Scatter(y=hist.get("val_loss", []), name="Val Loss"))
            fig.update_layout(xaxis_title="Epoch", yaxis_title="Loss", title="Fusion Model Loss")
            st.plotly_chart(fig, use_container_width=True)
        with tab_auc:
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(y=hist.get("train_auc", []), name="Train AUC"))
            fig2.add_trace(go.Scatter(y=hist.get("val_auc", []), name="Val AUC"))
            fig2.add_hline(y=0.85, line_dash="dash", line_color="red", annotation_text="Target AUC=0.85")
            fig2.update_layout(xaxis_title="Epoch", yaxis_title="AUC", title="Fusion Model AUC")
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("No training history found. Train the model and run dvc pull.")

    if report_path.exists():
        st.markdown("#### 📋 Benchmark Report")
        with open(report_path) as f:
            st.markdown(f.read())
    else:
        st.info("Run notebook 06_evaluate.ipynb to generate the benchmark report.")


# ═══════════════════════════════════════════════════════════════════════════
# PAGE 3: PREDICT INFANT
# ═══════════════════════════════════════════════════════════════════════════

elif page == "🔍 Predict Infant":
    st.title("🔍 Predict Infant — ID Risk Screening")

    if not st.session_state["_api_online"]:
        st.error("FastAPI backend is offline. Start it with `docker compose up` or run `uvicorn api.main:app --port 8000`.")
        st.stop()

    model_ok = (_api_get("/health") or {}).get("model_loaded", False)
    if not model_ok:
        st.warning("⚠️ Model not loaded. Complete training in Colab, then `git pull && dvc pull`.")
        st.stop()

    tab_manual, tab_file = st.tabs(["✍️ Manual Input", "📂 Load from File"])

    # ── Sample cases ──────────────────────────────────────────────────────

    SAMPLE_CASES = {
        "🟢 Case 1 — Healthy Full-Term Infant": {
            "desc": "38-week healthy infant. Passed all screens. No concerns at 6 months.",
            "age": 6.0, "ga": 38, "nicu": False, "nicu_days": 0,
            "hearing": "Pass", "vision": "Pass",
            "eeg_avail": False, "mri_avail": False, "hpo_avail": True,
            "bsr": 0.05, "ibi": 3.0, "delta": 100.0, "sef": 18.0,
            "myelin": "Normal for age", "cc_z": 0.0, "bvol": 0.0, "hpo_sel": [],
        },
        "🟡 Case 2 — Moderate Prematurity, Mild Risk": {
            "desc": "32-week preterm, 14-day NICU. Hearing referred. Mild speech delay at 18 months.",
            "age": 18.0, "ga": 32, "nicu": True, "nicu_days": 14,
            "hearing": "Refer", "vision": "Pass",
            "eeg_avail": True, "mri_avail": False, "hpo_avail": True,
            "bsr": 0.20, "ibi": 10.0, "delta": 150.0, "sef": 13.0,
            "myelin": "Normal for age", "cc_z": 0.0, "bvol": 0.0,
            "hpo_sel": ["HP:0000750 — Delayed speech and language development"],
        },
        "🟠 Case 3 — Hypotonic Infant with Global Delay": {
            "desc": "28-week preterm, 30-day NICU. Missing motor & speech milestones. Low muscle tone.",
            "age": 24.0, "ga": 28, "nicu": True, "nicu_days": 30,
            "hearing": "Refer", "vision": "Refer",
            "eeg_avail": True, "mri_avail": True, "hpo_avail": True,
            "bsr": 0.40, "ibi": 20.0, "delta": 200.0, "sef": 10.0,
            "myelin": "Mildly delayed", "cc_z": -1.5, "bvol": -1.0,
            "hpo_sel": [
                "HP:0001263 — Global developmental delay",
                "HP:0001290 — Hypotonia (low muscle tone)",
            ],
        },
        "🔴 Case 4 — Syndromic Presentation (Moderate ID Risk)": {
            "desc": "Microcephaly, wide-set eyes, wide nasal bridge. 26-week preterm, 45-day NICU.",
            "age": 12.0, "ga": 26, "nicu": True, "nicu_days": 45,
            "hearing": "Refer", "vision": "Refer",
            "eeg_avail": True, "mri_avail": True, "hpo_avail": True,
            "bsr": 0.55, "ibi": 30.0, "delta": 300.0, "sef": 8.0,
            "myelin": "Moderately delayed", "cc_z": -2.5, "bvol": -2.0,
            "hpo_sel": [
                "HP:0000252 — Microcephaly (small head circumference)",
                "HP:0000316 — Hypertelorism (wide-set eyes)",
                "HP:0000431 — Wide nasal bridge",
                "HP:0001263 — Global developmental delay",
            ],
        },
        "🔴 Case 5 — Confirmed Intellectual Disability (Severe)": {
            "desc": "Confirmed moderate ID. Profound delay, microcephaly, severely atrophied brain.",
            "age": 30.0, "ga": 25, "nicu": True, "nicu_days": 90,
            "hearing": "Refer", "vision": "Refer",
            "eeg_avail": True, "mri_avail": True, "hpo_avail": True,
            "bsr": 0.85, "ibi": 55.0, "delta": 400.0, "sef": 4.0,
            "myelin": "Severely delayed", "cc_z": -3.5, "bvol": -3.5,
            "hpo_sel": [
                "HP:0002342 — Intellectual disability, moderate",
                "HP:0001263 — Global developmental delay",
                "HP:0000252 — Microcephaly (small head circumference)",
                "HP:0001290 — Hypotonia (low muscle tone)",
                "HP:0001256 — Mild intellectual disability",
            ],
        },
    }

    # ── Tab A: Manual Input ───────────────────────────────────────────────

    with tab_manual:
        with st.expander("📋 Quick-Load Sample Cases — click to auto-fill all fields", expanded=True):
            st.markdown("Select a pre-defined clinical scenario then hit **Run ID Risk Assessment**.")
            for case_name, case_data in SAMPLE_CASES.items():
                col_desc, col_btn = st.columns([4, 1])
                col_desc.markdown(f"**{case_name}**  \n_{case_data['desc']}_")
                if col_btn.button("Load ▶", key=f"load_{case_name}"):
                    for k, v in case_data.items():
                        if k != "desc":
                            st.session_state[k] = v
                    st.rerun()

        st.markdown("#### ―― Or enter clinical data manually below ――")

        with st.expander("👶 Basic Info", expanded=True):
            col_a, col_b = st.columns(2)
            with col_a:
                age_months = st.slider("Age at assessment (months)", 0.0, 36.0, 6.0, step=0.5, key="age")
                ga_weeks   = st.slider("Gestational age at birth (weeks)", 24, 42, 38, key="ga")
            with col_b:
                nicu_admission = st.checkbox("Admitted to NICU after birth", key="nicu")
                nicu_days = st.number_input("NICU stay (days)", 0, 180,
                                            value=int(st.session_state.get("nicu_days", 0)), key="nicu_days") \
                    if nicu_admission else 0
                hearing_screen = st.selectbox("Newborn hearing screen", ["Pass", "Refer", "Not done"], key="hearing")
                vision_screen  = st.selectbox("Vision screen", ["Pass", "Refer", "Not done"], key="vision")

            basic_score = 0.0
            preterm_weeks = max(0, 37 - ga_weeks)
            basic_score += min(preterm_weeks / 15.0, 0.30)
            basic_score += min(nicu_days / 120.0, 0.20)
            basic_score += 0.10 if hearing_screen == "Refer" else 0.0
            basic_score += 0.08 if vision_screen == "Refer" else 0.0
            basic_score = min(basic_score, 0.55)

            st.caption(f"✏️ Corrected age: **{(age_months - (37 - ga_weeks) / 4.0 if ga_weeks < 37 else age_months):.1f} months**")

        # EEG
        with st.expander("🧠 EEG (if available)"):
            eeg_available = st.checkbox("EEG recording available", key="eeg_avail")
            eeg_symptom_score = 0.0
            if eeg_available:
                cols_e = st.columns(2)
                with cols_e[0]:
                    bsr      = st.slider("Burst-suppression ratio", 0.0, 1.0, 0.1, key="bsr")
                    ibi_mean = st.slider("Mean inter-burst interval (s)", 0.0, 60.0, 5.0, key="ibi")
                with cols_e[1]:
                    delta_pw = st.number_input("Delta band power (µV²/Hz)", value=100.0, key="delta")
                    sef95    = st.slider("SEF95 (Hz)", 0.0, 30.0, 15.0, key="sef")

                edf_up = st.file_uploader("Or upload EDF file", type=["edf"], key="edf_up")
                if edf_up is not None:
                    with st.spinner("Preprocessing EDF..."):
                        prep = _api_upload_file("/preprocess/edf", edf_up.name, edf_up.getvalue())
                    if prep:
                        st.success(f"EDF processed: {prep.get('n_epochs', 0)} epochs, "
                                   f"{prep.get('n_channels', 0)} channels, {prep.get('sample_rate', 0)}Hz")
                        feats = prep.get("features", [])
                        if feats:
                            st.dataframe(pd.DataFrame([feats],
                                columns=["Delta", "Theta", "Alpha", "Beta", "TotalPower",
                                         "BSR", "IBI_Mean", "IBI_Std", "SEF95", "Amp_Mean", "Amp_Std"]),
                                use_container_width=True)

                eeg_symptom_score = min(bsr * 0.40 + min(ibi_mean / 60.0, 1.0) * 0.25
                                         + max(0, (15.0 - sef95) / 15.0) * 0.15, 0.90)
            else:
                bsr, ibi_mean, sef95 = 0.1, 5.0, 15.0

        # MRI
        with st.expander("🫁 MRI (if available)"):
            mri_available = st.checkbox("Brain MRI available", key="mri_avail")
            mri_symptom_score = 0.0
            if mri_available:
                cols_m = st.columns(2)
                with cols_m[0]:
                    myelin = st.selectbox("Myelination status",
                        ["Normal for age", "Mildly delayed", "Moderately delayed", "Severely delayed"], key="myelin")
                    cc_z   = st.slider("Corpus callosum z-score", -4.0, 2.0, 0.0, key="cc_z")
                with cols_m[1]:
                    bvol_z = st.slider("Brain volume z-score", -4.0, 2.0, 0.0, key="bvol")

                nii_up = st.file_uploader("Or upload NIfTI file", type=["gz", "nii"], key="nii_up")
                if nii_up is not None:
                    with st.spinner("Preprocessing NIfTI..."):
                        prep = _api_upload_file("/preprocess/nifti", nii_up.name, nii_up.getvalue())
                    if prep:
                        st.success(f"NIfTI processed: subject={prep.get('subject_id')}, "
                                   f"shape={prep.get('shape')}, slices={prep.get('n_slices')}")
                        st.info(prep.get("myelination_note", ""))

                myelin_scores = {"Normal for age": 0.0, "Mildly delayed": 0.20,
                                 "Moderately delayed": 0.40, "Severely delayed": 0.65}
                mri_symptom_score = min(myelin_scores.get(myelin, 0.0)
                                         + max(0, -cc_z / 4.0) * 0.15
                                         + max(0, -bvol_z / 4.0) * 0.15, 0.90)
            else:
                myelin = "Normal for age"
                cc_z, bvol_z = 0.0, 0.0

        # HPO
        with st.expander("🧬 HPO Phenotype (if assessed)"):
            hpo_available = st.checkbox("HPO phenotype assessment available", key="hpo_avail")
            hpo_symptom_score = 0.0
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
                selected_hpo = st.multiselect("Select observed HPO terms:", HPO_OPTIONS, key="hpo_sel")

                HPO_SEVERITY = {
                    "HP:0002342": 0.40, "HP:0001249": 0.38, "HP:0001256": 0.32,
                    "HP:0001263": 0.28, "HP:0000252": 0.22, "HP:0001290": 0.18,
                    "HP:0000316": 0.14, "HP:0000431": 0.12, "HP:0000750": 0.10,
                    "HP:0000322": 0.08,
                }
                hpo_symptom_score = min(sum(HPO_SEVERITY.get(t.split(" — ")[0].strip(), 0.0)
                                            for t in selected_hpo), 0.90) if selected_hpo else 0.0

        # ── Predict ────────────────────────────────────────────────────────
        if st.button("🔍 Run ID Risk Assessment", type="primary", use_container_width=True):
            available_mods = []
            payload = {
                "hpo_symptom_score": float(hpo_symptom_score) if hpo_available else 0.0,
                "eeg_symptom_score": float(max(hpo_symptom_score if hpo_available else 0.0, basic_score * 0.5, eeg_symptom_score if eeg_available else 0.0)),
                "mri_symptom_score": float(mri_symptom_score) if mri_available else 0.0,
            }

            if eeg_available:
                payload["eeg"] = [[0.0] * 7680] * 19
                available_mods.append("EEG")
            if mri_available:
                payload["mri"] = [[[0.0] * 64] * 64] * 3
                available_mods.append("MRI")
            if hpo_available and selected_hpo:
                n = 5284
                vec = np.zeros(n, dtype=np.float32)
                for term_str in selected_hpo:
                    vec[int(hash(term_str.split(" — ")[0].strip().replace(":", "")) % n)] = 1.0
                payload["hpo"] = vec.tolist()
                available_mods.append("HPO")

            if not available_mods:
                st.error("Enable at least one modality to run prediction.")
            else:
                with st.spinner("Running inference via FastAPI..."):
                    result = _api_post("/predict", json=payload)

                if result:
                    prob = result["risk_probability"]
                    dq = result["dq_estimate"]
                    importance = result["modality_importance"]
                    confidence = result["confidence"]
                    warnings = result.get("warnings", [])

                    st.markdown("---")
                    st.subheader("📋 Prediction Results")

                    if prob < 0.5:
                        st.markdown('<div style="background:#22c55e;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em">🟢 TYPICAL DEVELOPMENT</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background:#ef4444;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em">🔴 ID RISK DETECTED</div>', unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Risk Probability", f"{prob * 100:.1f}%")
                    c2.metric("DQ Estimate", f"{dq:.0f}")
                    c2.markdown(dq_badge(dq), unsafe_allow_html=True)
                    c3.metric("Confidence", confidence)
                    st.plotly_chart(gauge_chart(prob * 100), use_container_width=True)

                    mod_names = ["EEG", "MRI", "HPO"]
                    colors = ["#22c55e" if m in available_mods else "#d1d5db" for m in mod_names]
                    fig_imp = go.Figure(go.Bar(x=importance, y=mod_names, orientation="h",
                                               marker_color=colors,
                                               text=[f"{v:.1%}" for v in importance],
                                               textposition="auto"))
                    fig_imp.update_layout(title="Modality Importance", xaxis_title="Weight", height=220,
                                           margin=dict(l=50, r=20, t=40, b=20))
                    st.plotly_chart(fig_imp, use_container_width=True)

                    for w in warnings:
                        st.warning(w)

                    with st.expander("🔬 Clinical Interpretation"):
                        if prob < 0.5:
                            st.markdown("This screening result does not indicate elevated ID risk. "
                                        "Continue routine developmental surveillance (ASQ-3 at 4, 9, 18, 24, 30 months).")
                        else:
                            st.markdown("""
This screening result suggests further evaluation is recommended.

> **This is NOT a diagnosis — it is a screening flag.**

**Recommended next steps:**
1. Referral to developmental pediatrician
2. Formal Bayley-4 developmental assessment
3. Audiology evaluation
4. Ophthalmology evaluation
5. Genetic panel if dysmorphic features present
6. Early Intervention referral (IDEA Part C)
7. Brain MRI if not already obtained
                            """)

                    st.info("⚕️ **EarlyMind is a research screening tool only.** Not FDA cleared. "
                            "Results must be interpreted by qualified healthcare professionals.")

    # ── Tab B: Load from File ─────────────────────────────────────────────

    with tab_file:
        st.markdown("#### Upload a subject JSON file")
        st.code("""{
  "hpo_symptom_score": 0.3,
  "eeg_symptom_score": 0.2,
  "mri_symptom_score": 0.4
}""", language="json")

        json_file = st.file_uploader("Upload subject JSON", type=["json"], key="json_up")
        if json_file is not None:
            data = json.load(json_file)
            st.json(data)

            if st.button("🔍 Run Assessment (from file)", type="primary"):
                with st.spinner("Running inference..."):
                    result = _api_post("/predict", json=data)

                if result:
                    prob = result["risk_probability"]
                    dq = result["dq_estimate"]
                    importance = result["modality_importance"]
                    confidence = result["confidence"]
                    warnings = result.get("warnings", [])

                    available_mods = []
                    if "eeg" in data: available_mods.append("EEG")
                    if "mri" in data: available_mods.append("MRI")
                    if "hpo" in data: available_mods.append("HPO")

                    st.markdown("---")
                    st.subheader("📋 Prediction Results")
                    if prob < 0.5:
                        st.markdown('<div style="background:#22c55e;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em">🟢 TYPICAL DEVELOPMENT</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div style="background:#ef4444;color:white;padding:20px;border-radius:12px;text-align:center;font-size:1.5em">🔴 ID RISK DETECTED</div>', unsafe_allow_html=True)

                    c1, c2, c3 = st.columns(3)
                    c1.metric("Risk Probability", f"{prob * 100:.1f}%")
                    c2.metric("DQ Estimate", f"{dq:.0f}")
                    c2.markdown(dq_badge(dq), unsafe_allow_html=True)
                    c3.metric("Confidence", confidence)
                    st.plotly_chart(gauge_chart(prob * 100), use_container_width=True)

                    mod_names = ["EEG", "MRI", "HPO"]
                    colors = ["#22c55e" if m in available_mods else "#d1d5db" for m in mod_names]
                    fig_imp = go.Figure(go.Bar(x=importance, y=mod_names, orientation="h",
                                               marker_color=colors,
                                               text=[f"{v:.1%}" for v in importance],
                                               textposition="auto"))
                    fig_imp.update_layout(title="Modality Importance", xaxis_title="Weight", height=220,
                                           margin=dict(l=50, r=20, t=40, b=20))
                    st.plotly_chart(fig_imp, use_container_width=True)

                    for w in warnings:
                        st.warning(w)

                    st.info("⚕️ **EarlyMind is a research screening tool only.** Not FDA cleared.")

# ═══════════════════════════════════════════════════════════════════════════
# PAGE 4: DOCUMENTATION
# ═══════════════════════════════════════════════════════════════════════════

# Documentation logic has been extracted into the standalone docs_app.py
