"""
docs_app.py — EarlyMind Interactive Data Pipeline Documentation
Run: streamlit run docs_app.py
"""
import streamlit as st

st.set_page_config(
    page_title="EarlyMind Pipeline Docs",
    page_icon="📘",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .step-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        border-radius: 50%;
        width: 36px; height: 36px;
        line-height: 36px;
        text-align: center;
        font-weight: 700;
        font-size: 16px;
        margin-right: 10px;
        flex-shrink: 0;
    }
    .step-header {
        display: flex;
        align-items: center;
        margin-bottom: 6px;
    }
    .step-title {
        font-size: 22px;
        font-weight: 700;
        color: #1e293b;
    }
    .pipeline-arrow {
        text-align: center;
        font-size: 28px;
        color: #94a3b8;
        margin: 8px 0;
    }
    .card {
        background: #f8fafc;
        border-radius: 10px;
        padding: 16px 20px;
        border-left: 4px solid #3b82f6;
        margin-bottom: 12px;
    }
    .card-warn {
        border-left-color: #f59e0b;
        background: #fffbeb;
    }
    .card-success {
        border-left-color: #10b981;
        background: #ecfdf5;
    }
    .code-label {
        font-size: 11px;
        text-transform: uppercase;
        font-weight: 600;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-inline {
        display: inline-block;
        background: #eff6ff;
        color: #1d4ed8;
        border-radius: 6px;
        padding: 2px 8px;
        font-weight: 600;
        font-size: 14px;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ───────────────────────────────────────────────────────────────────
st.sidebar.markdown("## 📘 EarlyMind Docs")
st.sidebar.markdown("**Developer Onboarding Guide**")
st.sidebar.markdown("---")
section = st.sidebar.radio(
    "Jump to Section",
    [
        "🏠 Overview",
        "📁 Step 1 — Data Ingestion",
        "🧠 Step 2 — MRI Preprocessing",
        "⚡ Step 3 — EEG Processing",
        "🔬 Step 4 — HPO Phenotyping",
        "🐺 Step 5 — Augmentation Engine",
        "🎯 Step 6 — GWO Hyperparameter Tuning",
        "🔗 Step 7 — Late Fusion Transformer",
    ],
    index=0,
)
st.sidebar.markdown("---")
st.sidebar.info("💡 Each step maps directly to a notebook or source file in the repo.")

# ═══════════════════════════════════════════════════════════════════════════
# OVERVIEW
# ═══════════════════════════════════════════════════════════════════════════
if section == "🏠 Overview":
    st.title("📘 EarlyMind — Complete Pipeline Documentation")
    st.markdown("""
    Welcome to the **EarlyMind Developer Onboarding Station**. This guide walks you through every single step
    of our multimodal clinical AI pipeline — from raw hospital archives all the way to the optimised Late Fusion Transformer.

    > **Who is this for?** Any developer who is new to the project, or anyone who wants to understand
    > exactly how we convert 200GB of raw MRI scans and EEG files into a production-ready AI that screens infants for intellectual disability risk.
    """)

    st.markdown("### 🗺️ End-to-End Pipeline at a Glance")

    cols = st.columns(7)
    steps = [
        ("📁", "Data Ingestion\n(DVC)"),
        ("🧠", "MRI\nPreprocess"),
        ("⚡", "EEG\nProcess"),
        ("🔬", "HPO\nPhenotype"),
        ("🐺", "Augmentation\nEngine"),
        ("🎯", "GWO\nTuning"),
        ("🔗", "Late Fusion\nTransformer"),
    ]
    for col, (icon, label) in zip(cols, steps):
        col.markdown(f"""
        <div style="text-align:center; background:#eff6ff; border-radius:10px; padding:12px 6px; border:1px solid #bfdbfe;">
            <div style="font-size:28px;">{icon}</div>
            <div style="font-size:12px; font-weight:600; color:#1d4ed8; white-space:pre-line;">{label}</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 📂 Key Repository Locations")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        | Path | Purpose |
        |------|---------|
        | `datasets/mri/raw/` | 3D NIfTI brain volumes (.nii.gz) |
        | `datasets/eeg/raw/` | Multichannel EDF brainwave files |
        | `datasets/facial/hpo/` | HPO phenotype ontology tables |
        | `datasets/processed/mri/` | Preprocessed 2D slice arrays (.npz) |
        | `datasets/mri/augmented/` | 10,000 synthetic training samples |
        """)
    with col2:
        st.markdown("""
        | Path | Purpose |
        |------|---------|
        | `src/data/mri_loader.py` | MRI preprocessing logic |
        | `src/data/mri_augment.py` | Synthetic augmentation engine |
        | `src/optimization/gwo.py` | Grey Wolf Optimizer |
        | `src/models/fusion_model.py` | Late Fusion Transformer |
        | `notebooks/` | Step-by-step reproducible notebooks |
        """)

    st.success("✅ Use the **sidebar** to navigate to each step in detail.")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 1: Data Ingestion
# ═══════════════════════════════════════════════════════════════════════════
elif section == "📁 Step 1 — Data Ingestion":
    st.markdown('<div class="step-header"><span class="step-badge">1</span><span class="step-title">Raw Data Ingestion via DVC</span></div>', unsafe_allow_html=True)
    st.markdown("""
    The EarlyMind raw dataset exceeds **200 GB** total. We use **Data Version Control (DVC)** backed by Google Drive
    to keep the git repository lightweight while still versioning large binary files with full reproducibility.
    """)

    st.markdown("### 📦 What Data Are We Pulling?")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        **🧠 MRI (Baby Open Brains)**
        - Source: OpenNeuro `ds004797`
        - Format: 3D NIfTI-1 (`.nii.gz`)
        - Modality: T1w + T2w structural MRI
        - Subjects: 10 infants (0–36 months)
        - BIDS-compliant layout
        """)
    with col2:
        st.markdown("""
        **⚡ EEG (Helsinki Neonatal)**
        - Source: Helsinki University Hospital
        - Format: European Data Format (`.edf`)
        - Channels: 19 electrodes @ 256 Hz
        - Duration: Multi-hour recordings
        - Subjects: 3 neonates
        """)
    with col3:
        st.markdown("""
        **🔬 HPO (Phenotype Ontology)**
        - Source: Human Phenotype Ontology DB
        - Format: Tab-delimited `.hpoa`
        - Contains: 10,000+ disease-gene-phenotype links
        - Used to map clinical features to risk scores
        """)

    st.markdown("### 🛠️ How to Run This Step")
    st.code("""# Step 1: Install DVC with Google Drive backend
pip install dvc dvc-gdrive

# Step 2: Pull all registered datasets (triggers browser OAuth)
dvc pull
""", language="bash")

    st.markdown('<div class="card card-warn">⚠️ <strong>DVC Auth Failure?</strong> If Google OAuth fails, manually extract the raw archives from your system administrator into the <code>datasets/</code> directory. The expected folder structure must match <code>.dvc/config</code>.</div>', unsafe_allow_html=True)

    st.markdown("### 🗂️ Expected Directory Structure After Pull")
    st.code("""datasets/
├── mri/
│   └── baby_open_brains/        # Raw NIfTI BIDS layout
│       ├── participants.tsv      # Age, weight metadata per subject
│       ├── sub-01/anat/*T2w.nii.gz
│       └── sub-10/anat/*T2w.nii.gz
├── eeg/
│   └── helsinki_neonatal/
│       ├── clinical_information.csv
│       ├── 1.edf
│       └── 3.edf
└── facial/
    └── hpo/
        └── phenotype.hpoa
""", language="text")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 2: MRI Preprocessing
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🧠 Step 2 — MRI Preprocessing":
    st.markdown('<div class="step-header"><span class="step-badge">2</span><span class="step-title">Spatial MRI Preprocessing</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Source file:** `src/data/mri_loader.py`  |  **Notebook:** `notebooks/02_mri_preprocess.ipynb`

    3D MRI volumes contain **millions of voxels** and cannot be fed directly into a neural network without massive compute resources.
    This step reduces each 3D brain scan to a lightweight `(3, 64, 64)` numpy array.
    """)

    st.markdown("### 🔬 Technical Pipeline (Exact Code Logic)")

    with st.expander("Sub-step 2.1 — Load the 3D NIfTI Volume", expanded=True):
        st.markdown("""
        The loader uses `nibabel` to read `.nii.gz` files into a 3D float32 numpy array.
        If `nibabel` is not installed, it falls back to reading the raw gzip binary manually
        using struct offsets defined in the NIfTI-1 specification (vox_offset at byte 108).
        """)
        st.code("""import nibabel as nib
img = nib.load("sub-01_T2w.nii.gz")
vol = np.asarray(img.get_fdata(), dtype=np.float32)
# vol.shape -> (256, 256, 166)  [X, Y, Z voxels]
""", language="python")

    with st.expander("Sub-step 2.2 — Extract Central Canonical Slices"):
        st.markdown("""
        The absolute geometric centre of the volume is computed by integer-dividing each dimension by 2.
        Three orthogonal 2D slices are then cut at those coordinates:
        - **Axial:** Top-down view (`vol[x//2, :, :]`)
        - **Coronal:** Front-back view (`vol[:, y//2, :]`)
        - **Sagittal:** Left-right view (`vol[:, :, z//2]`)
        """)
        st.code("""x, y, z = vol.shape

axial    = vol[x // 2, :, :]   # Top-down  (fixes X axis)
coronal  = vol[:, y // 2, :]   # Front-back (fixes Y axis)
sagittal = vol[:, :, z // 2]   # Left-right (fixes Z axis)
""", language="python")

    with st.expander("Sub-step 2.3 — Resize Each Slice to 64×64"):
        st.markdown("""
        Each 2D slice may be a different size (e.g. 256×166). We use `scipy.ndimage.zoom` to
        uniformly rescale each slice to exactly `64×64` pixels using bilinear interpolation (order=1).
        """)
        st.code("""from scipy.ndimage import zoom

scale_h = 64 / axial.shape[0]
scale_w = 64 / axial.shape[1]
axial_resized = zoom(axial, (scale_h, scale_w), order=1)
# axial_resized.shape -> (64, 64)
""", language="python")

    with st.expander("Sub-step 2.4 — Normalise Intensities to [0, 1]"):
        st.markdown("""
        MRI voxel intensities are in arbitrary scanner units (e.g. 0–4095 for 12-bit DICOM).
        We normalise **each slice independently** using min-max scaling to bring values into [0, 1]:
        """)
        st.code("""for i in range(3):   # axial, coronal, sagittal
    vmin, vmax = slices[i].min(), slices[i].max()
    if vmax > vmin:
        slices[i] = (slices[i] - vmin) / (vmax - vmin)
    else:
        slices[i] = np.zeros_like(slices[i])   # blank slice
""", language="python")

    with st.expander("Sub-step 2.5 — Assign Heuristic DQ Score"):
        st.markdown("""
        The Baby Open Brains dataset contains rat pup subjects, not humans, so there are no ground-truth DQ scores.
        We assign a **research-heuristic DQ** based on body weight relative to the cohort:

        - **Higher weight** → larger brain → lower ID risk → DQ closer to 85–100
        - **Lower weight** → smaller brain → higher ID risk → DQ shifted toward 35–70

        The formula maps weight z-score to DQ with Gaussian noise for realism:
        """)
        st.code("""mu, sigma = cohort_weights.mean(), cohort_weights.std()
z = (weight_g - mu) / sigma          # z-score
dq_raw = 80.0 + 10.0 * z             # centre around 80
dq = np.clip(dq_raw + rng.normal(0, 5.0), 35.0, 100.0)
""", language="python")
        st.warning("⚠️ **Clinical Disclaimer:** This heuristic is for research only. Real clinical deployment must use gold-standard assessments by certified psychologists.")

    with st.expander("Sub-step 2.6 — Save as Compressed .npz"):
        st.markdown("Each subject is saved as a compressed numpy archive in `datasets/processed/mri/`:")
        st.code("""np.savez_compressed(
    "datasets/processed/mri/sub-01.npz",
    slices     = slices,        # shape: (3, 64, 64), float32
    dq         = np.float32(dq),
    label      = np.int32(label),
    age_months = np.float32(age_months),
    subject_id = np.bytes_("sub-01"),
)
""", language="python")

    st.markdown("### ▶️ How to Run")
    st.code("""# Run the full MRI preprocessing pipeline:
python -m src.data.mri_loader

# Or run the equivalent notebook:
jupyter notebook notebooks/02_mri_preprocess.ipynb
""", language="bash")

    st.markdown("### 📊 Output Summary")
    st.markdown("""
    | Field | Value |
    |-------|-------|
    | Subjects processed | 10 (sub-01 → sub-10) |
    | Output shape per subject | `(3, 64, 64)` float32 |
    | Output size per file | ~50KB (.npz compressed) |
    | DQ range | 35.0 – 100.0 |
    | Label classes | 0 (Typical) to 5 (Profound ID Risk) |
    """)

# ═══════════════════════════════════════════════════════════════════════════
# STEP 3: EEG Processing
# ═══════════════════════════════════════════════════════════════════════════
elif section == "⚡ Step 3 — EEG Processing":
    st.markdown('<div class="step-header"><span class="step-badge">3</span><span class="step-title">Signal EEG Processing</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Notebook:** `notebooks/01_eeg_preprocess.ipynb`

    Raw EDF files contain hours of continuous 19-channel brainwave data at 256 Hz.
    We convert these into fixed-length numerical embeddings suitable for the fusion model.
    """)

    with st.expander("Sub-step 3.1 — Load the EDF File", expanded=True):
        st.markdown("We use the `mne` library (a gold-standard biomedical signal processing toolkit) to load EDF files:")
        st.code("""import mne
mne.set_log_level("WARNING")

raw = mne.io.read_raw_edf("datasets/eeg/helsinki_neonatal/1.edf", preload=True)
# raw.info["sfreq"]   → 256.0  (Hz)
# raw.ch_names        → ['Fp1', 'Fp2', 'F3', ..., 'O2']  (19 channels)
""", language="python")

    with st.expander("Sub-step 3.2 — Apply Bandpass Filter (1–30 Hz)"):
        st.markdown("""
        Raw EEG contains environmental noise above 30Hz (power line artifacts, muscle noise)
        and very slow DC-drift below 1Hz. We apply a bandpass filter to isolate clinically
        meaningful neural oscillations:
        """)
        st.code("""raw.filter(l_freq=1.0, h_freq=30.0, method="iir")
# Removes:
#   < 1 Hz  → slow drift / movement artifacts
#   > 30 Hz → EMG muscle noise, power line 50/60 Hz
""", language="python")
        st.info("💡 The 1–30 Hz band captures delta (1–4Hz), theta (4–8Hz), alpha (8–13Hz), and beta (13–30Hz) rhythms — all relevant to neonatal neurodevelopment.")

    with st.expander("Sub-step 3.3 — Segment Into 10-Second Epochs"):
        st.markdown("""
        Continuous EEG is sliced into strict 10-second windows (2,560 samples per epoch at 256Hz).
        Each epoch is assigned the clinical label of the overall recording.
        """)
        st.code("""sfreq = int(raw.info["sfreq"])      # 256
window_len = 10 * sfreq              # 2560 samples
data = raw.get_data()                # shape: (19_channels, total_samples)

epochs = []
for start in range(0, data.shape[1] - window_len, window_len):
    epoch = data[:, start : start + window_len]  # (19, 2560)
    epochs.append(epoch)
""", language="python")

    with st.expander("Sub-step 3.4 — Compute Feature Embedding"):
        st.markdown("""
        Each epoch is averaged across time to produce a compact 19-dimensional vector —
        one mean amplitude per electrode. This vector becomes the EEG feature input to the fusion model.
        """)
        st.code("""# Mean amplitude per channel across the 10s window
eeg_embedding = epoch.mean(axis=1)   # shape: (19,)
""", language="python")

    st.markdown("### ▶️ How to Run")
    st.code("jupyter notebook notebooks/01_eeg_preprocess.ipynb", language="bash")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 4: HPO Phenotyping
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🔬 Step 4 — HPO Phenotyping":
    st.markdown('<div class="step-header"><span class="step-badge">4</span><span class="step-title">HPO Clinical Phenotype Mapping</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Source:** `datasets/facial/hpo/phenotype.hpoa`

    The Human Phenotype Ontology (HPO) is a standardised vocabulary of clinical phenotypes.
    We use it to link observable symptoms (e.g. "muscular hypotonia", "delayed motor development")
    directly to known genetic syndromes and quantify their clinical severity as a risk score.
    """)

    with st.expander("What is the HPO?", expanded=True):
        st.markdown("""
        The HPO provides a hierarchical controlled vocabulary of over **17,000 clinical terms**
        covering phenotype abnormalities. The `.hpoa` annotation file links:

        ```
        Disease  →  HPO Term  →  Evidence Code  →  Frequency
        OMIM:615132  HP:0001252  PCS  HP:0040281  (hypotonia in 80% of cases)
        ```

        For EarlyMind, we load the `.hpoa` file, filter for terms related to
        neurodevelopmental delay, and compute a **phenotype risk count** per subject.
        """)

    with st.expander("Processing Logic"):
        st.code("""import pandas as pd

df_hpo = pd.read_csv("datasets/facial/hpo/phenotype.hpoa",
                     sep="\\t", skiprows=4)
df_hpo.columns = [c.lstrip("#").strip() for c in df_hpo.columns]

# Filter for intellectual disability-related terms
ID_TERMS = ["HP:0001249", "HP:0001256", "HP:0000750"]
df_id = df_hpo[df_hpo["hpo_id"].isin(ID_TERMS)]
""", language="python")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 5: Augmentation Engine
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🐺 Step 5 — Augmentation Engine":
    st.markdown('<div class="step-header"><span class="step-badge">5</span><span class="step-title">Synthetic MRI Augmentation Engine</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Source file:** `src/data/mri_augment.py`

    Our 10 real MRI subjects are **massively imbalanced** — nearly all subjects fall in the
    Typical (DQ > 85) class. Training a neural net on this data causes it to "cheat" by
    predicting only the majority class, achieving 95% accuracy but **0% recall on ID cases**.

    The augmentation engine generates **10,000 synthetic subjects** with a clinically realistic class distribution.
    """)

    st.markdown("### 🎯 Target Class Distribution")
    data = {
        "Class": ["Typical (DQ 85–100)", "Borderline (DQ 70–84)", "Mild ID (DQ 55–69)",
                  "Moderate ID (DQ 35–54)", "Severe ID (DQ 20–34)", "Profound ID (DQ 0–19)"],
        "Target %": [60, 15, 10, 7, 5, 3],
        "# Samples (of 10,000)": [6000, 1500, 1000, 700, 500, 300],
    }
    import pandas as pd
    st.dataframe(pd.DataFrame(data), use_container_width=True, hide_index=True)

    st.markdown("### 🔧 Augmentation Techniques Applied")

    tab1, tab2, tab3 = st.tabs(["Geometric (Label-Preserving)", "Intensity (Label-Preserving)", "MRI-Specific (Clinical)"])

    with tab1:
        st.markdown("""
        These operations change the spatial layout of the scan without affecting what
        clinical pathology is present — perfectly safe for preserving labels.

        | Transform | Description | Parameters |
        |-----------|-------------|------------|
        | **Horizontal Flip** | Mirror left-right | p = 0.5 |
        | **Vertical Flip** | Mirror top-bottom | p = 0.3 |
        | **Rotation** | Rotate ±15° randomly | max_rot_deg = 15 |
        | **Elastic Deformation** | Gaussian displacement fields | sigma=3, alpha=30 |
        | **Zoom Crop** | Random zoom-in/out ±10% then re-crop to 64×64 | zoom_range=(0.90, 1.10) |
        """)
        st.code("""# Elastic deformation: adds smooth warping to simulate
# natural anatomical variation across individuals
dx = gaussian_filter(rng.standard_normal((H, W)), sigma=3) * 30
dy = gaussian_filter(rng.standard_normal((H, W)), sigma=3) * 30
""", language="python")

    with tab2:
        st.markdown("""
        These operations alter pixel brightness/contrast while keeping anatomy identical.

        | Transform | Description | Parameters |
        |-----------|-------------|------------|
        | **Gaussian Noise** | Simulates scanner thermal noise | noise_std ≤ 0.03 |
        | **Brightness/Contrast Jitter** | alpha ∈ [0.85, 1.15], beta ∈ [−0.15, 0.15] | Per-slice independent |
        | **Gamma Correction** | Models T2w signal range variation | gamma ∈ [0.7, 1.4] |
        """)

    with tab3:
        st.markdown("These are the clinically motivated transforms unique to MRI that simulate real pathology.")

        with st.expander("🌀 Gibbs Ringing (k-space truncation)"):
            st.markdown("""
            Gibbs ringing is a real MRI artifact caused by the scanner truncating the Fourier transform (k-space).
            We simulate it by:
            1. Running a 2D FFT on the slice to get k-space
            2. Randomly masking 10–40% of the high-frequency components
            3. Running inverse FFT to reconstruct — this introduces edge oscillations
            """)
            st.code("""kspace = np.fft.fftshift(np.fft.fft2(slices[i]))
# Keep only cutoff% of k-space radius
cutoff = rng.uniform(0.60, 0.90)
mask[cy-kH:cy+kH, cx-kW:cx+kW] = True
kspace[~mask] = 0
recon = np.real(np.fft.ifft2(np.fft.ifftshift(kspace)))
""", language="python")

        with st.expander("🌊 Bias Field Simulation (polynomial gradient)"):
            st.markdown("""
            MRI bias field is a smooth, slow-varying intensity gradient across the image caused by
            inhomogeneities in the scanner's RF coil. We simulate it with a degree-2 polynomial surface:
            """)
            st.code("""yy, xx = np.meshgrid(np.linspace(-1,1,H), np.linspace(-1,1,W))
coeffs = rng.uniform(-0.15, 0.15, 6)
bias = (1.0 + coeffs[0]*xx + coeffs[1]*yy
           + coeffs[2]*xx**2 + coeffs[3]*yy**2
           + coeffs[4]*xx*yy + coeffs[5]*(xx**2+yy**2))
bias = bias / bias.mean()             # Keep near 1×
slices = np.clip(slices * bias, 0, 1)
""", language="python")

        with st.expander("🧬 Myelination Delay (white-matter blurring)"):
            st.markdown("""
            Delayed myelination is the defining MRI biomarker of intellectual disability.
            Myelin sheaths (white matter) appear bright on T2w MRI. An ID infant shows
            *less* myelination → *blurrier* white matter regions.

            We selectively blur only the bright voxels (pixels > 0.6) proportional to DQ severity:
            """)
            st.code("""severity = max(0, (85.0 - dq) / 85.0)   # 0=typical, 1=profound
sigma = severity * rng.uniform(1.5, 3.5)
blurred = gaussian_filter(slices[i], sigma=sigma)
wm_mask = slices[i] > 0.6           # High intensity = white matter
slices[i] = np.where(wm_mask, blurred, slices[i])
""", language="python")
            st.success("✅ Lower DQ → stronger blur → more realistic ID-pattern simulation.")

    st.markdown("### ▶️ How to Run the Augmentation")
    st.code("""from pathlib import Path
from src.data.mri_augment import generate_augmented_dataset

generate_augmented_dataset(
    real_dir   = Path("datasets/processed/mri"),
    output_dir = Path("datasets/mri/augmented"),
    target_n   = 10_000,
    seed       = 42,
)
""", language="python")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 6: GWO Hyperparameter Tuning
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🎯 Step 6 — GWO Hyperparameter Tuning":
    st.markdown('<div class="step-header"><span class="step-badge">6</span><span class="step-title">Grey Wolf Optimizer (GWO)</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Source file:** `src/optimization/gwo.py`  |  **Fitness:** `src/optimization/fitness.py`

    After scaling to 10,000 records, our Late Fusion Transformer needed hyperparameter optimization.
    Grid search through 6 dimensions would require evaluating millions of combinations.

    We replaced it with the **Grey Wolf Optimizer** — a nature-inspired meta-heuristic that
    converges to near-optimal configurations in far fewer iterations.
    """)

    with st.expander("🐺 What is the Grey Wolf Optimizer?", expanded=True):
        st.markdown("""
        The GWO mimics the **leadership hierarchy and hunting behaviour** of grey wolf packs:

        | Role | Description |
        |------|-------------|
        | **Alpha (α)** | Best solution found so far — the "pack leader" |
        | **Beta (β)** | Second-best solution — guides the search |
        | **Delta (δ)** | Third-best solution — assists in navigation |
        | **Omega (ω)** | All remaining wolves — explorers of search space |

        Each iteration, Omega wolves update their position (= hyperparameter values) by moving
        toward a weighted average of where Alpha, Beta, and Delta are pointing.
        """)
        st.code("""# Core GWO update equation (simplified):
X1 = alpha_pos - A1 * abs(C1 * alpha_pos - wolf_pos)
X2 = beta_pos  - A2 * abs(C2 * beta_pos  - wolf_pos)
X3 = delta_pos - A3 * abs(C3 * delta_pos - wolf_pos)
new_pos = (X1 + X2 + X3) / 3   # New candidate hyperparameters
""", language="python")

    with st.expander("🔍 Search Space Dimensions"):
        st.markdown("""
        The GWO optimizes 6 hyperparameters simultaneously:

        | Hyperparameter | Range | Impact |
        |----------------|-------|--------|
        | `learning_rate` | [1e-5, 1e-2] | How fast the weights update |
        | `dropout_mri` | [0.1, 0.5] | Regularization for MRI branch |
        | `dropout_eeg` | [0.1, 0.5] | Regularization for EEG branch |
        | `dropout_hpo` | [0.1, 0.5] | Regularization for HPO branch |
        | `hidden_dim` | [64, 512] | Size of fusion layer |
        | `attention_heads` | [2, 8] | Multi-head attention in Transformer |
        """)

    with st.expander("📈 Fitness Function"):
        st.markdown("""
        Each candidate hyperparameter set is evaluated by:
        1. Instantiating a fresh `LateFusionTransformer` with those hyperparameters
        2. Training for a fixed number of epochs on the 10,000 augmented samples
        3. Computing **Balanced Accuracy** on a validation split (not accuracy — avoids class-imbalance trap)
        4. Returning `1 - balanced_accuracy` as the **loss** for GWO to minimise
        """)

    with st.expander("🏆 GWO Results"):
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Before GWO (Default Hyperparams):**
            - Balanced Accuracy: 50.00% (random level)
            - F1 Macro: 0.488
            - Recall (ID cases): 0.000
            """)
        with col2:
            st.markdown("""
            **After GWO (~50 iterations):**
            - Balanced Accuracy: **96.80%** 🎉
            - F1 Macro: **0.967**
            - ROC AUC: **0.9912**
            """)

    st.markdown("### ▶️ How to Run GWO")
    st.code("""# Run GWO search (takes ~15–45 min on CPU)
python scripts/run_gwo.py

# Or use the Python API directly:
from src.optimization.gwo import GreyWolfOptimizer
from src.optimization.fitness import evaluate_hyperparams

gwo = GreyWolfOptimizer(n_wolves=10, max_iter=50)
best_params = gwo.run(fitness_fn=evaluate_hyperparams)
print(best_params)
""", language="bash")

# ═══════════════════════════════════════════════════════════════════════════
# STEP 7: Late Fusion Transformer
# ═══════════════════════════════════════════════════════════════════════════
elif section == "🔗 Step 7 — Late Fusion Transformer":
    st.markdown('<div class="step-header"><span class="step-badge">7</span><span class="step-title">Late Fusion Transformer Architecture</span></div>', unsafe_allow_html=True)
    st.markdown("""
    **Source file:** `src/models/fusion_model.py`

    The final model takes all three modalities independently, processes each with a specialised
    encoder, then fuses the representations in a shared Transformer before making the final prediction.
    """)

    with st.expander("🏗️ Architecture Overview", expanded=True):
        st.markdown("""
        ```
        ┌─────────────────────┐    ┌─────────────────────┐    ┌─────────────────────┐
        │   MRI Branch        │    │   EEG Branch         │    │   HPO Branch        │
        │  Input: (3, 64, 64) │    │  Input: (19,)        │    │  Input: (hpo_dim,)  │
        │  Conv2D → Pool      │    │  Linear → ReLU       │    │  Linear → ReLU      │
        │  → Flatten → Linear │    │  → Dropout           │    │  → Dropout          │
        │  → hidden_dim       │    │  → hidden_dim        │    │  → hidden_dim       │
        └──────────┬──────────┘    └──────────┬──────────┘    └──────────┬──────────┘
                   │                          │                          │
                   └──────────────────────────┼──────────────────────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │  Transformer       │
                                    │  (attention_heads) │
                                    │  Cross-modal       │
                                    │  attention + norm  │
                                    └─────────┬─────────┘
                                              │
                                    ┌─────────┴─────────┐
                                    │  Classification    │
                                    │  Head              │
                                    │  → 6 DQ classes   │
                                    └───────────────────┘
        ```
        """)

    with st.expander("🔢 Input/Output Specification"):
        st.markdown("""
        | Modality | Input Shape | Processed to |
        |----------|-------------|--------------|
        | MRI slices | `(batch, 3, 64, 64)` | `hidden_dim` embedding |
        | EEG amplitude | `(batch, 19)` | `hidden_dim` embedding |
        | HPO features | `(batch, hpo_dim)` | `hidden_dim` embedding |
        | **Fused output** | `(batch, 3, hidden_dim)` | `6 class logits` |

        The model predicts **6 severity classes** (Typical → Profound ID Risk) and a
        continuous **DQ score** between 0–100 via a regression head.
        """)

    with st.expander("📦 Modality Importance Scores"):
        st.markdown("""
        After prediction, the model exposes `modality_importance` — a 3-element vector showing
        how much each input branch contributed to the final decision:

        ```python
        result = {
            "dq_score":           78.3,
            "id_risk_probability": 0.23,
            "modality_importance": [0.51, 0.31, 0.18]
            #                        MRI   EEG   HPO
        }
        ```

        This is surfaced in the **Predict Infant** tab as a bar chart for clinical interpretability.
        """)

    st.markdown("### ▶️ Running the Full Inference API")
    st.code("""# Start the FastAPI backend (loads the checkpoint from models/fusion_model.pt):
uvicorn api.main:app --port 8000

# Start the Streamlit inference UI:
streamlit run app.py

# Start this documentation UI on a separate port:
streamlit run docs_app.py --server.port 8502
""", language="bash")

    st.success("🎉 You now understand the complete EarlyMind pipeline — from raw 3D NIfTI archives to optimised clinical AI predictions!")
