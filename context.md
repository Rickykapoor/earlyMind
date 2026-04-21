# EarlyMind 🧠 - Project Context

## Overview
EarlyMind is a production-ready multimodal deep learning system designed for the **early detection of Intellectual Disability (ID) risk** in infants aged 0–36 months. It serves as a research screening tool (not FDA cleared) to classify developmental quotient (DQ) severity bands rather than diagnosing clinical conditions.

## Architecture
The system utilizes a split-service architecture commonly deployed as a single stack via `docker-compose`:
1. **FastAPI Backend (`api/main.py`)**: Runs on port `8000`. Provides the core model inference endpoint (`/predict`), multimodal and unimodal evaluation routes, model health checks (`/health`), and heavy preprocessing routes for biological signal parsing (`/preprocess/edf` & `/preprocess/nifti`).
2. **Streamlit Frontend (`app.py`)**: Runs on port `8501`. A user-friendly dashboard connected directly to the FastAPI backend. It features three pages:
   - **Data Overview**: Explores properties of the Helsinki Neonatal EEG Dataset, Baby Open Brains MRI dataset, and Human Phenotype Ontology terms.
   - **Training Monitor**: Reads generated JSON reports (`reports/training_history.json`) to plot Training/Validation Loss and AUC curves.
   - **Predict Infant**: An interactive screening tool offering manual clinical inputs or loading JSON case files. It also ships with 5 pre-defined clinical cases (from typical development to profound ID risk).

## Data Modalities & Algorithms
EarlyMind aggregates three distinct biological and observational data structures using a **Late Fusion Transformer** (`src/models/fusion_model.py`):

### 1. EEG (Electroencephalogram)
* **Dataset**: Helsinki Neonatal EEG Dataset (19-channel neonatal EEG).
* **Specs**: Sampled at 256Hz, processed in 30-second epochs. Configured filtering includes 0.5–40.0Hz bandpass and 50Hz notch filter.
* **Architecture**: Processed by `src/models/eeg_encoder.py` leveraging a Multi-scale CNN combined with a Temporal Transformer stack.
* **Clinical Analytics**: Handled by parsers measuring Burst-suppression ratio (BSR), Mean Inter-burst interval (IBI) in seconds, Delta band power, and Spectral Edge Frequency (SEF95).

### 2. MRI (Magnetic Resonance Imaging)
* **Dataset**: Baby Open Brains (ds004797 from OpenNeuro). Includes 10 subjects sized iteratively over 0-36 months.
* **Specs**: T1w/T2w imaging downsampled to 3 standardized central slices (Axial, Coronal, Sagittal) dimensionally sized 64x64.
* **Architecture**: Evaluated via `src/models/mri_encoder.py` employing an EfficientNet-B0 backbone augmented with cross-slice attention mechanics.
* **Clinical Analytics**: Measures generalized observations including Myelination delay levels, Corpus callosum z-score reductions, and Brain volume z-scores.

### 3. HPO (Human Phenotype Ontology)
* **Dataset**: Derived from generalized facial and physical phenotypic annotations (`phenotype.hpoa`).
* **Specs**: Parses roughly 5,284 specialized phenotypic terms targeting genetic and dysmorphic disease signatures.
* **Architecture**: Handled via `src/models/hpo_encoder.py` using a Self-attention Multi-Layer Perceptron (MLP).
* **Key Indicators**: Extracts conditions indicating ID risk components such as Microcephaly (HP:0000252), Hypertelorism (HP:0000316), Hypotonia (HP:0001290), Global developmental delay (HP:0001263).

## Directory Structure
* `app.py`: The Streamlit web application.
* `api/`: Scripts governing the FastAPI router logic and background handling.
* `src/config.py`: A central `Config` dataclass initialized from `params.yaml`. Defines training metadata (patience, focal alpha, batch sizes), path references binding relative names to project root, and overarching pipeline scales.
* `src/models/`: Holds individual model feature extractors (`eeg_encoder.py`, `mri_encoder.py`, `hpo_encoder.py`) and the final combined `fusion_model.py`.
* `src/data/`: Domain-specific file parsers and dataset iterators (`eeg_loader.py`, `mri_loader.py`, `hpo_loader.py`, and `fusion_dataset.py`).
* `src/utils/`, `src/scripts/`, `src/training/`: Training loops implementing Focal Loss combined linearly with Severity Risk weighting formulas.
* `dvc.yaml` & `params.yaml`: The system relies heavily on iterative version control. DVC is applied structurally to pull necessary `.tar.gz` and NIfTI volumes locally prior to inference or tuning runs locally/Colab.

## Developmental Quotient (DQ) Severity Setup
The model output blends categorical scaling into intuitive Developmental Quotient labels:
| DQ Range  | Diagnostic Label |
| --------- | ---------------- |
| 85–100    | Typical Development |
| 70–84     | Borderline (Monitor) |
| 55–69     | Mild ID Risk |
| 35–54     | Moderate ID Risk |
| 20–34     | Severe ID Risk |
| 0–19      | Profound ID Risk |

## Operational Logistics & Deployment
EarlyMind incorporates configuration rules suited for robust local development, research testing, or HuggingFace web instances.
- Docker containers use `ghcr.io` builds natively defined in the `Dockerfile`.
- Accepts variable configurations easily via `EARLYMIND_CKPT_DIR` (checkpoint locators) and handles memory-managed file streaming (`EARLYMIND_MAX_EEG_MB`/`EARLYMIND_MAX_MRI_MB`).
