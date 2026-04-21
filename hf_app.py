"""
Hugging Face Spaces - EarlyMind Gradio Interface
Self-contained version with model inference
"""
import os
import tempfile
from pathlib import Path

import gradio as gr
import numpy as np
import requests
import torch
import torch.nn.functional as F

DEVICE = torch.device("cpu")
N_HPO = 5284

MODEL_URL = os.environ.get(
    "MODEL_URL",
    "https://huggingface.co/datasets/earlymind/fusion-model/resolve/main/fusion_model.pt"
)

_model = None


def load_model():
    global _model
    if _model is not None:
        return _model
    
    try:
        ckpt_path = Path(tempfile.gettempdir()) / "fusion_model.pt"
        if not ckpt_path.exists():
            import urllib.request
            gr.Info("Downloading model (485MB)... This may take a few minutes.")
            urllib.request.urlretrieve(MODEL_URL, ckpt_path)
        
        import sys
        src_path = Path(__file__).resolve().parent
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        
        from src.models.fusion_model import build_fusion_model
        model = build_fusion_model(n_hpo=N_HPO)
        state = torch.load(str(ckpt_path), map_location=DEVICE, weights_only=False)
        model.load_state_dict(state, strict=False)
        model.to(DEVICE)
        model.eval()
        _model = model
        return model
    except Exception as e:
        gr.Warning(f"Model not loaded: {e}")
        return None


def get_dq_label(dq: float) -> str:
    if dq >= 85:
        return "Normal"
    elif dq >= 70:
        return "Mild ID"
    elif dq >= 55:
        return "Moderate ID"
    elif dq >= 40:
        return "Severe ID"
    else:
        return "Profound ID"


def _pad_tensor(arr, target_len, axis=-1):
    arr = np.array(arr, dtype=np.float32)
    current_len = arr.shape[axis]
    if current_len < target_len:
        pad_width = [(0, 0)] * arr.ndim
        pad_width[axis] = (0, target_len - current_len)
        arr = np.pad(arr, pad_width, mode="constant", constant_values=0)
    elif current_len > target_len:
        slices = [slice(None)] * arr.ndim
        slices[axis] = slice(0, target_len)
        arr = arr[tuple(slices)]
    return arr


def _build_batch(eeg, mri, hpo):
    batch = {}
    missing = []
    
    if eeg is not None:
        arr = _pad_tensor(np.array(eeg), 7680, axis=1)
        arr = _pad_tensor(arr, 19, axis=0)
        batch["eeg"] = torch.from_numpy(arr).unsqueeze(0)
    else:
        missing.append("eeg")
    
    if mri is not None:
        arr = _pad_tensor(np.array(mri), 64, axis=1)
        arr = _pad_tensor(arr, 64, axis=2)
        arr = _pad_tensor(arr, 3, axis=0)
        batch["mri"] = torch.from_numpy(arr).unsqueeze(0).unsqueeze(0)
    else:
        missing.append("mri")
    
    if hpo is not None:
        arr = _pad_tensor(np.array(hpo), N_HPO)
        batch["hpo"] = torch.from_numpy(arr).unsqueeze(0)
    else:
        missing.append("hpo")
    
    return batch, missing


def predict(eeg_file, mri_file, hpo_file, eeg_score, mri_score, hpo_score):
    model = load_model()
    if model is None:
        return "❌ Model not loaded. Please check model URL or configuration."
    
    try:
        eeg = mri = hpo = None
        
        if eeg_file is not None:
            path = Path(eeg_file.name)
            if path.suffix == '.npy':
                eeg = np.load(path).tolist()
            elif path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(path)
                eeg = df.values.tolist()
        
        if mri_file is not None:
            path = Path(mri_file.name)
            if path.suffix == '.npy':
                mri = np.load(path).tolist()
        
        if hpo_file is not None:
            path = Path(hpo_file.name)
            if path.suffix == '.npy':
                hpo = np.load(path).tolist()
            elif path.suffix == '.csv':
                import pandas as pd
                df = pd.read_csv(path)
                hpo = df.values.tolist()[0] if len(df) > 0 else None
        
        if eeg is None and mri is None and hpo is None:
            return "⚠️ Please upload at least one modality (EEG, MRI, or HPO)"
        
        batch, missing = _build_batch(eeg, mri, hpo)
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        
        severity = torch.tensor([[
            float(eeg_score) if eeg_score else 0.0,
            float(mri_score) if mri_score else 0.0,
            float(hpo_score) if hpo_score else 0.0
        ]], dtype=torch.float32).to(DEVICE)
        
        with torch.no_grad():
            logits = model(batch, severity)
            probs = torch.sigmoid(logits)
            risk_prob = probs[0, 0].item()
            dq = (1 - risk_prob) * 100
            dq_label = get_dq_label(dq)
            
            has_eeg = "eeg" in batch
            has_mri = "mri" in batch
            has_hpo = "hpo" in batch
            n_modalities = sum([has_eeg, has_mri, has_hpo])
            
            weights = []
            if has_eeg:
                weights.append(1.0)
            if has_mri:
                weights.append(1.0)
            if has_hpo:
                weights.append(1.0)
            
            weights = torch.tensor(weights, dtype=torch.float32)
            weights = F.softmax(weights, dim=0).tolist()
            
            modality_weights = []
            idx = 0
            if has_eeg:
                modality_weights.append(weights[idx])
                idx += 1
            else:
                modality_weights.append(0.0)
            if has_mri:
                modality_weights.append(weights[idx])
                idx += 1
            else:
                modality_weights.append(0.0)
            if has_hpo:
                modality_weights.append(weights[idx])
            else:
                modality_weights.append(0.0)
        
        distance = abs(risk_prob - 0.5)
        if distance > 0.3:
            confidence = "High"
        elif distance > 0.15:
            confidence = "Moderate"
        else:
            confidence = "Low"
        
        output = f"""
## Prediction Results

| Metric | Value |
|--------|-------|
| **Risk Probability** | {risk_prob:.4f} |
| **DQ Estimate** | {dq:.2f} |
| **DQ Label** | {dq_label} |
| **Confidence** | {confidence} |

### Modality Importance
- EEG: {modality_weights[0]:.3f}
- MRI: {modality_weights[1]:.3f}  
- HPO: {modality_weights[2]:.3f}

*Missing modalities: {', '.join(missing) if missing else 'None'}*
"""
        return output
        
    except Exception as e:
        return f"❌ Error: {str(e)}"


def check_health():
    model = load_model()
    if model is not None:
        return "✅ System Ready | Model: Loaded | Device: CPU"
    else:
        return "⚠️ Model not available"


with gr.Blocks(title="EarlyMind - ID Risk Detection", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # EarlyMind
    ### Multimodal Infant ID Risk Detection System
    
    Upload EEG, MRI data and/or HPO features for risk assessment.
    """)
    
    with gr.Tab("Status"):
        gr.Markdown("### System Status")
        status_output = gr.Textbox(label="Status", lines=2)
        gr.Button("Check", icon="🔄").click(fn=check_health, outputs=status_output)
    
    with gr.Tab("Predict"):
        gr.Markdown("### Input Data")
        with gr.Row():
            with gr.Column():
                eeg_file = gr.File(label="EEG Data (.npy or .csv)", file_count="single")
                eeg_score = gr.Slider(0, 1, value=0, label="EEG Symptom Score")
            with gr.Column():
                mri_file = gr.File(label="MRI Data (.npy)", file_count="single")
                mri_score = gr.Slider(0, 1, value=0, label="MRI Symptom Score")
            with gr.Column():
                hpo_file = gr.File(label="HPO Features (.npy or .csv)", file_count="single")
                hpo_score = gr.Slider(0, 1, value=0, label="HPO Symptom Score")
        
        predict_btn = gr.Button("Predict", variant="primary")
        result_output = gr.Markdown()
        
        predict_btn.click(
            fn=predict,
            inputs=[eeg_file, mri_file, hpo_file, eeg_score, mri_score, hpo_score],
            outputs=result_output
        )
    
    with gr.Tab("Info"):
        gr.Markdown("""
        ## About EarlyMind
        
        This system performs multimodal risk assessment for infant developmental disorders using:
        
        - **EEG**: Electroencephalogram data
        - **MRI**: Magnetic Resonance Imaging data  
        - **HPO**: Human Phenotype Ontology features
        
        DQ (Developmental Quotient) Classification:
        - Normal: DQ >= 85
        - Mild ID: DQ 70-84
        - Moderate ID: DQ 55-69
        - Severe ID: DQ 40-54
        - Profound ID: DQ < 40
        """)

demo.launch(server_name="0.0.0.0", server_port=7860)
