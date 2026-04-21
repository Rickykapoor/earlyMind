import torch
import torch.nn as nn
import torch.nn.functional as F

class LateFusionTransformer(nn.Module):
    def __init__(self, n_hpo=5284):
        super().__init__()
        # Mock definition to satisfy torch.load and inference
        self.dummy_param = nn.Parameter(torch.zeros(1))

    def forward(self, batch, missing_modalities=None):
        # We need to return logits [B, 2], severity [B, 1], modality_importance [1, 3]
        B = 1
        if "eeg" in batch:
            B = batch["eeg"].shape[0]
            
        logits = torch.randn(B, 2).to(self.dummy_param.device)
        severity = torch.randn(B, 1).to(self.dummy_param.device) * 10 + 60
        modality_importance = torch.tensor([0.4, 0.4, 0.2]).to(self.dummy_param.device)
        
        return {
            "logits": logits,
            "severity": severity,
            "modality_importance": modality_importance
        }

def build_fusion_model(n_hpo=5284):
    return LateFusionTransformer(n_hpo=n_hpo)
