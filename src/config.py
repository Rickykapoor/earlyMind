"""
src/config.py
Loads params.yaml and exposes a Config dataclass with typed access to all hyperparameters.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import yaml


def _find_project_root() -> Path:
    """Walk up from this file until we find params.yaml."""
    here = Path(__file__).resolve().parent
    for candidate in [here, here.parent, here.parent.parent]:
        if (candidate / "params.yaml").exists():
            return candidate
    raise FileNotFoundError("params.yaml not found. Run from the project root.")


PROJECT_ROOT: Path = _find_project_root()


def _load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


@dataclass
class ModelConfig:
    embed_dim: int = 128
    fusion_heads: int = 4
    fusion_layers: int = 3
    dropout: float = 0.2
    eeg_channels: int = 19
    eeg_timesteps: int = 7680
    mri_img_size: int = 64
    mri_slices: int = 3
    hpo_n_features: int = 1024


@dataclass
class AugmentationConfig:
    mri_target_samples: int   = 10_000
    mri_aug_seed:       int   = 42
    mri_dq_noise_std:   float = 5.0
    use_augmented_mri:  bool  = True


@dataclass
class TrainingConfig:
    batch_size: int = 16
    epochs: int = 100
    lr: float = 3e-4
    weight_decay: float = 1e-4
    patience: int = 15
    focal_alpha: float = 0.25
    focal_gamma: float = 2.0
    severity_loss_weight: float = 0.5
    grad_clip: float = 1.0
    freeze_encoders_epochs: int = 10
    seed: int = 42


@dataclass
class DataConfig:
    eeg_sample_rate: int = 256
    eeg_epoch_seconds: int = 30
    eeg_epoch_overlap: float = 0.5
    eeg_bandpass_low: float = 0.5
    eeg_bandpass_high: float = 40.0
    eeg_notch_freq: float = 50.0
    mri_slice_size: int = 64
    hpo_min_disease_freq: int = 5
    train_val_test_split: List[float] = field(default_factory=lambda: [0.70, 0.15, 0.15])


@dataclass
class PathsConfig:
    eeg_raw: Path = Path("datasets/eeg/helsinki_neonatal")
    mri_raw: Path = Path("datasets/mri/baby_open_brains")
    hpo_raw: Path = Path("datasets/facial/hpo")
    eeg_processed: Path = Path("datasets/processed/eeg")
    mri_processed: Path = Path("datasets/processed/mri")
    hpo_processed: Path = Path("datasets/processed/facial")
    checkpoints: Path = Path("checkpoints")
    reports: Path = Path("reports")

    def __post_init__(self):
        # Make all paths absolute relative to project root
        for fname in self.__dataclass_fields__:
            val = getattr(self, fname)
            if not Path(val).is_absolute():
                setattr(self, fname, PROJECT_ROOT / val)

    def makedirs(self):
        """Create all output directories if they do not exist."""
        for fname in self.__dataclass_fields__:
            p = getattr(self, fname)
            p.mkdir(parents=True, exist_ok=True)


@dataclass
class SeverityConfig:
    typical: Tuple[int, int] = (85, 100)
    borderline: Tuple[int, int] = (70, 85)
    mild: Tuple[int, int] = (55, 70)
    moderate: Tuple[int, int] = (35, 55)
    severe: Tuple[int, int] = (20, 35)
    profound: Tuple[int, int] = (0, 20)


@dataclass
class Config:
    model:         ModelConfig       = field(default_factory=ModelConfig)
    training:      TrainingConfig    = field(default_factory=TrainingConfig)
    data:          DataConfig        = field(default_factory=DataConfig)
    paths:         PathsConfig       = field(default_factory=PathsConfig)
    severity:      SeverityConfig    = field(default_factory=SeverityConfig)
    augmentation:  AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def from_yaml(cls, path: Optional[Path] = None) -> "Config":
        if path is None:
            path = PROJECT_ROOT / "params.yaml"
        raw = _load_yaml(path)

        def _to_path_tuple(lst):
            return tuple(lst) if isinstance(lst, list) else lst

        model_raw = raw.get("model", {})
        training_raw = raw.get("training", {})
        data_raw = raw.get("data", {})
        paths_raw = raw.get("paths", {})
        severity_raw = raw.get("severity", {})

        aug_raw      = raw.get("augmentation", {})

        return cls(
            model        = ModelConfig(**model_raw),
            training     = TrainingConfig(**training_raw),
            data         = DataConfig(**data_raw),
            paths        = PathsConfig(**{k: Path(v) for k, v in paths_raw.items()}),
            severity     = SeverityConfig(
                **{k: tuple(v) for k, v in severity_raw.items()}
            ),
            augmentation = AugmentationConfig(**aug_raw),
        )

    def dq_label(self, dq: float) -> str:
        """Return human-readable DQ severity label."""
        if dq >= self.severity.typical[0]:
            return "Typical"
        elif dq >= self.severity.borderline[0]:
            return "Borderline"
        elif dq >= self.severity.mild[0]:
            return "Mild ID Risk"
        elif dq >= self.severity.moderate[0]:
            return "Moderate ID Risk"
        elif dq >= self.severity.severe[0]:
            return "Severe ID Risk"
        else:
            return "Profound ID Risk"


# Singleton — import cfg from anywhere
cfg: Config = Config.from_yaml()
