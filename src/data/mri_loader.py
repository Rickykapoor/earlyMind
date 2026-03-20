"""
src/data/mri_loader.py
NIfTI loading and slice extraction for Baby Open Brains (BIDS) dataset.
Handles T1w/T2w selection by age, center-of-mass slicing, and augmentation.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import nibabel as nib
import numpy as np
from scipy.ndimage import center_of_mass, zoom

from src.config import cfg
from src.utils.label_utils import parse_mri_participants_tsv

warnings.filterwarnings("ignore")

SLICE_SIZE = cfg.data.mri_slice_size  # 64


# ---------------------------------------------------------------------------
# Core loading helpers
# ---------------------------------------------------------------------------

def load_nifti(nii_path: str | Path) -> np.ndarray:
    """
    Load a NIfTI file and return a float32 3D numpy array.
    Squeezes any extra dimensions (e.g. 4D → 3D by taking first volume).
    """
    img = nib.load(str(nii_path))
    vol = img.get_fdata(dtype=np.float32)
    while vol.ndim > 3:
        vol = vol[..., 0]
    return vol


def normalize_volume(vol: np.ndarray) -> np.ndarray:
    """Intensity normalization to [0, 1]."""
    vmin, vmax = vol.min(), vol.max()
    return (vol - vmin) / (vmax - vmin + 1e-8)


def _resize_slice(slc: np.ndarray, target: int = SLICE_SIZE) -> np.ndarray:
    """Resize a 2D slice to (target, target) using scipy zoom."""
    h, w = slc.shape
    zh = target / h
    zw = target / w
    resized = zoom(slc, (zh, zw), order=1)
    # Clip to exact size in case of rounding
    resized = resized[:target, :target]
    if resized.shape[0] < target:
        pad = np.zeros((target - resized.shape[0], resized.shape[1]), dtype=np.float32)
        resized = np.vstack([resized, pad])
    if resized.shape[1] < target:
        pad = np.zeros((resized.shape[0], target - resized.shape[1]), dtype=np.float32)
        resized = np.hstack([resized, pad])
    return resized.astype(np.float32)


def extract_slices(vol: np.ndarray) -> np.ndarray:
    """
    Extract axial, coronal, sagittal slices at the center of mass.
    Returns: (3, SLICE_SIZE, SLICE_SIZE) float32 array.
    """
    vol = normalize_volume(vol)

    # Center of mass (only on nonzero voxels for robustness)
    binary = vol > 0.1
    if binary.sum() == 0:
        binary = np.ones_like(vol, dtype=bool)

    try:
        cx, cy, cz = [int(round(v)) for v in center_of_mass(binary)]
    except Exception:
        cx = vol.shape[0] // 2
        cy = vol.shape[1] // 2
        cz = vol.shape[2] // 2

    # Clamp indices
    cx = max(0, min(cx, vol.shape[0] - 1))
    cy = max(0, min(cy, vol.shape[1] - 1))
    cz = max(0, min(cz, vol.shape[2] - 1))

    axial    = vol[cx, :, :]   # (Y, Z)
    coronal  = vol[:, cy, :]   # (X, Z)
    sagittal = vol[:, :, cz]   # (X, Y)

    axial    = _resize_slice(axial)
    coronal  = _resize_slice(coronal)
    sagittal = _resize_slice(sagittal)

    return np.stack([axial, coronal, sagittal], axis=0)  # (3, 64, 64)


# ---------------------------------------------------------------------------
# BIDS file discovery
# ---------------------------------------------------------------------------

def find_structural_scan(
    subject_dir: Path,
    age_months: float,
    prefer_t2w: bool = False,
) -> Optional[Path]:
    """
    Find the appropriate structural scan (T1w or T2w) for a BIDS subject.

    For infants < 12 months: prefer T2w (inverted contrast).
    For subjects >= 12 months: prefer T1w.

    Looks under sub-XX/ses-*/anat/ and sub-XX/anat/.
    """
    use_t2w = prefer_t2w or (age_months < 12.0)
    preferred = "T2w" if use_t2w else "T1w"
    fallback  = "T1w" if use_t2w else "T2w"

    def _search(modality: str) -> Optional[Path]:
        patterns = [
            f"**/*_{modality}.nii.gz",
            f"**/*_{modality}.nii",
        ]
        for pat in patterns:
            hits = sorted(subject_dir.glob(pat))
            if hits:
                return hits[0]
        return None

    scan = _search(preferred)
    if scan is None:
        scan = _search(fallback)
    return scan


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------

def process_mri_dataset(
    mri_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, dict]:
    """
    Process all subjects in a BIDS MRI directory.
    Saves: output_dir/sub-XX.npy — (3, 64, 64) float32 slice array.

    Returns dict: {participant_id: {"slices": ..., "label": ..., "dq": ..., "age_months": ...}}
    """
    mri_dir   = Path(mri_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parse participants.tsv for ages + labels
    participants_tsv = mri_dir / "participants.tsv"
    label_df = None
    if participants_tsv.exists():
        try:
            label_df = parse_mri_participants_tsv(str(participants_tsv))
        except Exception as e:
            print(f"  [WARNING] Could not parse participants.tsv: {e}")

    # Build subject → age_months map
    age_map: Dict[str, float] = {}
    label_map: Dict[str, Tuple[int, float]] = {}
    if label_df is not None:
        for _, row in label_df.iterrows():
            pid = str(row["participant_id"]).strip()
            age_map[pid]   = float(row["age_months"])
            label_map[pid] = (int(row["label"]), float(row["dq"]))

    # Discover subject directories
    subject_dirs = sorted(
        [d for d in mri_dir.iterdir()
         if d.is_dir() and d.name.startswith("sub-")]
    )
    if len(subject_dirs) == 0:
        raise FileNotFoundError(f"No sub-* directories found in {mri_dir}")

    results = {}

    for subj_dir in subject_dirs:
        pid = subj_dir.name  # e.g. "sub-01"
        age_months = age_map.get(pid, 6.0)  # default 6 months
        label, dq  = label_map.get(pid, (0, 90.0))

        print(f"  Processing MRI subject: {pid}, age={age_months:.1f}mo")

        scan_path = find_structural_scan(subj_dir, age_months)
        if scan_path is None:
            print(f"    [WARNING] No structural scan found for {pid}, skipping.")
            continue

        try:
            vol    = load_nifti(scan_path)
            slices = extract_slices(vol)   # (3, 64, 64)
        except Exception as e:
            print(f"    [ERROR] Failed to process {pid}: {e}")
            continue

        out_path = output_dir / f"{pid}.npy"
        np.save(str(out_path), slices)

        results[pid] = {
            "slices": slices,
            "label": label,
            "dq": dq,
            "age_months": age_months,
            "scan_path": str(scan_path),
            "out_path": str(out_path),
        }

    print(f"  MRI preprocessing complete: {len(results)} subjects → {output_dir}")
    return results


# ---------------------------------------------------------------------------
# MRI Augmentation helpers (for training)
# ---------------------------------------------------------------------------

def augment_mri_slices(
    slices: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply random augmentations to a (3, 64, 64) MRI slice array.
    Returns augmented copy of same shape.
    """
    from scipy.ndimage import gaussian_filter, rotate

    if rng is None:
        rng = np.random.default_rng()

    x = slices.copy().astype(np.float32)  # (3, 64, 64)

    # 1. Random horizontal flip
    if rng.random() > 0.5:
        x = x[:, :, ::-1].copy()

    # 2. Random rotation ±10 degrees
    angle = rng.uniform(-10, 10)
    x = np.stack([
        rotate(x[i], angle, reshape=False, mode="reflect")
        for i in range(x.shape[0])
    ], axis=0)

    # 3. Gaussian blur
    sigma = rng.uniform(0.1, 2.0)
    x = np.stack([
        gaussian_filter(x[i], sigma=sigma)
        for i in range(x.shape[0])
    ], axis=0)

    # 4. Random intensity scaling
    scale = rng.uniform(0.85, 1.15)
    x = x * scale

    # 5. Random crop and resize back to 64×64
    crop_px = int(rng.integers(0, 8))  # crop 0–7 pixels on each side
    if crop_px > 0:
        _, H, W = x.shape
        x_crop = x[:, crop_px:H-crop_px, crop_px:W-crop_px]
        x = np.stack([
            _resize_slice(x_crop[i], SLICE_SIZE)
            for i in range(x_crop.shape[0])
        ], axis=0)

    # Clip to [0, 1]
    x = np.clip(x, 0.0, 1.0)
    return x.astype(np.float32)


def simulate_delayed_myelination(
    slices: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply augmentations that simulate delayed myelination appearance.
    Used for pretraining the MRI encoder with synthetic 'ID-risk' label.
    """
    from scipy.ndimage import gaussian_filter

    if rng is None:
        rng = np.random.default_rng()

    x = slices.copy().astype(np.float32)

    # Stronger blurring (simulates unmyelinated white matter contrast loss)
    sigma = rng.uniform(1.5, 3.5)
    x = np.stack([gaussian_filter(x[i], sigma=sigma) for i in range(x.shape[0])], axis=0)

    # Intensity shift (reduces WM/GM contrast)
    x = x * rng.uniform(0.7, 0.9) + rng.uniform(0.05, 0.15)

    # Re-normalize to [0, 1]
    x = (x - x.min()) / (x.max() - x.min() + 1e-8)
    return x.astype(np.float32)
