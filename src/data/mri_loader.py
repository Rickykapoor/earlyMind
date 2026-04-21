"""
src/data/mri_loader.py
=======================
MRI preprocessing pipeline for EarlyMind.

Reads Baby Open Brains (ds004797) T2w NIfTI volumes, extracts three
central canonical slices (axial, coronal, sagittal), normalises each to
[0, 1], and saves them as .npz files.

Also provides:
  - simulate_delayed_myelination()  — for augmentation/notebook demos
  - MRIDataset                      — PyTorch Dataset (real + augmented)

Usage (standalone):
    python -m src.data.mri_loader

DVC stage:
    python -c "from src.data.mri_loader import process_mri_dataset; \
               from src.config import cfg; \
               process_mri_dataset(cfg.paths.mri_raw, cfg.paths.mri_processed)"
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Developmental-quotient heuristic
# ---------------------------------------------------------------------------
# Baby Open Brains subjects are rat pups (~327–340 days old, weight 183–331 g).
# The dataset simulates structural MRI in neonatal neuroscience; it has no
# ground-truth human DQ scores.  We assign a *research-heuristic* DQ that is
# based on body weight z-score relative to the cohort mean (heavier/larger
# brain ↔ lower ID risk) and a small random seed so every rerun is
# reproducible.  This is clearly labelled as a heuristic — real deployment
# must use clinical DQ assessments.

_DQ_SEED = 2024


def _weight_to_dq(weight_g: float, cohort_weights: np.ndarray, rng: np.random.Generator) -> float:
    """
    Map body weight z-score → DQ in [0, 100] with realistic noise.

    Higher weight → closer to 'typical' (DQ ≈ 85–100).
    Lower weight  → shifted toward borderline/mild ID risk (DQ ≈ 55–80).
    """
    mu, sigma = cohort_weights.mean(), cohort_weights.std() + 1e-6
    z = (weight_g - mu) / sigma          # z-score relative to cohort
    # DQ = 80 + 10*z, clamped to [35, 100], with ±5 research noise
    dq_raw = 80.0 + 10.0 * z
    dq_noisy = dq_raw + rng.normal(0.0, 5.0)
    return float(np.clip(dq_noisy, 35.0, 100.0))


def _dq_to_label(dq: float) -> int:
    """Convert DQ float → integer class label (0 = typical, 5 = profound)."""
    if dq >= 85:
        return 0   # Typical
    elif dq >= 70:
        return 1   # Borderline
    elif dq >= 55:
        return 2   # Mild ID Risk
    elif dq >= 35:
        return 3   # Moderate ID Risk
    elif dq >= 20:
        return 4   # Severe ID Risk
    else:
        return 5   # Profound ID Risk


# ---------------------------------------------------------------------------
# NIfTI helpers (nibabel optional — falls back to raw gz read)
# ---------------------------------------------------------------------------

def _load_nifti_volume(nii_path: Path) -> np.ndarray:
    """Return 3-D float32 array from a NIfTI / NIfTI.gz file."""
    try:
        import nibabel as nib  # type: ignore
        img = nib.load(str(nii_path))
        data = np.asarray(img.get_fdata(), dtype=np.float32)
        return data
    except ImportError:
        pass

    # Fallback: read raw compressed NIfTI (gzip) manually.
    # NIfTI1 layout: 348-byte header, then voxel data (little-endian float32).
    import gzip
    import struct

    with gzip.open(str(nii_path), "rb") as fh:
        raw = fh.read()

    # Read dims from header bytes 40–55 (short[8], little-endian)
    dims = struct.unpack_from("<8h", raw, 40)
    ndim = dims[0]
    shape = tuple(dims[1 : ndim + 1])

    # Data offset from vox_offset field (byte 108, float32)
    (vox_offset,) = struct.unpack_from("<f", raw, 108)
    offset = int(vox_offset) if vox_offset >= 348 else 352

    # datatype field (byte 70, short)
    (dtype_code,) = struct.unpack_from("<h", raw, 70)
    dtype_map = {2: np.uint8, 4: np.int16, 8: np.int32, 16: np.float32, 64: np.float64}
    dtype = dtype_map.get(dtype_code, np.float32)

    n_voxels = int(np.prod(shape))
    data = np.frombuffer(raw[offset : offset + n_voxels * np.dtype(dtype).itemsize], dtype=dtype)
    data = data.reshape(shape, order="F").astype(np.float32)
    return data


def _extract_central_slices(vol: np.ndarray, size: int = 64) -> np.ndarray:
    """
    Extract the central axial, coronal, and sagittal slices from a 3-D volume.

    Returns shape (3, size, size) — float32, normalised to [0, 1].
    """
    # Ensure at least 3 spatial dims; drop time dim if 4-D (fMRI)
    if vol.ndim == 4:
        vol = vol[..., vol.shape[3] // 2]

    x, y, z = vol.shape

    def _centre_crop(img_2d: np.ndarray) -> np.ndarray:
        """Resize 2-D slice to (size, size) via naive zoom (no scipy dep)."""
        try:
            from scipy.ndimage import zoom  # type: ignore
            scale_h = size / img_2d.shape[0]
            scale_w = size / img_2d.shape[1]
            return zoom(img_2d, (scale_h, scale_w), order=1).astype(np.float32)
        except ImportError:
            pass
        # Very lightweight nearest-neighbour fallback
        rows = np.linspace(0, img_2d.shape[0] - 1, size).astype(int)
        cols = np.linspace(0, img_2d.shape[1] - 1, size).astype(int)
        return img_2d[np.ix_(rows, cols)].astype(np.float32)

    axial    = _centre_crop(vol[x // 2, :, :])   # fixed x
    coronal  = _centre_crop(vol[:, y // 2, :])   # fixed y
    sagittal = _centre_crop(vol[:, :, z // 2])   # fixed z

    slices = np.stack([axial, coronal, sagittal], axis=0)   # (3, H, W)

    # Normalise each slice independently to [0, 1]
    for i in range(3):
        vmin, vmax = slices[i].min(), slices[i].max()
        if vmax > vmin:
            slices[i] = (slices[i] - vmin) / (vmax - vmin)
        else:
            slices[i] = np.zeros_like(slices[i])

    return slices


# ---------------------------------------------------------------------------
# Myelination delay simulation (used by augmenter + notebook demos)
# ---------------------------------------------------------------------------

def simulate_delayed_myelination(
    slices: np.ndarray,
    severity: float = 0.5,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Simulate white-matter myelination delay by selectively blurring
    high-intensity regions (white matter) in the MRI slices.

    Parameters
    ----------
    slices   : (3, H, W) float32 normalised to [0, 1]
    severity : 0 = no effect, 1 = maximum delay (very blurred WM)
    rng      : numpy random generator (for reproducibility)

    Returns
    -------
    (3, H, W) float32 — augmented slices
    """
    if rng is None:
        rng = np.random.default_rng()

    from scipy.ndimage import gaussian_filter  # type: ignore

    out = slices.copy()
    sigma = severity * rng.uniform(1.5, 3.5)

    for i in range(3):
        blurred = gaussian_filter(out[i], sigma=sigma).astype(np.float32)
        # Only affect the bright (white-matter) voxels
        wm_mask = out[i] > 0.6
        out[i] = np.where(wm_mask, blurred, out[i])

    return out


# ---------------------------------------------------------------------------
# Main processing function
# ---------------------------------------------------------------------------

def process_mri_dataset(
    mri_dir: Path,
    output_dir: Path,
    slice_size: int = 64,
) -> Dict[str, dict]:
    """
    Preprocess all T2w NIfTI volumes in *mri_dir* and save as .npz to
    *output_dir*.

    Each .npz contains:
      slices      : float32 (3, slice_size, slice_size)
      dq          : float scalar — heuristic developmental quotient
      label       : int scalar  — DQ severity class 0–5
      age_months  : float scalar
      subject_id  : str

    Returns a dict {subject_id: {slices, dq, label, age_months}}.
    """
    mri_dir = Path(mri_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("MRI loader: scanning %s", mri_dir)

    # ------------------------------------------------------------------
    # 1. Collect subjects + parse participants.tsv
    # ------------------------------------------------------------------
    subj_dirs = sorted(d for d in mri_dir.iterdir() if d.is_dir() and d.name.startswith("sub-"))
    if not subj_dirs:
        raise FileNotFoundError(f"No sub-XX directories found in {mri_dir}")

    participants_tsv = mri_dir / "participants.tsv"
    meta: Dict[str, dict] = {}

    if participants_tsv.exists():
        import csv
        with open(participants_tsv, newline="") as fh:
            reader = csv.DictReader(fh, delimiter="\t")
            for row in reader:
                pid = row.get("participant_id", "").strip()
                if not pid:
                    continue
                try:
                    age_days = float(row.get("Age (days)", 0) or 0)
                except ValueError:
                    age_days = 330.0
                try:
                    weight_g = float(row.get("Weight (g)", 250) or 250)
                except ValueError:
                    weight_g = 250.0
                meta[pid] = {
                    "age_days": age_days,
                    "age_months": age_days / 30.44,
                    "weight_g": weight_g,
                }

    # Collect weights for cohort-level z-score
    all_weights = np.array([v["weight_g"] for v in meta.values()], dtype=np.float32)
    if len(all_weights) == 0:
        all_weights = np.array([250.0])

    rng = np.random.default_rng(_DQ_SEED)

    # ------------------------------------------------------------------
    # 2. Process each subject
    # ------------------------------------------------------------------
    results: Dict[str, dict] = {}

    for subj_dir in subj_dirs:
        pid = subj_dir.name  # e.g. "sub-01"

        # Prefer T2w (clearer for neonatal myelin), fall back to BOLD mean
        nii_candidates = (
            list(subj_dir.glob("**/anat/*T2w.nii.gz"))
            + list(subj_dir.glob("**/anat/*T1w.nii.gz"))
            + list(subj_dir.glob("**/func/*bold.nii.gz"))
        )
        if not nii_candidates:
            log.warning("No NIfTI found for %s — skipping", pid)
            continue

        nii_path = nii_candidates[0]
        log.info("Processing %s from %s", pid, nii_path.name)

        try:
            vol = _load_nifti_volume(nii_path)
        except Exception as exc:
            log.error("Failed to load %s: %s", nii_path, exc)
            continue

        slices = _extract_central_slices(vol, size=slice_size)  # (3, 64, 64)

        # Subject metadata
        subj_meta = meta.get(pid, {"age_days": 330.0, "age_months": 10.8, "weight_g": 250.0})
        dq = _weight_to_dq(subj_meta["weight_g"], all_weights, rng)
        label = _dq_to_label(dq)

        out_path = output_dir / f"{pid}.npz"
        np.savez_compressed(
            out_path,
            slices=slices,
            dq=np.float32(dq),
            label=np.int32(label),
            age_months=np.float32(subj_meta["age_months"]),
            subject_id=np.bytes_(pid),
        )

        results[pid] = {
            "slices": slices,
            "dq": dq,
            "label": label,
            "age_months": subj_meta["age_months"],
        }
        log.info("  ✓ %s → DQ=%.1f, label=%d", pid, dq, label)

    log.info("MRI preprocessing complete: %d subjects saved to %s", len(results), output_dir)
    return results


# ---------------------------------------------------------------------------
# PyTorch Dataset
# ---------------------------------------------------------------------------

class MRIDataset:
    """
    torch.utils.data.Dataset-compatible loader for MRI slice stacks.

    Parameters
    ----------
    real_dir       : path to datasets/processed/mri (10 real .npz files)
    augmented_dir  : path to datasets/mri/augmented (optional synthetic pool)
    use_augmented  : whether to include synthetic samples
    transform      : optional callable applied to each (3,H,W) slice array
    """

    def __init__(
        self,
        real_dir: Path,
        augmented_dir: Optional[Path] = None,
        use_augmented: bool = True,
        transform=None,
    ):
        real_dir = Path(real_dir)
        self._files = sorted(real_dir.glob("*.npz"))

        if use_augmented and augmented_dir is not None:
            aug_dir = Path(augmented_dir)
            if aug_dir.exists():
                self._files += sorted(aug_dir.glob("*.npz"))
            else:
                log.warning("Augmented dir %s not found — using real data only", aug_dir)

        if not self._files:
            raise FileNotFoundError(f"No .npz files found in {real_dir}")

        self.transform = transform
        log.info("MRIDataset: %d samples loaded", len(self._files))

    # Make it usable without torch imported
    def __len__(self) -> int:
        return len(self._files)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, float, int]:
        data = np.load(self._files[idx], allow_pickle=True)
        slices = data["slices"].astype(np.float32)   # (3, 64, 64)
        dq     = float(data["dq"])
        label  = int(data["label"])

        if self.transform is not None:
            slices = self.transform(slices)

        return slices, dq, label

    # Convenience: summary stats
    def label_distribution(self) -> dict:
        counts = {i: 0 for i in range(6)}
        names  = {0: "Typical", 1: "Borderline", 2: "Mild ID",
                  3: "Moderate ID", 4: "Severe ID", 5: "Profound ID"}
        for f in self._files:
            d = np.load(f, allow_pickle=True)
            counts[int(d["label"])] += 1
        return {names[k]: v for k, v in counts.items()}


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO, format="%(levelname)s  %(message)s")

    # Allow running from repo root or src/
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.config import cfg  # type: ignore

    results = process_mri_dataset(
        mri_dir=cfg.paths.mri_raw,
        output_dir=cfg.paths.mri_processed,
        slice_size=cfg.data.mri_slice_size,
    )
    print(f"\n✅  Saved {len(results)} MRI samples to {cfg.paths.mri_processed}")
    for pid, info in results.items():
        print(f"   {pid}: DQ={info['dq']:.1f}  label={info['label']}  age={info['age_months']:.1f}mo")
