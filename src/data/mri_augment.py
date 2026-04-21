"""
src/data/mri_augment.py
========================
Synthetic MRI slice augmentation engine for EarlyMind data balancing.

Takes the 10 real preprocessed MRI slice stacks (.npz) and generates a
target number of synthetic samples (default 10 000) via clinically
motivated augmentation techniques.

Augmentation palette
--------------------
Geometric (label-preserving):
  • Random horizontal / vertical flip
  • Random rotation  (±15°)
  • Elastic deformation
  • Random zoom-and-crop

Intensity (label-preserving):
  • Gaussian noise
  • Brightness + contrast jitter
  • Gamma correction

MRI-specific:
  • Bias-field simulation  (polynomial intensity gradient)
  • Gibbs ringing          (truncated k-space reconvolution)
  • Myelination-delay      (white-matter blurring, severity tied to DQ)

Label assignment
----------------
Each synthetic sample inherits the base DQ of its source subject plus
Gaussian noise (std = dq_noise_std ≈ 5 DQ points), then is clamped
to [0, 100].  An additional "severity-guided" pathway resamples DQ
from a target distribution to achieve the desired class balance.

Requirements
------------
numpy, scipy, scikit-image (all in environment.yml / requirements.txt).
No torch / nibabel needed at augmentation time.

Usage
-----
  from src.data.mri_augment import generate_augmented_dataset
  generate_augmented_dataset(
      real_dir  = Path("datasets/processed/mri"),
      output_dir= Path("datasets/mri/augmented"),
      target_n  = 10_000,
      seed      = 42,
  )
"""
from __future__ import annotations

import logging
import math
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from scipy.ndimage import (  # type: ignore
    gaussian_filter,
    map_coordinates,
    rotate,
    zoom,
)

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# DQ severity helpers
# ---------------------------------------------------------------------------

#: Target class distribution (fraction of total) — matches realistic
#: neonatal ID epidemiology and balances the training signal.
TARGET_DIST = {
    0: 0.60,  # Typical       DQ 85–100
    1: 0.15,  # Borderline    DQ 70–84
    2: 0.10,  # Mild ID       DQ 55–69
    3: 0.07,  # Moderate ID   DQ 35–54
    4: 0.05,  # Severe ID     DQ 20–34
    5: 0.03,  # Profound ID   DQ  0–19
}

#: DQ range centres for each class (used when resampling DQ for target dist)
_CLASS_DQ_RANGES = {
    0: (85.0, 100.0),
    1: (70.0,  85.0),
    2: (55.0,  70.0),
    3: (35.0,  55.0),
    4: (20.0,  35.0),
    5: (0.0,   20.0),
}


def _dq_to_label(dq: float) -> int:
    if dq >= 85:  return 0
    if dq >= 70:  return 1
    if dq >= 55:  return 2
    if dq >= 35:  return 3
    if dq >= 20:  return 4
    return 5


def _sample_dq_for_class(cls: int, rng: np.random.Generator) -> float:
    lo, hi = _CLASS_DQ_RANGES[cls]
    return float(rng.uniform(lo, hi))


# ---------------------------------------------------------------------------
# Augmentation class
# ---------------------------------------------------------------------------

class MRISliceAugmenter:
    """
    Applies a randomly-selected combination of augmentations to a
    (3, H, W) float32 MRI slice stack.

    Parameters
    ----------
    p_flip_h    : probability of horizontal flip per slice
    p_flip_v    : probability of vertical flip per slice
    max_rot_deg : maximum rotation angle (degrees)
    noise_std   : max σ for additive Gaussian noise
    zoom_range  : (min_factor, max_factor) for random zoom
    """

    def __init__(
        self,
        p_flip_h:    float = 0.5,
        p_flip_v:    float = 0.3,
        max_rot_deg: float = 15.0,
        noise_std:   float = 0.03,
        zoom_range:  Tuple[float, float] = (0.90, 1.10),
    ):
        self.p_flip_h    = p_flip_h
        self.p_flip_v    = p_flip_v
        self.max_rot_deg = max_rot_deg
        self.noise_std   = noise_std
        self.zoom_range  = zoom_range

    # ------------------------------------------------------------------
    # Individual transforms
    # ------------------------------------------------------------------

    def _flip(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random horizontal and/or vertical flip (applied identically to all 3 slices)."""
        if rng.random() < self.p_flip_h:
            slices = slices[:, :, ::-1].copy()
        if rng.random() < self.p_flip_v:
            slices = slices[:, ::-1, :].copy()
        return slices

    def _rotate(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random 2-D rotation applied independently per slice."""
        angle = rng.uniform(-self.max_rot_deg, self.max_rot_deg)
        out = np.stack(
            [rotate(slices[i], angle, reshape=False, mode="reflect", order=1)
             for i in range(3)],
            axis=0,
        )
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _elastic(
        self,
        slices: np.ndarray,
        rng: np.random.Generator,
        sigma: float = 3.0,
        alpha: float = 30.0,
    ) -> np.ndarray:
        """
        Elastic deformation using Gaussian-smoothed random displacement fields.
        The same field is applied to all 3 slices for anatomical consistency.
        """
        H, W = slices.shape[1], slices.shape[2]

        # Random displacement
        dx = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma) * alpha
        dy = gaussian_filter(rng.standard_normal((H, W)).astype(np.float32), sigma) * alpha

        x, y   = np.meshgrid(np.arange(W), np.arange(H))
        coords_x = np.clip(x + dx, 0, W - 1).ravel()
        coords_y = np.clip(y + dy, 0, H - 1).ravel()

        out = np.zeros_like(slices)
        for i in range(3):
            warped = map_coordinates(slices[i], [coords_y, coords_x], order=1, mode="reflect")
            out[i] = warped.reshape(H, W)

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _zoom_crop(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Random zoom followed by centre-crop back to original size."""
        factor = rng.uniform(*self.zoom_range)
        H, W   = slices.shape[1], slices.shape[2]
        out    = np.zeros_like(slices)

        for i in range(3):
            zoomed = zoom(slices[i], factor, order=1)
            zh, zw = zoomed.shape

            if factor >= 1.0:
                # Zoom-in: centre crop
                cy, cx = zh // 2, zw // 2
                cropped = zoomed[cy - H // 2 : cy + H - H // 2,
                                 cx - W // 2 : cx + W - W // 2]
                # Handle odd sizes
                out[i] = cropped[:H, :W]
            else:
                # Zoom-out: pad with edge values
                pad_y = (H - zh) // 2
                pad_x = (W - zw) // 2
                canvas = np.zeros((H, W), dtype=np.float32)
                canvas[pad_y:pad_y + zh, pad_x:pad_x + zw] = zoomed
                out[i] = canvas

        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _gaussian_noise(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Additive Gaussian noise — simulates scanner thermal noise."""
        sigma = rng.uniform(0.005, self.noise_std)
        noise = rng.normal(0.0, sigma, slices.shape).astype(np.float32)
        return np.clip(slices + noise, 0.0, 1.0)

    def _brightness_contrast(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Independent brightness and contrast jitter per slice."""
        out = slices.copy()
        for i in range(3):
            alpha = rng.uniform(0.85, 1.15)   # contrast
            beta  = rng.uniform(-0.15, 0.15)  # brightness
            out[i] = np.clip(alpha * out[i] + beta, 0.0, 1.0).astype(np.float32)
        return out

    def _gamma(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """Gamma correction — models MRI T2w signal range variation."""
        gamma = rng.uniform(0.7, 1.4)
        return np.power(np.clip(slices, 1e-8, 1.0), gamma).astype(np.float32)

    def _bias_field(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate MRI bias field: smooth low-frequency multiplicative
        intensity gradient (polynomial degree 2).
        """
        H, W = slices.shape[1], slices.shape[2]
        yy, xx = np.meshgrid(
            np.linspace(-1, 1, H, dtype=np.float32),
            np.linspace(-1, 1, W, dtype=np.float32),
            indexing="ij",
        )
        coeffs = rng.uniform(-0.15, 0.15, 6).astype(np.float32)
        bias   = (1.0
                  + coeffs[0] * xx
                  + coeffs[1] * yy
                  + coeffs[2] * xx ** 2
                  + coeffs[3] * yy ** 2
                  + coeffs[4] * xx * yy
                  + coeffs[5] * (xx ** 2 + yy ** 2))
        # Normalise bias to stay near 1
        bias = bias / bias.mean()
        out  = slices * bias[np.newaxis, :, :]
        return np.clip(out, 0.0, 1.0).astype(np.float32)

    def _gibbs_ringing(self, slices: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        """
        Simulate Gibbs ringing by truncating the k-space (2-D FFT) at a
        random cutoff, then reconstructing.  This adds the ringing artefact
        seen in undersampled MRI acquisitions.
        """
        cutoff = rng.uniform(0.60, 0.90)
        out    = np.zeros_like(slices)
        for i in range(3):
            kspace   = np.fft.fft2(slices[i])
            kspace   = np.fft.fftshift(kspace)
            H, W     = kspace.shape
            kH, kW   = int(H * cutoff) // 2, int(W * cutoff) // 2
            mask     = np.zeros_like(kspace, dtype=bool)
            cy, cx   = H // 2, W // 2
            mask[cy - kH:cy + kH, cx - kW:cx + kW] = True
            kspace[~mask] = 0
            kspace   = np.fft.ifftshift(kspace)
            recon    = np.real(np.fft.ifft2(kspace)).astype(np.float32)
            out[i]   = np.clip(recon, 0.0, 1.0)
        return out

    def _myelination_delay(
        self,
        slices: np.ndarray,
        dq: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        White-matter blurring proportional to ID severity.
        Lower DQ → stronger blur (more delayed myelination).
        """
        # Severity: 0 = typical, 1 = profound
        severity = max(0.0, (85.0 - dq) / 85.0) * rng.uniform(0.5, 1.0)
        if severity < 0.05:
            return slices

        sigma = severity * rng.uniform(1.5, 3.5)
        out   = slices.copy()
        for i in range(3):
            blurred = gaussian_filter(out[i], sigma=sigma).astype(np.float32)
            wm_mask = out[i] > 0.6
            out[i]  = np.where(wm_mask, blurred, out[i])
        return out

    # ------------------------------------------------------------------
    # Main augment call
    # ------------------------------------------------------------------

    def augment(
        self,
        slices: np.ndarray,
        dq: float,
        rng: np.random.Generator,
    ) -> np.ndarray:
        """
        Apply a randomly-selected pipeline of augmentation ops to one
        slice stack.  The combination is chosen stochastically each call
        ensuring diverse outputs.

        Parameters
        ----------
        slices : (3, H, W) float32 normalised [0, 1]
        dq     : current DQ label (used for myelination-delay severity)
        rng    : numpy random Generator

        Returns
        -------
        (3, H, W) float32 augmented slices
        """
        aug = slices.copy()

        # — Geometric (applied first, before intensity) —
        # Always apply at least one geometric op
        ops_geo = rng.choice(
            ["flip", "rotate", "elastic", "zoom"],
            size=rng.integers(1, 4),
            replace=False,
        )
        for op in ops_geo:
            if op == "flip":
                aug = self._flip(aug, rng)
            elif op == "rotate":
                aug = self._rotate(aug, rng)
            elif op == "elastic":
                aug = self._elastic(aug, rng)
            elif op == "zoom":
                aug = self._zoom_crop(aug, rng)

        # — Intensity —
        ops_int = rng.choice(
            ["noise", "brightness", "gamma", "bias", "gibbs"],
            size=rng.integers(1, 4),
            replace=False,
        )
        for op in ops_int:
            if op == "noise":
                aug = self._gaussian_noise(aug, rng)
            elif op == "brightness":
                aug = self._brightness_contrast(aug, rng)
            elif op == "gamma":
                aug = self._gamma(aug, rng)
            elif op == "bias":
                aug = self._bias_field(aug, rng)
            elif op == "gibbs":
                aug = self._gibbs_ringing(aug, rng)

        # — MRI-specific: myelination delay (always applied for ID classes) —
        if dq < 85.0 or rng.random() < 0.20:
            aug = self._myelination_delay(aug, dq, rng)

        return aug


# ---------------------------------------------------------------------------
# Main generation function
# ---------------------------------------------------------------------------

def generate_augmented_dataset(
    real_dir:   Path,
    output_dir: Path,
    target_n:   int   = 10_000,
    seed:       int   = 42,
    dq_noise_std: float = 5.0,
    slice_size: int   = 64,
    verbose:    bool  = True,
) -> None:
    """
    Generate *target_n* synthetic MRI slice stacks from real .npz files.

    Strategy
    --------
    1. Load all real samples from *real_dir*.
    2. Compute how many samples per class are needed to hit *target_n*
       following TARGET_DIST.
    3. For each sample to generate, pick a source subject (cycling through
       real subjects that belong to or are closest to the target class),
       perturb its DQ to land within the class range, run augmentations.
    4. Save as {idx:06d}.npz in *output_dir*.

    Parameters
    ----------
    real_dir      : path to datasets/processed/mri (real .npz files)
    output_dir    : path to write synthetic .npz files
    target_n      : total synthetic samples to generate
    seed          : random seed for full reproducibility
    dq_noise_std  : std for DQ perturbation around source DQ
    slice_size    : expected H=W dimension of slices (for sanity check)
    verbose       : show tqdm progress bar if tqdm is available
    """
    real_dir   = Path(real_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng     = np.random.default_rng(seed)
    augmenter = MRISliceAugmenter()

    # ------------------------------------------------------------------
    # 1. Load all real samples
    # ------------------------------------------------------------------
    real_files = sorted(real_dir.glob("*.npz"))
    if not real_files:
        raise FileNotFoundError(f"No real .npz files found in {real_dir}. "
                                f"Run mri_loader.process_mri_dataset first.")

    real_samples = []
    for f in real_files:
        d = np.load(f, allow_pickle=True)
        real_samples.append({
            "slices": d["slices"].astype(np.float32),
            "dq":     float(d["dq"]),
            "label":  int(d["label"]),
        })

    n_real = len(real_samples)
    log.info("Loaded %d real MRI samples", n_real)
    if verbose:
        print(f"[MRI Balancing] Loaded {n_real} real samples from {real_dir}")
        print(f"[MRI Balancing] Target: {target_n:,} synthetic samples")
        print(f"[MRI Balancing] Output: {output_dir}")

    # ------------------------------------------------------------------
    # 2. Per-class quota
    # ------------------------------------------------------------------
    class_quotas = {cls: max(1, round(frac * target_n))
                    for cls, frac in TARGET_DIST.items()}
    # Adjust for rounding drift so total == target_n exactly
    total_assigned = sum(class_quotas.values())
    diff = target_n - total_assigned
    class_quotas[0] += diff   # absorb rounding into largest class

    if verbose:
        print("\n[MRI Balancing] Class quotas:")
        class_names = ["Typical", "Borderline", "Mild ID", "Moderate ID", "Severe ID", "Profound ID"]
        for cls, quota in class_quotas.items():
            lo, hi = _CLASS_DQ_RANGES[cls]
            print(f"  Class {cls} ({class_names[cls]:12s}, DQ {lo:.0f}–{hi:.0f}): {quota:,} samples")
        print()

    # Build LUT: for each class, which real samples are closest in DQ?
    # We allow any real sample to serve as source for any class (we'll
    # re-project its DQ into the target class range).
    def _sources_for_class(cls: int):
        """Real samples sorted by DQ proximity to target class centre."""
        lo, hi = _CLASS_DQ_RANGES[cls]
        centre = (lo + hi) / 2.0
        by_dist = sorted(real_samples, key=lambda s: abs(s["dq"] - centre))
        return by_dist

    # ------------------------------------------------------------------
    # 3. Generate samples
    # ------------------------------------------------------------------
    try:
        from tqdm import tqdm  # type: ignore
        _tqdm = tqdm
    except ImportError:
        _tqdm = None

    global_idx = 0
    class_counts: dict = {i: 0 for i in range(6)}

    for cls, quota in class_quotas.items():
        sources = _sources_for_class(cls)
        lo, hi  = _CLASS_DQ_RANGES[cls]

        iter_range = range(quota)
        if verbose and _tqdm is not None:
            class_names = ["Typical", "Borderline", "Mild ID", "Moderate ID", "Severe ID", "Profound ID"]
            iter_range = _tqdm(iter_range, desc=f"  Class {cls} {class_names[cls]:12s}", leave=True)

        for i in iter_range:
            # Pick source cyclically (ensuring all real subjects contribute)
            src = sources[i % len(sources)]

            # Project DQ into target class range with small noise
            dq_target = _sample_dq_for_class(cls, rng)
            # Blend source DQ with target: 50% from source neighbourhood, 50% class-guided
            dq_blend = 0.5 * np.clip(src["dq"] + rng.normal(0, dq_noise_std), lo, hi) \
                     + 0.5 * dq_target
            dq_final = float(np.clip(dq_blend, lo, hi))
            label    = _dq_to_label(dq_final)

            # Augment slices
            aug_slices = augmenter.augment(src["slices"], dq_final, rng)

            # Save
            out_path = output_dir / f"{global_idx:06d}.npz"
            np.savez_compressed(
                out_path,
                slices    = aug_slices,
                dq        = np.float32(dq_final),
                label     = np.int32(label),
                source_id = np.bytes_(f"syn_class{cls}_src{i % n_real}"),
                synthetic = np.bool_(True),
            )
            global_idx  += 1
            class_counts[cls] += 1

    # ------------------------------------------------------------------
    # 4. Summary
    # ------------------------------------------------------------------
    if verbose:
        print(f"\n[MRI Balancing] ✅  Done! Generated {global_idx:,} samples.\n")
        print("[MRI Balancing] Final class distribution:")
        class_names = ["Typical", "Borderline", "Mild ID", "Moderate ID", "Severe ID", "Profound ID"]
        for cls, cnt in class_counts.items():
            bar = "█" * int(cnt / target_n * 40)
            pct = cnt / target_n * 100
            print(f"  {class_names[cls]:12s}: {cnt:6,}  ({pct:5.1f}%)  {bar}")

    log.info("Augmented dataset complete: %d samples in %s", global_idx, output_dir)
