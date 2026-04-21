"""
api/preprocessing.py
MNE-based EDF preprocessing and nibabel-based NIfTI preprocessing.
"""
from __future__ import annotations

from pathlib import Path
from typing import Tuple

import mne
import nibabel as nib
import numpy as np
import scipy.signal

mne.set_log_level("WARNING")


def preprocess_edf(edf_path: str) -> Tuple[int, int, int, float, np.ndarray, str]:
    raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
    sfreq = int(raw.info["sfreq"])
    n_channels = len(raw.ch_names)
    duration_sec = float(raw.times[-1]) + 1.0 / sfreq

    raw_filt = raw.copy()
    raw_filt.load_data()
    raw_filt.filter(l_freq=0.5, h_freq=40.0, fir_design="firwin")
    try:
        raw_filt.notch_filter(freqs=50.0, fir_design="firwin")
    except Exception:
        pass

    epochs, meta = _epoch_raw(raw_filt, sfreq=sfreq)
    n_epochs = epochs.shape[0]
    features = _extract_eeg_features(raw_filt, epochs)
    summary = _build_epoch_summary(n_epochs, n_channels, sfreq, epochs, meta)

    return n_epochs, n_channels, sfreq, duration_sec, features, summary


def _epoch_raw(raw: mne.io.Raw, sfreq: int, epoch_sec: int = 30, overlap: float = 0.5):
    total_samples = raw.n_times
    epoch_samples = int(epoch_sec * sfreq)
    step = int(epoch_samples * (1 - overlap))
    epochs, meta = [], []
    t_start = 0
    while t_start + epoch_samples <= total_samples:
        data, _ = raw[:, t_start:t_start + epoch_samples]
        epochs.append(data)
        meta.append({"start_s": t_start / sfreq, "end_s": (t_start + epoch_samples) / sfreq})
        t_start += step
    if not epochs:
        return np.empty((0, raw.n_channels, epoch_samples)), meta
    return np.stack(epochs).astype(np.float32), meta


def _extract_eeg_features(raw: mne.io.Raw, epochs: np.ndarray) -> np.ndarray:
    delta_band = (0.5, 4)
    theta_band = (4, 8)
    alpha_band = (8, 13)
    beta_band = (13, 30)

    def band_power(data, sfreq, lo, hi):
        freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=min(1024, data.shape[-1]))
        idx = np.logical_and(freqs >= lo, freqs <= hi)
        return float(np.mean(psd[idx])) if idx.sum() > 0 else 0.0

    if epochs.ndim == 3:
        data = np.mean(epochs, axis=0)
    else:
        data = raw.get_data()
    if data.shape[0] > 19:
        data = data[:19, :]

    features = []
    for ch in range(min(data.shape[0], 19)):
        d = np.nanmean(data[ch:ch+1], axis=0)
        dp = band_power(d, raw.info["sfreq"], *delta_band)
        tp = band_power(d, raw.info["sfreq"], *theta_band)
        ap = band_power(d, raw.info["sfreq"], *alpha_band)
        bp = band_power(d, raw.info["sfreq"], *beta_band)
        total_power = float(np.mean(np.var(d, axis=-1)))
        bsr_val = _burst_suppression_ratio(d)
        ibi_mean, ibi_std = _inter_burst_interval(d)
        sef95_val = _spectral_edge_freq(d, raw.info["sfreq"])
        amp_mean = float(np.mean(np.abs(d)))
        amp_std = float(np.std(d))
        features.append([dp, tp, ap, bp, total_power, bsr_val, ibi_mean, ibi_std, sef95_val, amp_mean, amp_std])

    return np.mean(features, axis=0) if features else np.zeros(11)


def _burst_suppression_ratio(data: np.ndarray) -> float:
    norm = np.abs(data) / (np.std(data) + 1e-8)
    return float(np.mean(norm < 0.3))


def _inter_burst_interval(data: np.ndarray) -> Tuple[float, float]:
    norm = np.abs(data) / (np.std(data) + 1e-8)
    above = norm >= 0.3
    transitions = np.diff(above.astype(int))
    onsets = np.where(transitions == 1)[0]
    offsets = np.where(transitions == -1)[0]
    if offsets.size > 0 and offsets[0] < onsets[0]:
        offsets = offsets[1:]
    if onsets.size > 0 and offsets.size > 0 and offsets[0] < onsets[-1]:
        onsets = onsets[:-1]
    if len(onsets) > 0 and len(offsets) > 0:
        ibi = np.diff(np.vstack([onsets, offsets]), axis=0)[0] / 256.0
        return float(np.mean(ibi)), float(np.std(ibi)) if len(ibi) > 1 else 0.0
    return 0.0, 0.0


def _spectral_edge_freq(data: np.ndarray, sfreq: int, percentile: float = 95) -> float:
    freqs, psd = scipy.signal.welch(data, fs=sfreq, nperseg=min(1024, len(data)))
    cumsum = np.cumsum(psd)
    idx = np.searchsorted(cumsum, percentile / 100 * cumsum[-1])
    return float(freqs[min(idx, len(freqs) - 1)])


def _build_epoch_summary(n_epochs: int, n_channels: int, sfreq: int, epochs: np.ndarray, meta: list) -> str:
    if n_epochs == 0:
        return "No epochs extracted. Recording too short (< 30s)."
    lines = [
        f"Extracted {n_epochs} epochs of 30s (50% overlap) from a {n_channels}-channel recording at {sfreq}Hz.",
        f"Epoch tensor shape: {epochs.shape}.",
    ]
    if meta:
        lines.append(f"Recording covers: {meta[0]['start_s']:.1f}s to {meta[-1]['end_s']:.1f}s.")
    return " ".join(lines)


def preprocess_nifti(nii_path: str) -> Tuple[str, list, int, list, str]:
    nii = nib.load(nii_path)
    data = nii.get_fdata()
    shape = list(data.shape[:3])
    subject_id = Path(nii_path).stem.split(".")[0]

    D, H, W = data.shape[:3]
    axial = _normalize_slice(data[D // 2, :, :])
    coronal = _normalize_slice(data[:, H // 2, :])
    sagittal = _normalize_slice(data[:, :, W // 2])
    slices = np.stack([axial, coronal, sagittal], axis=0).astype(np.float32)
    myelination_note = _assess_myelination(slices)

    return subject_id, shape, 3, list(slices[0].shape), myelination_note


def _normalize_slice(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    mn, mx = img.min(), img.max()
    if mx - mn < 1e-8:
        return np.zeros_like(img)
    return (img - mn) / (mx - mn)


def _assess_myelination(slices: np.ndarray) -> str:
    total_contrast = 0.0
    for sl in slices:
        grad = np.gradient(sl)
        contrast = np.mean([np.std(g) for g in grad])
        total_contrast += contrast
    avg = total_contrast / max(len(slices), 1)
    if avg > 0.12:
        return "Good white/gray matter contrast — likely age-appropriate myelination."
    elif avg > 0.06:
        return "Moderate contrast — possible mild myelination delay. Clinical correlation recommended."
    return "Low contrast — possible myelination delay. Clinical correlation and formal neuroradiology review recommended."
