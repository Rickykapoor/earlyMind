"""
src/data/eeg_loader.py
EDF loading + MNE-based feature extraction for Helsinki Neonatal EEG dataset.
Outputs both tabular feature vectors and raw epoched arrays per subject.
"""
from __future__ import annotations

import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mne
import numpy as np
import pandas as pd

from src.config import cfg
from src.utils.label_utils import parse_eeg_clinical_csv, add_seizure_labels

warnings.filterwarnings("ignore", category=RuntimeWarning)
mne.set_log_level("WARNING")


# ---------------------------------------------------------------------------
# Constants (from params.yaml via cfg)
# ---------------------------------------------------------------------------

SFREQ          = cfg.data.eeg_sample_rate       # 256 Hz
EPOCH_SEC      = cfg.data.eeg_epoch_seconds      # 30 s
OVERLAP        = cfg.data.eeg_epoch_overlap       # 0.5
BANDPASS_LOW   = cfg.data.eeg_bandpass_low        # 0.5 Hz
BANDPASS_HIGH  = cfg.data.eeg_bandpass_high       # 40.0 Hz
NOTCH_FREQ     = cfg.data.eeg_notch_freq          # 50 Hz
N_CHANNELS     = cfg.model.eeg_channels           # 19
EPOCH_SAMPLES  = cfg.model.eeg_timesteps          # 7680 = 30s × 256Hz

BAND_FREQS = {
    "delta": (0.5, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 13.0),
    "beta":  (13.0, 30.0),
}


# ---------------------------------------------------------------------------
# Core helpers
# ---------------------------------------------------------------------------

def load_edf(edf_path: str | Path, target_sfreq: int = SFREQ) -> mne.io.RawArray:
    """Load .edf and resample to target_sfreq. Returns MNE Raw object."""
    raw = mne.io.read_raw_edf(str(edf_path), preload=True, verbose=False)
    if int(raw.info["sfreq"]) != target_sfreq:
        raw.resample(target_sfreq, verbose=False)
    return raw


def preprocess_raw(raw: mne.io.RawArray) -> mne.io.RawArray:
    """Apply bandpass + notch filters in place."""
    raw.filter(BANDPASS_LOW, BANDPASS_HIGH, fir_design="firwin", verbose=False)
    raw.notch_filter(NOTCH_FREQ, verbose=False)
    return raw


def epoch_raw(raw: mne.io.RawArray) -> np.ndarray:
    """
    Epoch raw signal into overlapping windows.

    Returns
    -------
    epochs : np.ndarray, shape (n_epochs, n_channels, EPOCH_SAMPLES)
    """
    data = raw.get_data()          # (n_channels, n_times)
    n_ch, n_times = data.shape

    step = int(EPOCH_SEC * SFREQ * (1.0 - OVERLAP))   # 50% overlap
    epoch_len = EPOCH_SAMPLES                           # 7680 samples

    epochs = []
    start = 0
    while start + epoch_len <= n_times:
        chunk = data[:, start : start + epoch_len]     # (n_ch, epoch_len)
        # Pad/trim channels if needed
        if n_ch < N_CHANNELS:
            pad = np.zeros((N_CHANNELS - n_ch, epoch_len))
            chunk = np.vstack([chunk, pad])
        else:
            chunk = chunk[:N_CHANNELS, :]
        epochs.append(chunk)
        start += step

    if len(epochs) == 0:
        # Single epoch: pad if recording is shorter than 30 s
        chunk = data[:N_CHANNELS, :]
        pad_len = epoch_len - chunk.shape[1]
        if pad_len > 0:
            chunk = np.hstack([chunk, np.zeros((N_CHANNELS, pad_len))])
        else:
            chunk = chunk[:, :epoch_len]
        epochs = [chunk]

    return np.array(epochs, dtype=np.float32)   # (n_epochs, 19, 7680)


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

def _band_power(epoch_1d: np.ndarray, band: Tuple[float, float], sfreq: int) -> float:
    """
    Estimate bandpower of a 1D signal using Welch's method.
    Returns power in µV²/Hz.
    """
    from scipy.signal import welch
    fmin, fmax = band
    freq, psd = welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
    idx = np.logical_and(freq >= fmin, freq <= fmax)
    return float(np.trapz(psd[idx], freq[idx]))


def _burst_suppression_ratio(epoch_1d: np.ndarray, threshold_uv: float = 5.0) -> float:
    """
    Burst-Suppression Ratio: fraction of samples with amplitude < threshold_uv.
    Input is assumed to be in µV.
    """
    suppressed = np.abs(epoch_1d) < threshold_uv
    return float(suppressed.mean())


def _inter_burst_intervals(epoch_1d: np.ndarray, sfreq: int, threshold_uv: float = 5.0) -> Tuple[float, float]:
    """
    Compute mean and std of inter-burst intervals in seconds.
    A 'burst' is a contiguous region where |signal| >= threshold_uv.
    """
    above = (np.abs(epoch_1d) >= threshold_uv).astype(int)
    diff = np.diff(above)
    burst_starts = np.where(diff == 1)[0]
    burst_ends   = np.where(diff == -1)[0]

    if len(burst_starts) < 2:
        return (0.0, 0.0)

    # Align starts and ends
    if burst_ends[0] < burst_starts[0]:
        burst_ends = burst_ends[1:]
    min_len = min(len(burst_starts), len(burst_ends))
    burst_starts = burst_starts[:min_len]
    burst_ends   = burst_ends[:min_len]

    ibi_samples = np.diff(burst_starts)
    ibi_sec = ibi_samples / sfreq
    return (float(ibi_sec.mean()), float(ibi_sec.std()))


def _spectral_edge_freq(epoch_1d: np.ndarray, sfreq: int, percentile: float = 95.0) -> float:
    """
    SEF95: frequency below which `percentile`% of total power lies.
    """
    from scipy.signal import welch
    freq, psd = welch(epoch_1d, fs=sfreq, nperseg=min(256, len(epoch_1d)))
    cumulative = np.cumsum(psd)
    total = cumulative[-1]
    if total <= 0:
        return 0.0
    idx = np.searchsorted(cumulative, total * (percentile / 100.0))
    idx = min(idx, len(freq) - 1)
    return float(freq[idx])


def extract_epoch_features(epoch: np.ndarray, sfreq: int = SFREQ) -> np.ndarray:
    """
    Extract per-channel features for one epoch (shape: n_channels × n_samples).
    Returns a 1D feature vector by averaging across channels where appropriate.

    Feature vector layout (per epoch):
        delta_mean, theta_mean, alpha_mean, beta_mean,
        total_power_mean, bsr_mean, ibi_mean, ibi_std,
        sef95_mean, amp_mean_mean, amp_std_mean
    → 11 scalar features per epoch
    """
    n_ch = epoch.shape[0]

    delta_list, theta_list, alpha_list, beta_list = [], [], [], []
    total_pw_list, bsr_list, ibi_mean_list, ibi_std_list = [], [], [], []
    sef95_list, amp_mean_list, amp_std_list = [], [], []

    for ch in range(n_ch):
        sig = epoch[ch]

        delta_list.append(_band_power(sig, BAND_FREQS["delta"], sfreq))
        theta_list.append(_band_power(sig, BAND_FREQS["theta"], sfreq))
        alpha_list.append(_band_power(sig, BAND_FREQS["alpha"], sfreq))
        beta_list.append(_band_power(sig, BAND_FREQS["beta"], sfreq))

        total_pw = (
            _band_power(sig, (BANDPASS_LOW, BANDPASS_HIGH), sfreq)
        )
        total_pw_list.append(total_pw)

        bsr_list.append(_burst_suppression_ratio(sig))
        im, is_ = _inter_burst_intervals(sig, sfreq)
        ibi_mean_list.append(im)
        ibi_std_list.append(is_)

        sef95_list.append(_spectral_edge_freq(sig, sfreq))
        amp_mean_list.append(float(np.mean(np.abs(sig))))
        amp_std_list.append(float(np.std(sig)))

    features = np.array([
        np.mean(delta_list),
        np.mean(theta_list),
        np.mean(alpha_list),
        np.mean(beta_list),
        np.mean(total_pw_list),
        np.mean(bsr_list),
        np.mean(ibi_mean_list),
        np.mean(ibi_std_list),
        np.mean(sef95_list),
        np.mean(amp_mean_list),
        np.mean(amp_std_list),
    ], dtype=np.float32)

    return features  # (11,)


def extract_subject_features(epochs: np.ndarray) -> np.ndarray:
    """
    Aggregate per-epoch features into a subject-level feature vector.
    epochs : (n_epochs, 19, 7680)
    Returns: (11,) float32 vector — mean across epochs
    """
    per_epoch = [extract_epoch_features(epochs[i]) for i in range(len(epochs))]
    return np.mean(per_epoch, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Dataset-level preprocessing
# ---------------------------------------------------------------------------

def process_eeg_dataset(
    eeg_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, dict]:
    """
    Process all EDF files in eeg_dir.
    Saves per subject:
        output_dir/{subject}_features.npy  — (11,) tabular features
        output_dir/{subject}_epochs.npy    — (n_epochs, 19, 7680) raw epochs

    Also attempts to load labels from clinical_information.csv.

    Returns dict: {subject_id: {"features": ..., "epochs": ..., "label": ..., "dq": ...}}
    """
    eeg_dir = Path(eeg_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    edf_files = sorted(eeg_dir.glob("*.edf"))
    if len(edf_files) == 0:
        raise FileNotFoundError(f"No .edf files found in {eeg_dir}")

    # Load clinical labels if available
    label_df = None
    clinical_csv = eeg_dir / "clinical_information.csv"
    annotations_csv = eeg_dir / "annotations_2017_A.csv"

    if clinical_csv.exists():
        try:
            label_df = parse_eeg_clinical_csv(str(clinical_csv))
            if annotations_csv.exists():
                label_df = add_seizure_labels(label_df, str(annotations_csv))
        except Exception as e:
            print(f"  [WARNING] Could not parse clinical CSV: {e}")

    results = {}

    for edf_path in edf_files:
        subject_id = edf_path.stem  # e.g. "1", "2", "3"
        print(f"  Processing EEG subject: {subject_id}")

        try:
            raw = load_edf(edf_path)
            raw = preprocess_raw(raw)
            epochs = epoch_raw(raw)       # (n_epochs, 19, 7680)
            feats  = extract_subject_features(epochs)  # (11,)
        except Exception as e:
            print(f"    [ERROR] Failed to process {edf_path.name}: {e}")
            continue

        # Save
        feat_path   = output_dir / f"{subject_id}_features.npy"
        epoch_path  = output_dir / f"{subject_id}_epochs.npy"
        np.save(str(feat_path),  feats)
        np.save(str(epoch_path), epochs)

        # Attach labels
        label, dq = 0, 85.0
        if label_df is not None:
            match = label_df[label_df["subject_id"].str.contains(subject_id, case=False)]
            if len(match) > 0:
                label = int(match.iloc[0]["label"])
                dq    = float(match.iloc[0]["dq"])

        results[subject_id] = {
            "features": feats,
            "epochs": epochs,
            "label": label,
            "dq": dq,
            "feat_path": str(feat_path),
            "epoch_path": str(epoch_path),
        }

    print(f"  EEG preprocessing complete: {len(results)} subjects → {output_dir}")
    return results


# ---------------------------------------------------------------------------
# EEG Augmentation helpers (for training)
# ---------------------------------------------------------------------------

def augment_eeg_epochs(
    epochs: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Apply data augmentation to a batch of EEG epochs.
    epochs : (B, 19, T) or (19, T) for single epoch
    Returns augmented copy, same shape.
    """
    if rng is None:
        rng = np.random.default_rng()

    x = epochs.copy().astype(np.float32)
    single = x.ndim == 2
    if single:
        x = x[np.newaxis]   # (1, 19, T)

    B, C, T = x.shape

    # 1. Gaussian noise
    x += rng.normal(0, 0.01, size=x.shape).astype(np.float32)

    # 2. Channel dropout (zero 2–3 channels per sample)
    n_drop = rng.integers(2, 4)
    drop_ch = rng.choice(C, size=n_drop, replace=False)
    x[:, drop_ch, :] = 0.0

    # 3. Time shift ±50 samples
    shift = int(rng.integers(-50, 51))
    if shift > 0:
        x = np.concatenate([np.zeros((B, C, shift), dtype=np.float32), x[:, :, :-shift]], axis=2)
    elif shift < 0:
        x = np.concatenate([x[:, :, -shift:], np.zeros((B, C, -shift), dtype=np.float32)], axis=2)

    # 4. Amplitude scaling
    scale = rng.uniform(0.8, 1.2, size=(B, 1, 1)).astype(np.float32)
    x = x * scale

    if single:
        x = x[0]
    return x
