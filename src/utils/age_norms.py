"""
src/utils/age_norms.py
DQ computation, age normalization, and age-band utilities.
All severity references use DQ (Developmental Quotient), never IQ.
"""
from __future__ import annotations

from typing import Optional, Tuple


# ---------------------------------------------------------------------------
# DQ calculation
# ---------------------------------------------------------------------------

def compute_dq(developmental_age_months: float, chronological_age_months: float) -> float:
    """
    Developmental Quotient = (Developmental Age / Chronological Age) × 100.

    Valid range: 0–36 months (Bayley-4, ASQ-3, Vineland-3, DAYC-2).
    Returns 100.0 if chronological_age_months == 0 (avoid division by zero).
    """
    if chronological_age_months <= 0:
        return 100.0
    dq = (developmental_age_months / chronological_age_months) * 100.0
    return float(max(0.0, min(100.0, dq)))


def compute_corrected_age(
    chronological_age_months: float,
    gestational_age_weeks: float,
    full_term_weeks: float = 40.0,
) -> float:
    """
    Corrected age = chronological age − prematurity correction.
    Prematurity in months = (full_term_weeks − ga_weeks) / 4.33.
    Clamped at 0 (cannot be negative).
    """
    prematurity_months = max(0.0, (full_term_weeks - gestational_age_weeks) / 4.33)
    corrected = max(0.0, chronological_age_months - prematurity_months)
    return float(corrected)


# ---------------------------------------------------------------------------
# DQ severity bands
# ---------------------------------------------------------------------------

DQ_BANDS = {
    "typical":    (85, 100),
    "borderline": (70, 85),
    "mild":       (55, 70),
    "moderate":   (35, 55),
    "severe":     (20, 35),
    "profound":   (0,  20),
}


def dq_to_label(dq: float) -> str:
    """Return string severity label for a DQ score."""
    for label, (low, high) in DQ_BANDS.items():
        if low <= dq < high:
            return label
    if dq >= 100:
        return "typical"
    return "profound"


def dq_to_risk_flag(dq: float) -> int:
    """Return binary risk flag: 1 if DQ < 85 (borderline or worse), else 0."""
    return int(dq < 85)


def dq_to_id_flag(dq: float) -> int:
    """Return binary ID flag: 1 if DQ < 70 (mild ID or worse), else 0."""
    return int(dq < 70)


# ---------------------------------------------------------------------------
# Age band classification
# ---------------------------------------------------------------------------

AGE_BANDS = [
    ("neonate",  0,   3),
    ("infant",   3,  12),
    ("toddler1", 12, 24),
    ("toddler2", 24, 36),
]


def age_to_band(age_months: float) -> str:
    """
    Return the developmental age band label for a given age in months.
    Clips ages outside 0–36 months to the nearest valid band.
    """
    for band_name, low, high in AGE_BANDS:
        if low <= age_months < high:
            return band_name
    if age_months < 0:
        return "neonate"
    return "toddler2"  # 36+ months


def age_band_limits(band: str) -> Tuple[int, int]:
    """Return (low_months, high_months) for a named age band."""
    for name, low, high in AGE_BANDS:
        if name == band:
            return (low, high)
    raise ValueError(f"Unknown age band: {band!r}. Valid: {[b[0] for b in AGE_BANDS]}")


# ---------------------------------------------------------------------------
# Age normalization for model input
# ---------------------------------------------------------------------------

AGE_NORM_MEAN = 18.0   # months — midpoint of 0–36M range
AGE_NORM_STD  = 10.39  # std of uniform(0, 36) distribution


def normalize_age(age_months: float) -> float:
    """Z-score normalize age in months for use as a model feature."""
    return (age_months - AGE_NORM_MEAN) / AGE_NORM_STD


def denormalize_age(z: float) -> float:
    """Convert z-score back to months."""
    return z * AGE_NORM_STD + AGE_NORM_MEAN


# ---------------------------------------------------------------------------
# DQ-range sampling helpers (used by label_utils.py)
# ---------------------------------------------------------------------------

import numpy as np


def sample_dq(
    label: int,
    hie_grade: Optional[int] = None,
    has_seizures: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sample a DQ value based on risk label and HIE grade.

    Parameters
    ----------
    label       : 0 = typical, 1 = ID risk
    hie_grade   : 0, 1, 2, 3 (None if not applicable)
    has_seizures: whether seizures were annotated
    rng         : numpy random Generator (None = default_rng)

    Returns
    -------
    float : DQ in [0, 100]
    """
    if rng is None:
        rng = np.random.default_rng()

    if label == 0:
        dq = rng.uniform(80, 100)
    else:
        if hie_grade is None or hie_grade == 0:
            dq = rng.uniform(55, 75)
        elif hie_grade == 1:
            dq = rng.uniform(55, 75)
        elif hie_grade == 2:
            dq = rng.uniform(35, 55)
        else:
            dq = rng.uniform(15, 35)

        if has_seizures:
            dq = max(0.0, dq - 10.0)

    return float(np.clip(dq, 0.0, 100.0))
