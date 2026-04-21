"""
src/utils/label_utils.py
Dataset-specific label converters for all three modalities.
Converts raw clinical annotations to (label, dq) pairs.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from src.utils.age_norms import sample_dq


# ---------------------------------------------------------------------------
# EEG / Helsinki Neonatal — HIE-grade → (label, dq)
# ---------------------------------------------------------------------------

def hie_grade_to_label_dq(
    hie_grade: Optional[int],
    has_seizures: bool = False,
    rng: Optional[np.random.Generator] = None,
) -> tuple:
    """
    Map HIE grade integer (0–3) to binary risk label and a sampled DQ.

    HIE grade 0 or None → label 0 (typical)
    HIE grades 1–3      → label 1 (ID risk)
    Seizures add extra DQ penalty (handled in sample_dq).
    """
    if rng is None:
        rng = np.random.default_rng()

    label = 0 if (hie_grade is None or hie_grade == 0) else 1
    dq = sample_dq(label, hie_grade=hie_grade, has_seizures=has_seizures, rng=rng)
    return label, dq


def _parse_hie_grade_from_text(diagnosis_text: str) -> Optional[int]:
    """
    Parse HIE grade (0–3) from free-text Diagnosis column.

    Helsinki Neonatal dataset uses text like:
      'mild/moderate asphyxia', 'severe asphyxia', 'prematurity',
      'infarction', 'neonatal convulsions', NaN, etc.

    Grade mapping:
      None : unknown / NaN — treated as grade 0 by caller
      1    : mild asphyxia, prematurity
      2    : mild/moderate asphyxia, infarction, haemorrhage, neonatal convulsions
      3    : severe asphyxia, diffuse edema
    """
    if not diagnosis_text or str(diagnosis_text).strip().lower() in ("nan", "none", ""):
        return None  # unknown

    diag = str(diagnosis_text).lower().strip()

    if "severe asphyxia" in diag or "diffuse edema" in diag:
        return 3
    if "mild/moderate" in diag or "moderate asphyxia" in diag:
        return 2
    if "mild asphyxia" in diag:
        return 1
    if "infarction" in diag or "haemorrhage" in diag or "hemorrhage" in diag:
        return 2
    if "neonatal convulsions" in diag:
        return 2
    if "prematurity" in diag:
        return 1
    if "asphyxia" in diag:
        return 1
    return None


def parse_eeg_clinical_csv(clinical_csv_path: str) -> pd.DataFrame:
    """
    Load clinical_information.csv from Helsinki Neonatal dataset.

    Actual columns: ID | EEG file | Gender | BW (g) | GA (weeks) | ... | Diagnosis | ...
    Returns standardised DataFrame:
      subject_id (str), eeg_file (str, e.g. 'eeg10'), hie_grade (int|None), label (int), dq (float)
    """
    df = pd.read_csv(clinical_csv_path)
    df.columns = df.columns.str.strip()

    col_lower = {c.lower().strip(): c for c in df.columns}

    # Subject numeric ID column
    id_col = col_lower.get("id", df.columns[0])

    # EEG filename column — 'EEG file' in Helsinki dataset
    eeg_file_col = (
        col_lower.get("eeg file") or
        col_lower.get("eeg_file") or
        None
    )

    # Diagnosis/grade column
    diag_col = (
        col_lower.get("diagnosis") or
        col_lower.get("hie_grade") or
        col_lower.get("grade") or
        col_lower.get("outcome") or
        col_lower.get("severity") or
        None
    )

    rng = np.random.default_rng(seed=42)
    records = []

    for _, row in df.iterrows():
        subj_id = str(row[id_col]).strip()

        eeg_file = ""
        if eeg_file_col:
            eeg_file = str(row[eeg_file_col]).strip().lower()

        diag_text = str(row[diag_col]) if diag_col else ""
        grade = _parse_hie_grade_from_text(diag_text)

        label, dq = hie_grade_to_label_dq(grade, has_seizures=False, rng=rng)
        records.append({
            "subject_id": subj_id,
            "eeg_file":   eeg_file,   # e.g. 'eeg10' → matches eeg10.edf
            "hie_grade":  grade,
            "label":      label,
            "dq":         dq,
        })

    return pd.DataFrame(records)


def add_seizure_labels(
    df: pd.DataFrame,
    annotations_csv_path: str,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Merge seizure annotation info into clinical DataFrame.
    Subjects with ≥1 reviewer annotating a seizure → label forced to 1, DQ −10.

    Helsinki annotations_2017_A.csv format:
      WIDE binary matrix — COLUMNS = subject IDs (as strings '1'..'79'),
      ROWS = reviewer annotations (1.0 = seizure annotated by that reviewer).
      A subject has confirmed seizures if its column sum > 0.

    df must have column 'subject_id' (numeric string, e.g. '10').
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    try:
        ann = pd.read_csv(annotations_csv_path)
    except Exception:
        return df  # annotations unreadable — skip silently

    # Build set of subject IDs whose column sums > 0
    subjects_with_seizures: set = set()
    for col in ann.columns:
        try:
            # Column header is the subject ID integer as a string
            sid = str(int(float(str(col).strip())))
            col_sum = pd.to_numeric(ann[col], errors="coerce").sum()
            if col_sum > 0:
                subjects_with_seizures.add(sid)
        except (ValueError, TypeError):
            pass

    result_rows = []
    for _, row in df.iterrows():
        row = row.copy()
        sid = str(row["subject_id"]).strip()
        if sid in subjects_with_seizures:
            row["label"] = 1
            row["dq"]    = max(0.0, float(row["dq"]) - 10.0)
        result_rows.append(row)

    return pd.DataFrame(result_rows)


# ---------------------------------------------------------------------------
# MRI / Baby Open Brains — participants.tsv → (label, dq)
# ---------------------------------------------------------------------------

def parse_mri_participants_tsv(participants_tsv_path: str) -> pd.DataFrame:
    """
    Parse participants.tsv from BIDS dataset.
    Returns DataFrame: participant_id, age_months, label, dq.

    Baby Open Brains = healthy controls → all label 0.
    Age in tsv may be in years (float < 10) — converted to months.
    """
    df = pd.read_csv(participants_tsv_path, sep="\t")
    df.columns = df.columns.str.strip()
    col_lower = {c.lower().strip(): c for c in df.columns}

    age_col = (
        col_lower.get("age") or
        col_lower.get("age_months") or
        col_lower.get("age_years") or
        col_lower.get("scan_age") or
        None
    )

    rng = np.random.default_rng(seed=42)
    records = []
    for _, row in df.iterrows():
        pid = str(row.get("participant_id", row.iloc[0])).strip()

        age_months = 6.0  # default
        if age_col is not None:
            try:
                age_raw = float(row[age_col])
                age_months = age_raw * 12.0 if age_raw < 10 else age_raw
            except (ValueError, TypeError):
                pass

        records.append({
            "participant_id": pid,
            "age_months":     age_months,
            "label":          0,
            "dq":             float(rng.uniform(80, 100)),
        })

    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# HPO / phenotype.hpoa — disease name → (label, dq)
# ---------------------------------------------------------------------------

ID_KEYWORDS = [
    "down syndrome", "trisomy 21", "fragile x", "angelman",
    "williams", "cornelia de lange", "kabuki", "prader-willi",
    "rett syndrome", "phelan-mcdermid", "intellectual disability",
    "intellectual developmental disorder", "global developmental delay",
    "mowat-wilson", "christianson", "kleefstra", "koolen-de vries",
    "charge", "rubinstein-taybi", "sotos", "weaver", "bohring-opitz",
]

_DISEASE_DQ_MAP = {
    "down syndrome":             (45, 65),
    "trisomy 21":                (45, 65),
    "fragile x":                 (40, 70),
    "angelman":                  (20, 45),
    "severe":                    (10, 35),
    "profound":                  (0,  20),
    "moderate":                  (35, 55),
    "mild":                      (55, 70),
    "global developmental delay":(40, 65),
}


def hpo_disease_to_label(disease_name: str) -> int:
    """Return 1 if disease is ID-relevant, 0 otherwise."""
    name_lower = disease_name.lower()
    return int(any(kw in name_lower for kw in ID_KEYWORDS))


def hpo_disease_to_dq(
    disease_name: str,
    label: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """Sample DQ for an HPO disease based on name patterns."""
    if rng is None:
        rng = np.random.default_rng()
    if label == 0:
        return float(rng.uniform(85, 100))
    name_lower = disease_name.lower()
    for keyword, (lo, hi) in _DISEASE_DQ_MAP.items():
        if keyword in name_lower:
            return float(rng.uniform(lo, hi))
    return float(rng.uniform(35, 65))
