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
) -> tuple[int, float]:
    """
    Map HIE grade integer (0–3) to binary risk label and a sampled DQ.

    HIE grade 0 → label 0 (typical)
    HIE grades 1–3 → label 1 (ID risk)
    Seizures add extra DQ penalty (handled in sample_dq).
    """
    if rng is None:
        rng = np.random.default_rng()

    if hie_grade is None or hie_grade == 0:
        label = 0
    else:
        label = 1

    dq = sample_dq(label, hie_grade=hie_grade, has_seizures=has_seizures, rng=rng)
    return label, dq


def parse_eeg_clinical_csv(clinical_csv_path: str) -> pd.DataFrame:
    """
    Load clinical_information.csv from Helsinki Neonatal dataset.
    Inspects columns dynamically and returns DataFrame with standardized columns:
      subject_id, hie_grade (int 0–3), label (int 0/1), dq (float)
    """
    df = pd.read_csv(clinical_csv_path)
    df.columns = df.columns.str.strip()

    # Detect subject ID column
    id_col = None
    for candidate in ["subject_id", "subject", "id", "patient_id", "file", "eeg_file"]:
        if candidate in df.columns.str.lower().tolist():
            id_col = df.columns[df.columns.str.lower() == candidate][0]
            break
    if id_col is None:
        # Fall back to first column
        id_col = df.columns[0]

    # Detect HIE grade column
    grade_col = None
    for candidate in ["hie_grade", "grade", "hie", "outcome", "severity"]:
        if candidate in df.columns.str.lower().tolist():
            grade_col = df.columns[df.columns.str.lower() == candidate][0]
            break
    if grade_col is None:
        raise ValueError(
            f"Cannot find HIE grade column. Available columns: {df.columns.tolist()}"
        )

    rng = np.random.default_rng(seed=42)
    records = []
    for _, row in df.iterrows():
        try:
            grade_raw = str(row[grade_col]).strip().lower()
            if grade_raw in ("nan", "none", ""):
                grade = None
            elif grade_raw in ("0", "normal", "no hie"):
                grade = 0
            elif grade_raw in ("1", "mild"):
                grade = 1
            elif grade_raw in ("2", "moderate"):
                grade = 2
            elif grade_raw in ("3", "severe"):
                grade = 3
            else:
                # Try numeric conversion
                try:
                    grade = int(float(grade_raw))
                except ValueError:
                    grade = None
        except Exception:
            grade = None

        label, dq = hie_grade_to_label_dq(grade, has_seizures=False, rng=rng)
        records.append({
            "subject_id": str(row[id_col]).strip(),
            "hie_grade": grade,
            "label": label,
            "dq": dq,
        })

    return pd.DataFrame(records)


def add_seizure_labels(
    df: pd.DataFrame,
    annotations_csv_path: str,
    rng: Optional[np.random.Generator] = None,
) -> pd.DataFrame:
    """
    Merge seizure annotation info into clinical DataFrame.
    Subjects with any annotated seizure have label forced to 1 and DQ shifted −10.

    annotations_csv_path: path to annotations_2017_A.csv
    df must have column 'subject_id'.
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    ann = pd.read_csv(annotations_csv_path)
    ann.columns = ann.columns.str.strip()

    # Detect subject column in annotations
    subj_col = None
    for candidate in ["subject", "subject_id", "id", "patient", "file"]:
        if candidate in ann.columns.str.lower().tolist():
            subj_col = ann.columns[ann.columns.str.lower() == candidate][0]
            break
    if subj_col is None:
        subj_col = ann.columns[0]

    subjects_with_seizures = set(ann[subj_col].astype(str).str.strip().unique())

    def _update_row(row):
        has_sz = str(row["subject_id"]) in subjects_with_seizures
        if has_sz:
            row = row.copy()
            row["label"] = 1
            row["dq"] = max(0.0, float(row["dq"]) - 10.0)
        return row

    return df.apply(_update_row, axis=1)


# ---------------------------------------------------------------------------
# MRI / Baby Open Brains — participants.tsv → (label, dq)
# ---------------------------------------------------------------------------

def parse_mri_participants_tsv(participants_tsv_path: str) -> pd.DataFrame:
    """
    Parse participants.tsv from BIDS dataset.
    Returns DataFrame with columns: participant_id, age_months, label, dq.

    Baby Open Brains = healthy controls → all label 0.
    Age in tsv may be in years (float) — convert to months.
    """
    df = pd.read_csv(participants_tsv_path, sep="\t")
    df.columns = df.columns.str.strip()

    # Detect age column
    age_col = None
    for candidate in ["age", "age_months", "age_years", "scan_age"]:
        if candidate in df.columns.str.lower().tolist():
            age_col = df.columns[df.columns.str.lower() == candidate][0]
            break

    rng = np.random.default_rng(seed=42)
    records = []
    for _, row in df.iterrows():
        pid = str(row.get("participant_id", row.iloc[0])).strip()

        # Parse age
        if age_col is not None:
            try:
                age_raw = float(row[age_col])
                # Heuristic: if age < 10, assume years; else assume months
                if age_raw < 10:
                    age_months = age_raw * 12.0
                else:
                    age_months = age_raw
            except (ValueError, TypeError):
                age_months = 6.0  # default: infant
        else:
            age_months = 6.0  # default

        # All subjects in Baby Open Brains are healthy controls
        label = 0
        dq = float(rng.uniform(80, 100))

        records.append({
            "participant_id": pid,
            "age_months": age_months,
            "label": label,
            "dq": dq,
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

# DQ sampling ranges per known disease pattern
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
    for kw in ID_KEYWORDS:
        if kw in name_lower:
            return 1
    return 0


def hpo_disease_to_dq(
    disease_name: str,
    label: int,
    rng: Optional[np.random.Generator] = None,
) -> float:
    """
    Sample DQ for an HPO disease based on disease name patterns.
    Only meaningful for label=1 diseases.
    """
    if rng is None:
        rng = np.random.default_rng()

    if label == 0:
        return float(rng.uniform(85, 100))

    name_lower = disease_name.lower()
    for keyword, (lo, hi) in _DISEASE_DQ_MAP.items():
        if keyword in name_lower:
            return float(rng.uniform(lo, hi))

    # Default ID range
    return float(rng.uniform(35, 65))
