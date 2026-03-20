"""
src/data/hpo_loader.py
HPO phenotype.hpoa → frequency-weighted feature matrix for FacialEncoder.
No face images are used; all features are derived from HPO term annotations.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.config import cfg
from src.utils.label_utils import hpo_disease_to_label, hpo_disease_to_dq

# ---------------------------------------------------------------------------
# HPO subtree roots to retain (clinically relevant for ID/dysmorphology)
# ---------------------------------------------------------------------------

HPO_SUBTREE_PREFIXES = [
    "HP:0000234",   # Abnormality of the head
    "HP:0000478",   # Abnormality of the eye
    "HP:0000598",   # Abnormality of the ear
    "HP:0000924",   # Abnormality of the skeletal system
    "HP:0012638",   # Abnormality of nervous system physiology
    "HP:0001249",   # Intellectual disability (direct subtree)
]

# Frequency HP terms → numeric weights
FREQ_HP_WEIGHTS: Dict[str, float] = {
    "HP:0040280": 1.0,   # Obligate
    "HP:0040281": 0.9,   # Very frequent (80–99%)
    "HP:0040282": 0.6,   # Frequent (30–79%)
    "HP:0040283": 0.3,   # Occasional (5–29%)
    "HP:0040284": 0.1,   # Very rare (1–4%)
    "HP:0040285": 0.0,   # Excluded
}


def _parse_frequency(freq_str: str) -> float:
    """
    Convert a Frequency column value to a float weight in [0, 1].
    Handles HP:XXXXXXX codes, percentage strings, and missing values.
    """
    if not freq_str or pd.isna(freq_str) or str(freq_str).strip() in ("", "nan"):
        return 0.5  # unknown → 0.5

    freq_str = str(freq_str).strip()

    # HP term codes
    if freq_str.startswith("HP:"):
        return FREQ_HP_WEIGHTS.get(freq_str, 0.5)

    # Percentage pattern: "50%" or "50/100" or "50"
    pct_match = re.search(r"(\d+\.?\d*)\s*%", freq_str)
    if pct_match:
        return float(pct_match.group(1)) / 100.0

    frac_match = re.match(r"(\d+)\s*/\s*(\d+)", freq_str)
    if frac_match:
        num, denom = int(frac_match.group(1)), int(frac_match.group(2))
        return num / denom if denom > 0 else 0.5

    # Textual qualifiers
    low = freq_str.lower()
    if "oblig" in low:         return 1.0
    if "very frequent" in low: return 0.9
    if "frequent" in low:      return 0.6
    if "occasional" in low:    return 0.3
    if "very rare" in low:     return 0.1
    if "excluded" in low:      return 0.0

    # Try raw numeric
    try:
        v = float(freq_str)
        return v / 100.0 if v > 1.0 else v
    except ValueError:
        return 0.5


def _is_relevant_hpo_term(hpo_id: str) -> bool:
    """
    Check if an HPO term ID belongs to one of the clinically relevant subtrees.
    Note: without a full HPO ontology graph, we use a prefix heuristic since
    HPO IDs within each subtree share numeric ranges.
    We keep all terms here and rely on the kept disease filter + min_df to reduce noise.
    """
    # All phenotypic terms start with HP:
    return str(hpo_id).startswith("HP:")


# ---------------------------------------------------------------------------
# Main loader
# ---------------------------------------------------------------------------

def load_hpo_matrix(
    hpo_dir: str | Path,
    min_df: int = 5,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], List[str], StandardScaler]:
    """
    Load HPO data and construct frequency-weighted feature matrix.

    Parameters
    ----------
    hpo_dir : str | Path — path to datasets/facial/hpo/
    min_df  : int — minimum number of diseases a term must appear in
    rng     : optional random generator for DQ sampling

    Returns
    -------
    X        : np.ndarray (n_diseases, n_features)
    y        : np.ndarray (n_diseases,) — binary labels
    dq       : np.ndarray (n_diseases,) — DQ severity estimates
    feat_names : list of str — feature names
    disease_names : list of str — disease names (row index)
    scaler   : fitted StandardScaler
    """
    if rng is None:
        rng = np.random.default_rng(seed=42)

    hpo_dir = Path(hpo_dir)
    phenotype_path      = hpo_dir / "phenotype.hpoa"
    genes_diseases_path = hpo_dir / "genes_to_diseases.txt"

    if not phenotype_path.exists():
        raise FileNotFoundError(f"phenotype.hpoa not found at {phenotype_path}")

    # -------------------------------------------------------------------
    # Step 1: Load phenotype.hpoa
    # -------------------------------------------------------------------
    print("  Loading phenotype.hpoa …")
    # Find the header row dynamically
    skip = 0
    with open(phenotype_path, "r") as f:
        for i, line in enumerate(f):
            if line.startswith('DatabaseID') or line.startswith('#DatabaseID'):
                skip = i
                break

    sep = "\t"
    df = pd.read_csv(phenotype_path, sep=sep, skiprows=skip, low_memory=False)

    # Handle comment character in header
    if df.columns[0].startswith("#"):
        df.columns = [c.lstrip("#").strip() for c in df.columns]

    print(f"    Columns: {df.columns.tolist()}")
    print(f"    Shape: {df.shape}")

    # Standardize column names (lowercase)
    df.columns = [c.strip() for c in df.columns]

    # Detect column names dynamically
    col_map = {c.lower(): c for c in df.columns}

    disease_col   = col_map.get("databaseid",    col_map.get("database_id",   col_map.get("diseaseid",   df.columns[0])))
    name_col      = col_map.get("diseasename",   col_map.get("disease_name",  col_map.get("name",        df.columns[1])))
    hpo_col       = col_map.get("hpo_id",        col_map.get("hpoid",         col_map.get("hpo",         "HPO_ID")))
    freq_col      = col_map.get("frequency",     "Frequency")
    qual_col      = col_map.get("qualifier",     "Qualifier")
    aspect_col    = col_map.get("aspect",        "Aspect")

    # -------------------------------------------------------------------
    # Step 2: Filter to phenotype rows (Aspect == "P") and remove "NOT"
    # -------------------------------------------------------------------
    if aspect_col in df.columns:
        df = df[df[aspect_col].astype(str).str.strip().str.upper() == "P"].copy()
    if qual_col in df.columns:
        df = df[df[qual_col].astype(str).str.strip().str.upper() != "NOT"].copy()

    print(f"    After filtering: {df.shape[0]} rows")

    # -------------------------------------------------------------------
    # Step 3: Filter to ID-relevant diseases
    # -------------------------------------------------------------------
    from src.utils.label_utils import ID_KEYWORDS

    disease_names_all = df[name_col].astype(str).str.strip()
    id_mask = disease_names_all.str.lower().apply(
        lambda n: any(kw in n for kw in ID_KEYWORDS)
    )
    non_id_mask = ~id_mask

    df_id     = df[id_mask].copy()
    df_non_id = df[non_id_mask].copy()

    print(f"    ID-relevant rows: {len(df_id)}, Non-ID rows: {len(df_non_id)}")

    # Combine (keep some non-ID for negative class)
    df_all = pd.concat([df_id, df_non_id], ignore_index=True)

    # -------------------------------------------------------------------
    # Step 4: Parse frequency weights
    # -------------------------------------------------------------------
    if freq_col in df_all.columns:
        df_all["freq_weight"] = df_all[freq_col].apply(_parse_frequency)
    else:
        df_all["freq_weight"] = 0.5

    # -------------------------------------------------------------------
    # Step 5: Pivot — rows = diseases, cols = HPO terms
    # -------------------------------------------------------------------
    print("  Building pivot table …")

    # Use disease name as index (cleaner than DB IDs)
    df_all["__disease__"] = df_all[name_col].astype(str).str.strip()
    df_all["__hpo__"]     = df_all[hpo_col].astype(str).str.strip()

    # Group and take max weight if duplicates exist
    grouped = (
        df_all.groupby(["__disease__", "__hpo__"])["freq_weight"]
        .max()
        .reset_index()
    )

    pivot = grouped.pivot(index="__disease__", columns="__hpo__", values="freq_weight").fillna(0.0)

    # -------------------------------------------------------------------
    # Step 6: Filter to HPO terms with min_df appearance
    # -------------------------------------------------------------------
    col_counts = (pivot > 0).sum(axis=0)
    pivot = pivot.loc[:, col_counts >= min_df]
    print(f"    Pivot shape after min_df={min_df}: {pivot.shape}")

    # -------------------------------------------------------------------
    # Step 7: Add gene features if available
    # -------------------------------------------------------------------
    gene_count_map: Dict[str, int] = {}
    if genes_diseases_path.exists():
        try:
            gd = pd.read_csv(genes_diseases_path, sep="\t", comment="#",
                             header=None, low_memory=False)
            # Columns: gene_id, gene_symbol, disease_id, disease_name
            if gd.shape[1] >= 4:
                disease_gene_col = gd.iloc[:, 3].astype(str).str.strip()
                for dname in disease_gene_col:
                    gene_count_map[dname] = gene_count_map.get(dname, 0) + 1
        except Exception as e:
            print(f"    [WARNING] Could not load genes_to_diseases.txt: {e}")

    pivot["gene_count"]    = pivot.index.map(lambda d: gene_count_map.get(d, 0)).astype(float)
    pivot["has_known_gene"]= (pivot["gene_count"] > 0).astype(float)

    # -------------------------------------------------------------------
    # Step 8: Build labels and DQ values
    # -------------------------------------------------------------------
    disease_names: List[str] = pivot.index.tolist()
    y_labels = np.array([hpo_disease_to_label(n) for n in disease_names], dtype=np.float32)
    dq_values = np.array(
        [hpo_disease_to_dq(n, int(lbl), rng=rng) for n, lbl in zip(disease_names, y_labels)],
        dtype=np.float32
    )

    # -------------------------------------------------------------------
    # Step 9: StandardScaler
    # -------------------------------------------------------------------
    X_raw = pivot.values.astype(np.float32)
    feat_names = pivot.columns.tolist()

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_raw).astype(np.float32)

    print(f"  HPO matrix final shape: {X_scaled.shape}")
    print(f"  ID-positive: {int(y_labels.sum())}, Non-ID: {int((y_labels == 0).sum())}")

    return X_scaled, y_labels, dq_values, feat_names, disease_names, scaler


def process_hpo_dataset(
    hpo_dir: str | Path,
    output_dir: str | Path,
) -> Dict[str, np.ndarray]:
    """
    Full preprocessing pipeline for HPO data.
    Saves:
        hpo_matrix.npy        — (n_diseases, n_features)
        hpo_labels.npy        — (n_diseases,) binary
        hpo_dq.npy            — (n_diseases,) DQ values
        hpo_feature_names.npy — (n_features,) strings
        hpo_disease_names.npy — (n_diseases,) strings
    Updates params.yaml with actual hpo_n_features.

    Returns dict with all arrays.
    """
    import pickle

    hpo_dir    = Path(hpo_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X, y, dq, feat_names, disease_names, scaler = load_hpo_matrix(
        hpo_dir,
        min_df=cfg.data.hpo_min_disease_freq,
    )

    np.save(str(output_dir / "hpo_matrix.npy"),        X)
    np.save(str(output_dir / "hpo_labels.npy"),        y)
    np.save(str(output_dir / "hpo_dq.npy"),            dq)
    np.save(str(output_dir / "hpo_feature_names.npy"), np.array(feat_names, dtype=object))
    np.save(str(output_dir / "hpo_disease_names.npy"), np.array(disease_names, dtype=object))

    # Save scaler
    with open(str(output_dir / "hpo_scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)

    # Update params.yaml with actual feature count
    params_path = Path(__file__).resolve().parents[2] / "params.yaml"
    if params_path.exists():
        import yaml
        with open(params_path, "r") as f:
            params = yaml.safe_load(f)
        params.setdefault("model", {})["hpo_n_features"] = int(X.shape[1])
        with open(params_path, "w") as f:
            yaml.dump(params, f, default_flow_style=False)
        print(f"  Updated params.yaml: hpo_n_features = {X.shape[1]}")

    print(f"  HPO processing complete → {output_dir}")
    return {
        "X": X,
        "y": y,
        "dq": dq,
        "feat_names": feat_names,
        "disease_names": disease_names,
        "scaler": scaler,
    }
