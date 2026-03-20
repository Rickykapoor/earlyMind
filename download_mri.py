"""
download_mri.py — Standalone script to download Baby Open Brains MRI dataset.
Uses openneuro-py to pull dataset ds004797 to datasets/mri/baby_open_brains/.

Usage:
    /opt/anaconda3/envs/infant_id/bin/python download_mri.py
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

DATASET_ID  = "ds004797"
VERSION     = "1.0.1"         # latest stable version
OUTPUT_DIR  = Path("datasets/mri/baby_open_brains")


def download_with_openneuro_py():
    """Use openneuro-py library to download the dataset."""
    try:
        import openneuro
    except ImportError:
        print("Installing openneuro-py ...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "openneuro-py"])
        import openneuro

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATASET_ID} v{VERSION} to {OUTPUT_DIR} ...")

    openneuro.download(
        dataset=DATASET_ID,
        version=VERSION,
        target_dir=str(OUTPUT_DIR),
        include=[
            "participants.tsv",
            "dataset_description.json",
            "README",
            "sub-*/ses-1/anat/*_T2w.nii.gz",
            "sub-*/ses-1/anat/*_T1w.nii.gz",
        ],
    )
    print("Download complete!")


def verify_download():
    """Check that expected files exist and print summary."""
    if not OUTPUT_DIR.exists():
        print(f"ERROR: {OUTPUT_DIR} does not exist.")
        return False

    subjects = sorted([d for d in OUTPUT_DIR.iterdir() if d.name.startswith("sub-")])
    print(f"\nFound {len(subjects)} subject folders:")

    for subj in subjects:
        t2_files = list(subj.glob("**/*T2w.nii.gz"))
        t1_files = list(subj.glob("**/*T1w.nii.gz"))
        status_t2 = f"T2w ✅" if t2_files else "T2w ❌"
        status_t1 = f"T1w ✅" if t1_files else "T1w ❌"
        print(f"  {subj.name}: {status_t2} | {status_t1}")

    tsv = OUTPUT_DIR / "participants.tsv"
    if tsv.exists():
        import pandas as pd
        df = pd.read_csv(tsv, sep="\t")
        print(f"\nparticipants.tsv: {len(df)} rows, columns: {df.columns.tolist()}")
    else:
        print("\nWARNING: participants.tsv not found")

    return len(subjects) > 0


if __name__ == "__main__":
    print("=" * 60)
    print(" EarlyMind — MRI Dataset Download")
    print(f" Dataset: {DATASET_ID} (Baby Open Brains)")
    print(f" Output:  {OUTPUT_DIR.resolve()}")
    print("=" * 60)

    if OUTPUT_DIR.exists() and any(OUTPUT_DIR.iterdir()):
        print(f"\n{OUTPUT_DIR} already exists and is non-empty.")
        print("Skipping download. To re-download, delete the directory first.")
        verify_download()
    else:
        try:
            download_with_openneuro_py()
            verify_download()
        except Exception as e:
            print(f"\nERROR during download: {e}")
            print("\nAlternative manual download:")
            print(f"  pip install openneuro-py")
            print(f"  openneuro download --dataset={DATASET_ID} --output={OUTPUT_DIR}")
            sys.exit(1)
