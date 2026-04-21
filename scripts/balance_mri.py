#!/usr/bin/env python3
"""
scripts/balance_mri.py
=======================
CLI script to run the full MRI data balancing pipeline for EarlyMind.

Steps
-----
  1. Preprocess raw T2w NIfTI volumes → datasets/processed/mri/  (real .npz)
  2. Augment real samples → datasets/mri/augmented/             (synthetic .npz)
  3. Print final summary + label distribution

Usage
-----
  # From repo root (default: 10 000 synthetic samples, seed 42):
  python scripts/balance_mri.py

  # Custom target and seed:
  python scripts/balance_mri.py --target 8000 --seed 123

  # Skip preprocessing (real .npz already exist):
  python scripts/balance_mri.py --skip-preprocess

  # Preview counts without writing files:
  python scripts/balance_mri.py --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Make sure project root is on path regardless of CWD
_HERE  = Path(__file__).resolve().parent
_ROOT  = _HERE.parent
sys.path.insert(0, str(_ROOT))

from src.config import cfg  # type: ignore
from src.data.mri_loader  import process_mri_dataset, MRIDataset  # type: ignore
from src.data.mri_augment import generate_augmented_dataset  # type: ignore


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fmt_time(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.1f}s"
    m, s = divmod(int(seconds), 60)
    return f"{m}m {s}s"


def _banner(msg: str) -> None:
    sep = "─" * (len(msg) + 4)
    print(f"\n┌{sep}┐")
    print(f"│  {msg}  │")
    print(f"└{sep}┘")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="EarlyMind — MRI data balancing pipeline",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--target", "-n",
        type=int,
        default=getattr(getattr(cfg, "augmentation", None), "mri_target_samples", 10_000),
        help="Number of synthetic MRI samples to generate",
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=getattr(getattr(cfg, "augmentation", None), "mri_aug_seed", 42),
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--dq-noise-std",
        type=float,
        default=getattr(getattr(cfg, "augmentation", None), "mri_dq_noise_std", 5.0),
        help="Std-dev of Gaussian noise added to DQ labels",
    )
    parser.add_argument(
        "--real-dir",
        type=Path,
        default=cfg.paths.mri_processed,
        help="Directory of real preprocessed MRI .npz files",
    )
    parser.add_argument(
        "--raw-dir",
        type=Path,
        default=cfg.paths.mri_raw,
        help="Directory of raw MRI NIfTI files (baby_open_brains)",
    )
    parser.add_argument(
        "--aug-dir",
        type=Path,
        default=Path("datasets/mri/augmented"),
        help="Output directory for synthetic .npz files",
    )
    parser.add_argument(
        "--skip-preprocess",
        action="store_true",
        help="Skip Step 1 (real preprocessing) and jump straight to augmentation",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Compute and print quotas without generating any files",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging verbosity",
    )
    args = parser.parse_args()

    # Make aug-dir absolute relative to repo root if given as relative
    aug_dir = args.aug_dir if args.aug_dir.is_absolute() else _ROOT / args.aug_dir

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    total_start = time.perf_counter()

    # ==================================================================
    # Header
    # ==================================================================
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║   EarlyMind 🧠 — MRI Data Balancing Pipeline         ║")
    print("╚══════════════════════════════════════════════════════╝")
    print(f"  Target synthetic samples : {args.target:,}")
    print(f"  Random seed              : {args.seed}")
    print(f"  DQ noise std             : {args.dq_noise_std}")
    print(f"  Real dir                 : {args.real_dir}")
    print(f"  Augmented dir            : {aug_dir}")
    print(f"  Dry run                  : {args.dry_run}")

    if args.dry_run:
        from src.data.mri_augment import TARGET_DIST, _CLASS_DQ_RANGES  # type: ignore
        _banner("DRY RUN — Class quota preview")
        class_names = ["Typical", "Borderline", "Mild ID", "Moderate ID", "Severe ID", "Profound ID"]
        for cls, frac in TARGET_DIST.items():
            quota = round(frac * args.target)
            lo, hi = _CLASS_DQ_RANGES[cls]
            print(f"  Class {cls}  {class_names[cls]:12s}  DQ {lo:.0f}–{hi:.0f}  →  {quota:,} samples")
        print(f"\n  Total: {args.target:,} samples")
        print("\n  [Dry run complete — no files written]\n")
        return

    # ==================================================================
    # Step 1 — Preprocess real NIfTI volumes
    # ==================================================================
    if not args.skip_preprocess:
        _banner("Step 1 / 2 — Preprocessing real MRI volumes")
        t0 = time.perf_counter()

        results = process_mri_dataset(
            mri_dir    = args.raw_dir,
            output_dir = args.real_dir,
            slice_size = cfg.data.mri_slice_size,
        )
        elapsed = time.perf_counter() - t0
        n_real = len(results)
        print(f"\n  ✅  Preprocessed {n_real} subjects → {args.real_dir}")
        print(f"  ⏱   {_fmt_time(elapsed)}\n")

        for pid, info in results.items():
            print(f"     {pid}: DQ={info['dq']:5.1f}  label={info['label']}"
                  f"  age={info['age_months']:.1f}mo")
    else:
        real_files = list(args.real_dir.glob("*.npz"))
        n_real     = len(real_files)
        print(f"\n  ⚡  Skipping preprocessing — found {n_real} real .npz files in {args.real_dir}")

    print()

    # ==================================================================
    # Step 2 — Generate augmented dataset
    # ==================================================================
    _banner("Step 2 / 2 — Generating synthetic MRI samples")
    t0 = time.perf_counter()

    generate_augmented_dataset(
        real_dir      = args.real_dir,
        output_dir    = aug_dir,
        target_n      = args.target,
        seed          = args.seed,
        dq_noise_std  = args.dq_noise_std,
        slice_size    = cfg.data.mri_slice_size,
        verbose       = True,
    )

    elapsed_aug = time.perf_counter() - t0
    print(f"\n  ⏱   Augmentation: {_fmt_time(elapsed_aug)}")

    # ==================================================================
    # Step 3 — Summary + verification
    # ==================================================================
    _banner("Verification")

    aug_files = sorted(aug_dir.glob("*.npz"))
    generated = len(aug_files)
    print(f"  Files in {aug_dir}: {generated:,}")

    # Load combined dataset and print distribution
    try:
        ds = MRIDataset(
            real_dir      = args.real_dir,
            augmented_dir = aug_dir,
            use_augmented = True,
        )
        dist = ds.label_distribution()
        print(f"  Combined dataset size : {len(ds):,} samples\n")
        print("  Label distribution:")
        for label_name, count in dist.items():
            pct = count / len(ds) * 100
            bar = "█" * int(pct / 2)
            print(f"    {label_name:12s}: {count:6,}  ({pct:5.1f}%)  {bar}")
    except Exception as exc:
        print(f"  [Warning] Could not compute distribution: {exc}")

    total_elapsed = time.perf_counter() - total_start
    print(f"\n  ⏱   Total time: {_fmt_time(total_elapsed)}")
    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   🎉  MRI data balancing complete!                   ║")
    print("╚══════════════════════════════════════════════════════╝\n")


if __name__ == "__main__":
    main()
