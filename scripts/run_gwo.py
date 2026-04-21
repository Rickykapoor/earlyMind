#!/usr/bin/env python3
"""
scripts/run_gwo.py
===================
CLI entry-point for EarlyMind Grey Wolf Optimizer hyperparameter search.

Usage
-----
  # Full run (10 wolves × 30 iterations = up to 300 evaluations)
  python scripts/run_gwo.py

  # Quick test run
  python scripts/run_gwo.py --wolves 5 --iters 5

  # Dry run — just print search space, no evaluation
  python scripts/run_gwo.py --dry-run

  # Use a custom seed
  python scripts/run_gwo.py --wolves 15 --iters 50 --seed 123

  # Update params.yaml automatically with best hyperparams
  python scripts/run_gwo.py --update-params

Output
------
  reports/gwo_results.json   — full results (fitness history, best config)
  Prints best hyperparameters to stdout.

Connecting to real training
---------------------------
Once the training pipeline is implemented in src/training/, swap the
fitness evaluator to real-training mode:

  evaluator = FitnessEvaluator(
      search_space,
      use_real_training=True,
      training_fn=lambda hp: train_and_get_auc(hp, quick_epochs=5),
  )
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on Python path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.optimization.fitness import FitnessEvaluator
from src.optimization.gwo import GreyWolfOptimizer
from src.optimization.search_space import SearchSpace

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s  %(levelname)s  %(name)s  %(message)s",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="EarlyMind — Grey Wolf Optimizer hyperparameter search",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--wolves",        type=int,   default=10,
                   help="Number of wolves (population size)")
    p.add_argument("--iters",         type=int,   default=30,
                   help="Max number of GWO iterations")
    p.add_argument("--seed",          type=int,   default=42,
                   help="Random seed")
    p.add_argument("--patience",      type=int,   default=10,
                   help="Early-stop patience (iterations without improvement)")
    p.add_argument("--tol",           type=float, default=1e-4,
                   help="Minimum improvement to reset patience counter")
    p.add_argument("--output",        type=str,
                   default="reports/gwo_results.json",
                   help="Path to save GWO result JSON")
    p.add_argument("--cache",         type=str,
                   default="reports/gwo_eval_cache.json",
                   help="Path to persist evaluation cache")
    p.add_argument("--dry-run",       action="store_true",
                   help="Print search space and exit (no evaluation)")
    p.add_argument("--update-params", action="store_true",
                   help="Write best hyperparams back to params.yaml")
    p.add_argument("--params-yaml",   type=str,
                   default="params.yaml",
                   help="Path to params.yaml (used with --update-params)")
    return p


# ---------------------------------------------------------------------------
# params.yaml updater
# ---------------------------------------------------------------------------

def update_params_yaml(yaml_path: Path, best_hparams: dict) -> None:
    """
    Write the best hyperparams found by GWO back into params.yaml,
    preserving all other keys.
    """
    if not yaml_path.exists():
        print(f"  ⚠  {yaml_path} not found — skipping params update.")
        return

    with open(yaml_path) as f:
        config = yaml.safe_load(f) or {}

    # ── training section ──────────────────────────────────────────────
    training = config.setdefault("training", {})
    training["lr"]                   = float(best_hparams["lr"])
    training["weight_decay"]         = float(best_hparams["weight_decay"])
    training["batch_size"]           = int(best_hparams["batch_size"])
    training["focal_alpha"]          = float(best_hparams["focal_alpha"])
    training["focal_gamma"]          = float(best_hparams["focal_gamma"])
    training["severity_loss_weight"] = float(best_hparams["severity_loss_weight"])

    # ── model section ─────────────────────────────────────────────────
    model = config.setdefault("model", {})
    model["embed_dim"]    = int(best_hparams["embed_dim"])
    model["dropout"]      = float(best_hparams["dropout"])
    model["fusion_heads"] = int(best_hparams["fusion_heads"])
    model["fusion_layers"]= int(best_hparams["fusion_layers"])

    with open(yaml_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=True)

    print(f"\n  ✅  params.yaml updated with GWO best hyperparameters → {yaml_path}")


# ---------------------------------------------------------------------------
# Pretty-print results
# ---------------------------------------------------------------------------

def print_comparison(current: dict, best: dict) -> None:
    """Show a side-by-side comparison of default vs. GWO-optimal params."""
    keys = list(best.keys())

    print("\n┌────────────────────────────────────────────────────────────────┐")
    print("│  Hyperparameter Comparison: Default vs GWO-Optimal             │")
    print("└────────────────────────────────────────────────────────────────┘")
    print(f"  {'Parameter':<28}  {'Default':>12}  {'GWO Best':>12}  Change")
    print(f"  {'─────────':<28}  {'───────':>12}  {'────────':>12}  ──────")

    for k in keys:
        cur_v = current.get(k)
        new_v = best.get(k)
        if cur_v is None:
            marker = "  NEW"
        elif isinstance(new_v, float) and isinstance(cur_v, float):
            ratio = new_v / cur_v if cur_v != 0 else float("inf")
            marker = f"  ×{ratio:.2f}" if abs(ratio - 1.0) > 0.05 else "  ─"
        else:
            marker = "  ✓" if new_v == cur_v else f"  → {new_v}"

        cv = f"{cur_v:.4g}" if isinstance(cur_v, float) else str(cur_v)
        nv = f"{new_v:.4g}" if isinstance(new_v, float) else str(new_v)
        print(f"  {k:<28}  {cv:>12}  {nv:>12} {marker}")


# ---------------------------------------------------------------------------
# Default EarlyMind config (from params.yaml baseline)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "lr":                   3e-4,
    "weight_decay":         1e-4,
    "batch_size":           16,
    "focal_alpha":          0.25,
    "focal_gamma":          2.0,
    "severity_loss_weight": 0.5,
    "embed_dim":            128,
    "dropout":              0.2,
    "fusion_heads":         4,
    "fusion_layers":        3,
}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = build_parser().parse_args()

    print("\n╔══════════════════════════════════════════════════════╗")
    print("║   EarlyMind 🐺 — Grey Wolf Optimizer                 ║")
    print("╚══════════════════════════════════════════════════════╝")

    # ── Build search space ────────────────────────────────────────────
    space = SearchSpace()
    space.summary()

    if args.dry_run:
        print("  ── Dry run mode: no evaluation performed. ──\n")

        # Show what the default config decodes to
        default_pos = space.encode(DEFAULT_CONFIG)
        print("  Default EarlyMind config as GWO position:")
        print(space.describe(space.clip(default_pos)))
        return

    # ── Build fitness evaluator ───────────────────────────────────────
    evaluator = FitnessEvaluator(
        search_space=space,
        use_real_training=False,          # set True once training loop exists
        cache_path=Path(args.cache),
        verbose=False,
    )

    # ── Run GWO ──────────────────────────────────────────────────────
    gwo = GreyWolfOptimizer(
        fitness_fn = evaluator,
        n_dims     = space.n_dims,
        decode_fn  = space.decode,
        n_wolves   = args.wolves,
        max_iter   = args.iters,
        seed       = args.seed,
        tol        = args.tol,
        patience   = args.patience,
        verbose    = True,
    )

    result = gwo.run()

    # ── Save results ──────────────────────────────────────────────────
    out_path = Path(args.output)
    result.save(out_path)
    print(f"\n  📄 Full results saved → {out_path}")

    # ── Show comparison ───────────────────────────────────────────────
    print_comparison(DEFAULT_CONFIG, result.best_hyperparams)

    print(f"\n  Proxy AUC improvement: "
          f"{DEFAULT_CONFIG_SCORE:.4f} → {result.best_fitness:.4f} "
          f"(+{result.best_fitness - DEFAULT_CONFIG_SCORE:.4f})")

    # ── Optionally update params.yaml ─────────────────────────────────
    if args.update_params:
        update_params_yaml(Path(args.params_yaml), result.best_hyperparams)

    print()


# Compute default config proxy score for comparison display
def _compute_default_score() -> float:
    from src.optimization.fitness import proxy_fitness
    return proxy_fitness(DEFAULT_CONFIG)


try:
    DEFAULT_CONFIG_SCORE = _compute_default_score()
except Exception:
    DEFAULT_CONFIG_SCORE = 0.72


if __name__ == "__main__":
    main()
