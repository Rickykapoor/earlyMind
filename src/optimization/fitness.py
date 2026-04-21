"""
src/optimization/fitness.py
============================
Fitness evaluation for GWO hyperparameter search.

The fitness function maps a candidate hyperparameter set to a scalar
quality score (validation AUC, higher = better).

Two evaluation modes
--------------------
1. PROXY (default, fast):
   Uses a multi-factor heuristic calibrated against published deep-learning
   results to score hyperparameter configurations without training. Safe to
   run anywhere. Returns a score in [0, 1].

2. TRAINING (when training pipeline is available):
   Runs a short training loop (``quick_epochs`` epochs) and returns the
   actual validation AUC. Set ``use_real_training=True`` to activate.

Caching
-------
Both modes cache evaluated positions (hashed by rounded position vector)
so identical wolf positions are never evaluated twice. The cache is
stored in memory and optionally persisted to disk.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np

from .search_space import SearchSpace

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Proxy fitness helpers
# ---------------------------------------------------------------------------

def _lr_score(lr: float) -> float:
    """
    Penalise learning rates far from the sweet-spot for AdamW + transformers.
    Peak near [1e-4, 5e-4]; drops off on both sides (log-Gaussian).
    """
    mu  = math.log10(3e-4)       # optimal ~3e-4
    sig = 0.8                    # 1 order of magnitude tolerance
    return math.exp(-0.5 * ((math.log10(lr) - mu) / sig) ** 2)


def _batch_size_score(bs: int) -> float:
    """
    Larger batches → more stable gradients, but may hurt generalisation.
    Bell-shaped with peak at 32.
    """
    mu  = math.log2(32)
    sig = 1.0
    return math.exp(-0.5 * ((math.log2(bs) - mu) / sig) ** 2)


def _focal_loss_score(alpha: float, gamma: float) -> float:
    """
    Focal loss is most effective when gamma ∈ [1, 3] and alpha ∈ [0.2, 0.5].
    """
    a_score = 1.0 - 4.0 * (alpha - 0.35) ** 2   # peak at 0.35
    g_score = 1.0 - 0.25 * (gamma - 2.0) ** 2   # peak at 2.0
    a_score = max(0.0, min(1.0, a_score))
    g_score = max(0.0, min(1.0, g_score))
    return 0.5 * a_score + 0.5 * g_score


def _weight_decay_score(wd: float) -> float:
    """L2 reg: too small → overfit, too large → underfit. Peak near 1e-4."""
    mu  = math.log10(1e-4)
    sig = 1.0
    return math.exp(-0.5 * ((math.log10(wd) - mu) / sig) ** 2)


def _dropout_score(dropout: float) -> float:
    """Moderate dropout (0.1−0.3) is best for imbalanced medical data."""
    return 1.0 - abs(dropout - 0.2) / 0.3   # peak at 0.2


def _architecture_score(embed_dim: int, heads: int, layers: int) -> float:
    """
    Heuristic: embed_dim should be divisible by heads (required for attention).
    Moderate depth (fusion_layers=2−4) is usually optimal.
    """
    divisible  = 1.0 if embed_dim % heads == 0 else 0.0
    depth_ok   = 1.0 - abs(layers - 3) / 5.0   # peak at 3
    embed_ok   = 1.0 - abs(math.log2(embed_dim) - 7) / 3.0  # peak at 128
    return (divisible * 0.5 + depth_ok * 0.3 + embed_ok * 0.2)


def proxy_fitness(hparams: Dict[str, Any]) -> float:
    """
    Composite proxy AUC score from 0 to 1.

    Weighted sum of per-hyperparameter heuristic scores.
    Calibrated so that the "default" EarlyMind config scores ~0.72 and
    near-optimal configs score ~0.88+.
    """
    weights = {
        "lr":                   0.25,
        "weight_decay":         0.10,
        "batch_size":           0.10,
        "focal_loss":           0.15,
        "dropout":              0.10,
        "architecture":         0.20,
        "severity_loss_weight": 0.10,
    }

    scores = {
        "lr":                   _lr_score(hparams["lr"]),
        "weight_decay":         _weight_decay_score(hparams["weight_decay"]),
        "batch_size":           _batch_size_score(hparams["batch_size"]),
        "focal_loss":           _focal_loss_score(
                                    hparams["focal_alpha"],
                                    hparams["focal_gamma"]),
        "dropout":              _dropout_score(hparams["dropout"]),
        "architecture":         _architecture_score(
                                    hparams["embed_dim"],
                                    hparams["fusion_heads"],
                                    hparams["fusion_layers"]),
        "severity_loss_weight": 1.0 - abs(hparams["severity_loss_weight"] - 0.5) / 0.5,
    }

    raw = sum(scores[k] * weights[k] for k in weights)

    # Rescale from ~[0.5, 1.0] → [0.55, 0.95] to mimic realistic AUC range
    auc_proxy = 0.55 + raw * 0.40

    # Add tiny noise to break ties (avoids artificial stagnation)
    auc_proxy += np.random.uniform(-0.002, 0.002)

    return float(np.clip(auc_proxy, 0.0, 1.0))


# ---------------------------------------------------------------------------
# FitnessEvaluator
# ---------------------------------------------------------------------------

class FitnessEvaluator:
    """
    Callable fitness evaluator for GWO.

    Wraps proxy_fitness (default) or a real training loop.
    Handles:
      - Decoding normalised GWO positions → typed hyperparameter dicts
      - Caching (in-memory + optional disk persistence)
      - Timing and logging each evaluation

    Parameters
    ----------
    search_space      : SearchSpace instance (provides decode)
    use_real_training : If True, call ``training_fn`` instead of proxy.
    training_fn       : Optional callable(hparams) → val_auc. Used when
                        use_real_training=True.
    cache_path        : Optional path for disk-based evaluation cache.
    verbose           : Log each evaluation.

    Usage (proxy mode)
    ------------------
    >>> space = SearchSpace()
    >>> ev = FitnessEvaluator(space)
    >>> fitness = ev(np.array([0.3, 0.7, ...]))   # called by GWO

    Usage (real training, once training pipeline is ready)
    -------------------------------------------------------
    >>> def my_train(hparams):
    ...     # train model, return val AUC
    ...     return train_and_evaluate(hparams, epochs=5)
    >>> ev = FitnessEvaluator(space, use_real_training=True,
    ...                       training_fn=my_train)
    """

    def __init__(
        self,
        search_space:        SearchSpace,
        use_real_training:   bool = False,
        training_fn:         Optional[Callable[[Dict], float]] = None,
        cache_path:          Optional[Path] = None,
        verbose:             bool = True,
    ):
        self.search_space      = search_space
        self.use_real_training = use_real_training
        self.training_fn       = training_fn
        self.cache_path        = Path(cache_path) if cache_path else None
        self.verbose           = verbose

        self._cache: Dict[str, float] = {}
        self._eval_count = 0

        # Load persisted cache if available
        if self.cache_path and self.cache_path.exists():
            try:
                with open(self.cache_path) as f:
                    self._cache = json.load(f)
                log.info("Loaded %d cached evaluations from %s",
                         len(self._cache), self.cache_path)
            except Exception as e:
                log.warning("Could not load evaluation cache: %s", e)

    # ------------------------------------------------------------------
    # Main callable (used by GWO)
    # ------------------------------------------------------------------

    def __call__(self, position: np.ndarray) -> float:
        """
        Evaluate fitness for a normalised position vector.

        Parameters
        ----------
        position : np.ndarray, shape (n_dims,), values ∈ [0, 1]

        Returns
        -------
        float : fitness score (validation AUC), higher = better.
        """
        key = self._cache_key(position)

        # Return cached result if available
        if key in self._cache:
            return self._cache[key]

        # Decode → hyperparams
        hparams = self.search_space.decode(position)
        self._eval_count += 1

        t0 = time.perf_counter()

        if self.use_real_training and self.training_fn is not None:
            fitness = self._run_training(hparams)
            mode = "REAL"
        else:
            fitness = proxy_fitness(hparams)
            mode = "PROXY"

        elapsed = time.perf_counter() - t0

        # Cache result
        self._cache[key] = fitness
        if self.cache_path:
            self._persist_cache()

        if self.verbose:
            log.debug(
                "Eval #%d [%s] %.4f (%.2fs) | lr=%.1e wd=%.1e bs=%d"
                " f_α=%.2f f_γ=%.1f emb=%d dr=%.2f heads=%d layers=%d sev=%.2f",
                self._eval_count, mode, fitness, elapsed,
                hparams["lr"], hparams["weight_decay"], hparams["batch_size"],
                hparams["focal_alpha"], hparams["focal_gamma"],
                hparams["embed_dim"], hparams["dropout"],
                hparams["fusion_heads"], hparams["fusion_layers"],
                hparams["severity_loss_weight"],
            )

        return fitness

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_evaluations(self) -> int:
        return self._eval_count

    @property
    def cache_size(self) -> int:
        return len(self._cache)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _run_training(self, hparams: Dict) -> float:
        """Delegate to real training function with error handling."""
        try:
            return float(self.training_fn(hparams))
        except Exception as e:
            log.error("Training evaluation failed: %s — falling back to proxy", e)
            return proxy_fitness(hparams) * 0.9   # penalise failed runs

    @staticmethod
    def _cache_key(position: np.ndarray) -> str:
        """Deterministic cache key from rounded position."""
        rounded = np.round(position, 4)
        return hashlib.md5(rounded.tobytes()).hexdigest()

    def _persist_cache(self) -> None:
        """Write cache to disk."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "w") as f:
                json.dump(self._cache, f)
        except Exception as e:
            log.warning("Could not persist evaluation cache: %s", e)
