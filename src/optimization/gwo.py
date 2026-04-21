"""
src/optimization/gwo.py
========================
Grey Wolf Optimizer (GWO) — core algorithm implementation.

Reference:
    Mirjalili, S., Mirjalili, S. M., & Lewis, A. (2014).
    Grey wolf optimizer. Advances in Engineering Software, 69, 46-61.

EarlyMind usage:
    Searches the hyperparameter space (learning rate, batch size, etc.)
    to maximize validation AUC. Works as a two-level optimization on top
    of AdamW:
      - Level 1 (outer): GWO searches hyperparameter space
      - Level 2 (inner): AdamW optimises model weights with those hyperparams

Algorithm
---------
Wolf hierarchy:
    α (alpha)  → Best fitness position  (leads the hunt)
    β (beta)   → 2nd best fitness       (assists alpha)
    δ (delta)  → 3rd best fitness       (assists alpha & beta)
    ω (omega)  → All remaining wolves   (update via the three leaders)

Update equations (per dimension, per wolf):
    a  = 2 - t × (2 / max_iter)         # linearly 2 → 0
    A  = 2·a·r₁ - a                     # convergence / exploration factor
    C  = 2·r₂                           # random weight ∈ [0, 2]

    D_x = |C · X_x - X_wolf|            # x ∈ {α, β, δ}
    movement_x = X_x - A · D_x

    X(t+1) = (movement_α + movement_β + movement_δ) / 3

When |A| > 1  → exploration  (wolves diverge from prey)
When |A| < 1  → exploitation (wolves converge on prey)
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class WolfPosition:
    """One wolf = one candidate hyperparameter vector + its fitness score."""
    position: np.ndarray       # shape (n_dims,), normalised to [0, 1]
    fitness:  float = -np.inf  # validation AUC (higher = better)

    def copy(self) -> "WolfPosition":
        return WolfPosition(self.position.copy(), self.fitness)


@dataclass
class GWOHistory:
    """Full run history for analysis and saving."""
    iteration_best:  List[float] = field(default_factory=list)  # best AUC per iter
    iteration_mean:  List[float] = field(default_factory=list)  # mean AUC per iter
    alpha_positions: List[List[float]] = field(default_factory=list)
    elapsed_seconds: List[float] = field(default_factory=list)


@dataclass
class GWOResult:
    """Final result returned to caller."""
    best_position:   List[float]
    best_fitness:    float
    best_hyperparams: Dict
    history:         GWOHistory
    n_evaluations:   int
    total_time_s:    float
    converged:       bool

    def to_dict(self) -> dict:
        return {
            "best_fitness":   self.best_fitness,
            "best_hyperparams": self.best_hyperparams,
            "n_evaluations":  self.n_evaluations,
            "total_time_s":   self.total_time_s,
            "converged":      self.converged,
            "history": {
                "iteration_best":  self.history.iteration_best,
                "iteration_mean":  self.history.iteration_mean,
            },
        }

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)
        log.info("GWO results saved to %s", path)


# ---------------------------------------------------------------------------
# Core GWO class
# ---------------------------------------------------------------------------

class GreyWolfOptimizer:
    """
    Grey Wolf Optimizer for hyperparameter search.

    Parameters
    ----------
    fitness_fn      : Callable[[np.ndarray], float]
                      Maps a normalised position vector ∈ [0,1]^n_dims
                      to a scalar fitness (higher = better).
    n_dims          : Dimensionality of the search space.
    n_wolves        : Population size (number of candidate solutions).
    max_iter        : Maximum number of iterations.
    seed            : Random seed for reproducibility.
    tol             : Convergence tolerance. Stop early if improvement
                      in best fitness over `patience` iterations < tol.
    patience        : Iterations to wait before early stopping.
    verbose         : Log progress each iteration.

    Example
    -------
    >>> gwo = GreyWolfOptimizer(fitness_fn=my_fn, n_dims=10, n_wolves=10)
    >>> result = gwo.run()
    >>> print(result.best_hyperparams)
    """

    def __init__(
        self,
        fitness_fn:  Callable[[np.ndarray], float],
        n_dims:      int,
        decode_fn:   Optional[Callable[[np.ndarray], Dict]] = None,
        n_wolves:    int   = 10,
        max_iter:    int   = 30,
        seed:        int   = 42,
        tol:         float = 1e-4,
        patience:    int   = 10,
        verbose:     bool  = True,
    ):
        self.fitness_fn = fitness_fn
        self.n_dims     = n_dims
        self.decode_fn  = decode_fn  # optional: position → hyperparams dict
        self.n_wolves   = n_wolves
        self.max_iter   = max_iter
        self.tol        = tol
        self.patience   = patience
        self.verbose    = verbose
        self.rng        = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(self) -> GWOResult:
        """
        Execute the full GWO optimisation loop.

        Returns
        -------
        GWOResult with best position, fitness, decoded hyperparams, history.
        """
        t_start = time.perf_counter()
        history = GWOHistory()

        # ── 1. Initialise wolf population ──────────────────────────────
        wolves = [
            WolfPosition(self.rng.uniform(0.0, 1.0, self.n_dims))
            for _ in range(self.n_wolves)
        ]

        if self.verbose:
            self._banner(f"Grey Wolf Optimizer — {self.n_dims}D space, "
                         f"{self.n_wolves} wolves, {self.max_iter} iterations")

        # ── 2. Initial evaluation ──────────────────────────────────────
        n_evals = 0
        for wolf in wolves:
            wolf.fitness = self.fitness_fn(wolf.position)
            n_evals += 1

        # Identify α, β, δ
        alpha, beta, delta = self._rank_leaders(wolves)

        if self.verbose:
            print(f"\n  {'Iter':>4}  {'α AUC':>8}  {'Mean AUC':>8}  {'a':>6}  {'Time':>7}")
            print(f"  {'----':>4}  {'------':>8}  {'--------':>8}  {'------':>6}  {'-------':>7}")

        # ── 3. Main iteration loop ─────────────────────────────────────
        stagnation = 0
        prev_best  = alpha.fitness

        for t in range(self.max_iter):
            t_iter = time.perf_counter()

            # Linearly decrease a: 2 → 0
            a = 2.0 - t * (2.0 / self.max_iter)

            # Update each omega wolf
            for wolf in wolves:
                wolf.position = self._update_position(
                    wolf.position, alpha.position, beta.position, delta.position, a
                )
                # Clip to [0, 1]
                wolf.position = np.clip(wolf.position, 0.0, 1.0)

                # Re-evaluate fitness
                wolf.fitness = self.fitness_fn(wolf.position)
                n_evals += 1

            # Re-rank leaders
            alpha, beta, delta = self._rank_leaders(wolves)

            # Record history
            fitnesses = [w.fitness for w in wolves]
            history.iteration_best.append(float(alpha.fitness))
            history.iteration_mean.append(float(np.mean(fitnesses)))
            history.alpha_positions.append(alpha.position.tolist())
            history.elapsed_seconds.append(time.perf_counter() - t_start)

            if self.verbose:
                elapsed = time.perf_counter() - t_iter
                print(f"  {t+1:>4}  {alpha.fitness:>8.4f}  "
                      f"{np.mean(fitnesses):>8.4f}  {a:>6.3f}  {elapsed:>6.1f}s")

            # Early stopping check
            improvement = alpha.fitness - prev_best
            if improvement < self.tol:
                stagnation += 1
                if stagnation >= self.patience:
                    if self.verbose:
                        print(f"\n  ⚡ Early stopping at iter {t+1} "
                              f"(no improvement ≥ {self.tol} for {self.patience} iters)")
                    break
            else:
                stagnation = 0
            prev_best = alpha.fitness

        # ── 4. Decode best position ────────────────────────────────────
        best_hparams = {}
        if self.decode_fn is not None:
            best_hparams = self.decode_fn(alpha.position)

        total_time = time.perf_counter() - t_start

        result = GWOResult(
            best_position    = alpha.position.tolist(),
            best_fitness     = float(alpha.fitness),
            best_hyperparams = best_hparams,
            history          = history,
            n_evaluations    = n_evals,
            total_time_s     = total_time,
            converged        = stagnation >= self.patience,
        )

        if self.verbose:
            self._print_summary(result)

        return result

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _update_position(
        self,
        X:       np.ndarray,
        X_alpha: np.ndarray,
        X_beta:  np.ndarray,
        X_delta: np.ndarray,
        a:       float,
    ) -> np.ndarray:
        """
        Compute new wolf position using the three leader wolves.

        Each leader contributes one movement vector; the new position
        is the average of the three movements.
        """
        movements = []
        for X_leader in (X_alpha, X_beta, X_delta):
            r1 = self.rng.uniform(0.0, 1.0, self.n_dims)
            r2 = self.rng.uniform(0.0, 1.0, self.n_dims)

            A = 2.0 * a * r1 - a          # ∈ [-a, a]
            C = 2.0 * r2                   # ∈ [0, 2]

            D = np.abs(C * X_leader - X)   # distance to leader
            movement = X_leader - A * D    # step toward leader

            movements.append(movement)

        # New position = average of three leader-guided movements
        return (movements[0] + movements[1] + movements[2]) / 3.0

    def _rank_leaders(
        self, wolves: List[WolfPosition]
    ) -> Tuple[WolfPosition, WolfPosition, WolfPosition]:
        """
        Return (alpha, beta, delta) = top-3 wolves by fitness.
        Each is a *copy* so mutations don't affect the stored best.
        """
        sorted_wolves = sorted(wolves, key=lambda w: w.fitness, reverse=True)
        alpha = sorted_wolves[0].copy()
        beta  = sorted_wolves[1].copy() if len(sorted_wolves) > 1 else alpha.copy()
        delta = sorted_wolves[2].copy() if len(sorted_wolves) > 2 else beta.copy()
        return alpha, beta, delta

    @staticmethod
    def _banner(msg: str) -> None:
        sep = "─" * (len(msg) + 4)
        print(f"\n┌{sep}┐")
        print(f"│  {msg}  │")
        print(f"└{sep}┘")

    @staticmethod
    def _print_summary(result: GWOResult) -> None:
        print("\n╔══════════════════════════════════════════════════════╗")
        print("║   🐺  GWO Optimisation Complete                      ║")
        print("╚══════════════════════════════════════════════════════╝")
        print(f"  Best fitness (val AUC)  : {result.best_fitness:.4f}")
        print(f"  Total evaluations       : {result.n_evaluations}")
        print(f"  Total time              : {result.total_time_s:.1f}s")
        print(f"  Converged early         : {result.converged}")
        if result.best_hyperparams:
            print("\n  ─── Best Hyperparameters ───────────────────────────")
            for k, v in result.best_hyperparams.items():
                if isinstance(v, float):
                    print(f"    {k:<28}: {v:.6g}")
                else:
                    print(f"    {k:<28}: {v}")
