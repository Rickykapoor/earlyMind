"""
src/optimization/search_space.py
==================================
Hyperparameter search space definition for EarlyMind GWO.

Each dimension in the GWO position vector maps to one hyperparameter.
The GWO works internally with normalised values in [0, 1]; this module
handles encoding → GWO space and decoding → typed Python values.

Supported dimension types:
  - "float"  : continuous, linear scale
  - "float_log" : continuous, log scale (good for lr, weight_decay)
  - "int"    : discrete integers
  - "pow2"   : powers of 2 (good for batch_size, embed_dim)
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np


# ---------------------------------------------------------------------------
# Dimension descriptor
# ---------------------------------------------------------------------------

@dataclass
class Dimension:
    """
    One hyperparameter dimension in the search space.

    Attributes
    ----------
    name    : Python key in the hyperparameter dict
    lo      : Lower bound (in original scale)
    hi      : Upper bound (in original scale)
    dtype   : "float" | "float_log" | "int" | "pow2"
    group   : Optional grouping tag (e.g. "training", "model")
    """
    name:   str
    lo:     float
    hi:     float
    dtype:  str    = "float"
    group:  str    = "misc"

    def __post_init__(self):
        valid = {"float", "float_log", "int", "pow2"}
        if self.dtype not in valid:
            raise ValueError(f"dtype must be one of {valid}, got {self.dtype!r}")
        if self.hi <= self.lo:
            raise ValueError(f"hi ({self.hi}) must be > lo ({self.lo}) for {self.name}")

    # ------------------------------------------------------------------
    # Encode: original value → [0, 1]
    # ------------------------------------------------------------------

    def encode(self, value: float) -> float:
        """Map an original-scale value to the normalised [0, 1] range."""
        if self.dtype == "float_log":
            lo_l = math.log10(self.lo)
            hi_l = math.log10(self.hi)
            return (math.log10(max(value, self.lo)) - lo_l) / (hi_l - lo_l)
        elif self.dtype == "pow2":
            # Find index in powers-of-2 list
            opts = self._pow2_options()
            idx  = min(range(len(opts)), key=lambda i: abs(opts[i] - value))
            return idx / max(len(opts) - 1, 1)
        else:  # float or int
            return (value - self.lo) / (self.hi - self.lo)

    # ------------------------------------------------------------------
    # Decode: [0, 1] → original value
    # ------------------------------------------------------------------

    def decode(self, x: float) -> Any:
        """Map a normalised value in [0, 1] to the original-scale typed value."""
        x = float(np.clip(x, 0.0, 1.0))

        if self.dtype == "float":
            return float(self.lo + x * (self.hi - self.lo))

        elif self.dtype == "float_log":
            lo_l = math.log10(self.lo)
            hi_l = math.log10(self.hi)
            return float(10 ** (lo_l + x * (hi_l - lo_l)))

        elif self.dtype == "int":
            val = self.lo + x * (self.hi - self.lo)
            return int(round(val))

        elif self.dtype == "pow2":
            opts = self._pow2_options()
            idx  = int(round(x * (len(opts) - 1)))
            idx  = max(0, min(idx, len(opts) - 1))
            return int(opts[idx])

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _pow2_options(self) -> List[int]:
        """All powers of 2 in [lo, hi]."""
        start_exp = math.ceil(math.log2(max(self.lo, 1)))
        end_exp   = math.floor(math.log2(max(self.hi, 1)))
        return [2 ** e for e in range(start_exp, end_exp + 1)]

    def __repr__(self) -> str:
        return (f"Dimension({self.name!r}, [{self.lo}, {self.hi}], "
                f"dtype={self.dtype!r}, group={self.group!r})")


# ---------------------------------------------------------------------------
# Search Space
# ---------------------------------------------------------------------------

class SearchSpace:
    """
    Full hyperparameter search space for EarlyMind GWO.

    Defines all 10 hyperparameter dimensions, provides encode/decode,
    and enforces bounds.

    Usage
    -----
    >>> space = SearchSpace()
    >>> space.summary()
    >>> hparams = space.decode(np.array([0.3, 0.7, ...]))
    >>> position = space.encode(hparams)
    """

    #: Default EarlyMind dimensions (edit here to change search space)
    DEFAULT_DIMS: List[Dimension] = [
        # ── Training hyperparameters ─────────────────────────────────
        Dimension("lr",                   1e-5,  1e-2,  "float_log",  "training"),
        Dimension("weight_decay",         1e-6,  1e-2,  "float_log",  "training"),
        Dimension("batch_size",           8,     64,    "pow2",       "training"),
        Dimension("focal_alpha",          0.10,  0.90,  "float",      "training"),
        Dimension("focal_gamma",          0.50,  5.00,  "float",      "training"),
        Dimension("severity_loss_weight", 0.10,  1.00,  "float",      "training"),
        # ── Model architecture ───────────────────────────────────────
        Dimension("embed_dim",            64,    256,   "pow2",       "model"),
        Dimension("dropout",              0.00,  0.50,  "float",      "model"),
        Dimension("fusion_heads",         2,     8,     "int",        "model"),
        Dimension("fusion_layers",        1,     6,     "int",        "model"),
    ]

    def __init__(self, dims: Optional[List[Dimension]] = None):
        self.dims = dims if dims is not None else list(self.DEFAULT_DIMS)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def n_dims(self) -> int:
        return len(self.dims)

    @property
    def dim_names(self) -> List[str]:
        return [d.name for d in self.dims]

    # ------------------------------------------------------------------
    # Encode / Decode
    # ------------------------------------------------------------------

    def decode(self, position: np.ndarray) -> Dict[str, Any]:
        """
        Convert a GWO position vector (n_dims values in [0,1]) to a
        typed hyperparameter dictionary.

        Parameters
        ----------
        position : np.ndarray, shape (n_dims,)

        Returns
        -------
        dict with keys from dim names, values in original scale / correct type.
        """
        if len(position) != self.n_dims:
            raise ValueError(
                f"Expected {self.n_dims} dimensions, got {len(position)}"
            )
        return {dim.name: dim.decode(x)
                for dim, x in zip(self.dims, position)}

    def encode(self, hparams: Dict[str, Any]) -> np.ndarray:
        """
        Convert a typed hyperparameter dict back to a normalised position vector.
        Useful for seeding GWO with known good starting points.
        """
        return np.array([
            dim.encode(hparams[dim.name])
            for dim in self.dims
        ], dtype=np.float64)

    def random_position(self, rng: Optional[np.random.Generator] = None) -> np.ndarray:
        """Sample a random position uniformly from [0, 1]^n_dims."""
        rng = rng or np.random.default_rng()
        return rng.uniform(0.0, 1.0, self.n_dims)

    def clip(self, position: np.ndarray) -> np.ndarray:
        """Clip position to [0, 1]^n_dims."""
        return np.clip(position, 0.0, 1.0)

    # ------------------------------------------------------------------
    # Utilities
    # ------------------------------------------------------------------

    def summary(self) -> None:
        """Print a formatted summary of all dimensions."""
        print("\n┌────────────────────────────────────────────────────────────────┐")
        print("│  EarlyMind GWO Hyperparameter Search Space                     │")
        print("└────────────────────────────────────────────────────────────────┘")
        print(f"  {'#':>2}  {'Name':<28} {'Type':<12} {'Min':<12} {'Max':<12} Group")
        print(f"  {'--':>2}  {'----':<28} {'----':<12} {'---':<12} {'---':<12} -----")
        for i, d in enumerate(self.dims):
            lo_str = f"{d.lo:.2g}"
            hi_str = f"{d.hi:.2g}"
            print(f"  {i:>2}  {d.name:<28} {d.dtype:<12} {lo_str:<12} {hi_str:<12} {d.group}")
        print(f"\n  Total: {self.n_dims} dimensions\n")

    def describe(self, position: np.ndarray) -> str:
        """Return a human-readable string of a decoded position."""
        hparams = self.decode(position)
        lines = []
        for name, val in hparams.items():
            if isinstance(val, float):
                lines.append(f"  {name:<28}: {val:.6g}")
            else:
                lines.append(f"  {name:<28}: {val}")
        return "\n".join(lines)
