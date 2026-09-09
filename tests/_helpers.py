"""Provides finite-difference utilities for validating analytic gradients."""

from __future__ import annotations

import numpy as np


def finite_difference_gradient(func, x: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    """Estimate a gradient with centered finite differences."""
    base = np.asarray(x, dtype=float)
    grad = np.zeros_like(base)
    for idx in range(base.size):
        delta = np.zeros_like(base)
        delta[idx] = eps
        grad[idx] = (func(base + delta) - func(base - delta)) / (2.0 * eps)
    return grad
