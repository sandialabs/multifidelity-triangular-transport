"""Provide shared array and standardization utilities."""

from __future__ import annotations

from typing import Iterable

import numpy as np

from .params import StandardizationParams


def summarize_minimize_result(result, optimization) -> dict[str, object]:
    """Return one serializable summary for every SciPy training path."""
    jac = getattr(result, "jac", None)
    return {
        "success": bool(result.success),
        "status": int(result.status),
        "message": str(result.message),
        "nit": None if getattr(result, "nit", None) is None else int(result.nit),
        "nfev": None if getattr(result, "nfev", None) is None else int(result.nfev),
        "njev": None if getattr(result, "njev", None) is None else int(result.njev),
        "fun": None if getattr(result, "fun", None) is None else float(result.fun),
        "grad_norm": None if jac is None else float(np.linalg.norm(np.asarray(jac, dtype=float))),
        "optimizer": str(optimization.optimizer),
        "gtol": float(optimization.gtol),
        "maxiter": int(optimization.maxiter),
        "reg_cst": float(optimization.reg_cst),
    }


def ensure_2d(array: np.ndarray, expected_dim: int | None = None) -> np.ndarray:
    """Coerce input data to a 2D floating-point array."""
    out = np.asarray(array, dtype=float)
    if out.ndim == 1:
        out = out[None, :]
    if out.ndim != 2:
        raise ValueError(f"Expected a 2D array, received shape {out.shape}.")
    if expected_dim is not None and out.shape[1] != expected_dim:
        raise ValueError(f"Expected second dimension {expected_dim}, received {out.shape[1]}.")
    return out


def standardize_dataset(data: np.ndarray) -> tuple[np.ndarray, StandardizationParams]:
    """Standardize one dataset and return the fitted parameters."""
    array = ensure_2d(data)
    mean = np.mean(array, axis=0)
    std = np.std(array, axis=0)
    if np.any(std <= 0.0):
        raise ValueError("Standardization failed because at least one feature has zero variance.")
    standardized = (array - mean) / std
    return standardized, StandardizationParams(mean=mean, std=std)


def apply_standardization(data: np.ndarray, params: StandardizationParams) -> np.ndarray:
    """Apply stored standardization parameters to a dataset."""
    array = ensure_2d(data, expected_dim=params.mean.shape[0])
    return (array - params.mean) / params.std


def invert_standardization(data: np.ndarray, params: StandardizationParams) -> np.ndarray:
    """Map standardized data back to the original coordinate system."""
    array = ensure_2d(data, expected_dim=params.mean.shape[0])
    return array * params.std + params.mean


def validate_fixed_indices(fixed_indices: Iterable[int], input_dim: int) -> list[int]:
    """Validate sorted unique conditioning indices for inverse solves."""
    indices = [int(index) for index in fixed_indices]
    if indices != sorted(indices):
        raise ValueError("fixed_indices must be sorted in ascending order.")
    if len(set(indices)) != len(indices):
        raise ValueError("fixed_indices must be unique.")
    for index in indices:
        if index < 0 or index >= input_dim:
            raise ValueError(f"fixed index {index} is out of bounds for dimension {input_dim}.")
    return indices

def validate_leading_fixed_indices(fixed_indices: Iterable[int], input_dim: int) -> list[int]:
    """Validate conditioning indices for a leading triangular prefix."""
    indices = validate_fixed_indices(fixed_indices, input_dim)
    expected = list(range(len(indices)))
    if indices != expected:
        raise ValueError(
            "conditioning requires fixed_indices to be a leading "
            f"prefix {expected}, received {indices}."
        )
    return indices
