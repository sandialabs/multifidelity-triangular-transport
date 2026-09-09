"""Build and evaluate Hermite-function bases used by transport maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_NORMALIZATION_CACHE: dict[tuple[tuple[tuple[int, ...], ...], float, int, float], np.ndarray] = {}


def generate_multi_indices(input_dimension: int, total_order: int) -> list[tuple[int, ...]]:
    """Enumerate all multi-indices with total degree up to ``total_order``."""
    if input_dimension < 1:
        raise ValueError("input_dimension must be positive.")
    if total_order < 0:
        raise ValueError("total_order must be non-negative.")

    multi_indices: list[tuple[int, ...]] = []

    def recurse(prefix: tuple[int, ...], remaining_order: int, variables_left: int) -> None:
        if variables_left == 1:
            multi_indices.append(prefix + (remaining_order,))
            return
        for order in range(remaining_order + 1):
            recurse(prefix + (order,), remaining_order - order, variables_left - 1)

    for order in range(total_order + 1):
        recurse((), order, input_dimension)
    return multi_indices


def _monic_hermite_table(max_order: int, x: np.ndarray) -> np.ndarray:
    """Build monic probabilists' Hermite polynomials up to ``max_order``."""
    samples = np.asarray(x, dtype=float).reshape(-1)
    table = np.empty((max_order + 1, samples.size), dtype=float)
    table[0] = 1.0
    if max_order >= 1:
        table[1] = samples
    for order in range(1, max_order):
        table[order + 1] = samples * table[order] - order * table[order - 1]
    return table


def _eval_1d_tables(
    orders: Sequence[int],
    x: np.ndarray,
    sigma: float,
    include_derivatives: bool = False,
) -> tuple[dict[int, np.ndarray], dict[int, np.ndarray] | None]:
    """Evaluate all requested 1D orders, optionally with derivatives."""
    unique_orders = tuple(sorted(set(int(order) for order in orders)))
    if len(unique_orders) == 0:
        return {}, {} if include_derivatives else None

    samples = np.asarray(x, dtype=float).reshape(-1)
    polynomial_table = _monic_hermite_table(unique_orders[-1], samples)
    gaussian = np.exp(-(samples ** 2) / sigma)

    values: dict[int, np.ndarray] = {}
    derivatives: dict[int, np.ndarray] | None = {} if include_derivatives else None
    zeros = np.zeros_like(samples)
    ones = np.ones_like(samples)

    for order in unique_orders:
        if order > 1:
            values[order] = polynomial_table[order] * gaussian
        else:
            values[order] = polynomial_table[order]

        if derivatives is None:
            continue
        if order == 0:
            derivatives[order] = zeros
        elif order == 1:
            derivatives[order] = ones
        else:
            derivatives[order] = (
                order * polynomial_table[order - 1] * gaussian
                - (2.0 / sigma) * samples * polynomial_table[order] * gaussian
            )
    return values, derivatives


def _eval_1d(order: int, x: np.ndarray, sigma: float) -> np.ndarray:
    """Evaluate one normalized 1D Hermite-function factor."""
    values, _ = _eval_1d_tables([order], x, sigma)
    return values[order]


def compute_normalization(multi_indices: Sequence[tuple[int, ...]], sigma: float, grid_size: int = 257, bound: float = 6.0) -> np.ndarray:
    """Estimate per-basis normalization factors on a bounded grid."""
    if len(multi_indices) == 0:
        return np.zeros(0, dtype=float)
    cache_key = (
        tuple(tuple(int(order) for order in multi_index) for multi_index in multi_indices),
        float(sigma),
        int(grid_size),
        float(bound),
    )
    cached = _NORMALIZATION_CACHE.get(cache_key)
    if cached is not None:
        return cached.copy()

    input_dim = len(multi_indices[0])
    grid = np.linspace(-bound, bound, grid_size)
    one_d_max: list[dict[int, float]] = []
    for axis in range(input_dim):
        orders = sorted({multi_index[axis] for multi_index in multi_indices})
        axis_table = {}
        value_table, _ = _eval_1d_tables(orders, grid, sigma)
        for order in orders:
            axis_table[order] = max(np.max(np.abs(value_table[order])), 1.0)
        one_d_max.append(axis_table)

    scales = np.ones(len(multi_indices), dtype=float)
    for idx, multi_index in enumerate(multi_indices):
        denom = 1.0
        for axis, order in enumerate(multi_index):
            denom *= one_d_max[axis][order]
        scales[idx] = 1.0 / denom
    _NORMALIZATION_CACHE[cache_key] = scales.copy()
    return scales.copy()


def evaluate_multiindex_matrix(
    x: np.ndarray,
    multi_indices: Sequence[tuple[int, ...]],
    normalization: np.ndarray,
    sigma: float,
) -> np.ndarray:
    """Evaluate a multivariate Hermite basis matrix on sample points."""
    samples = np.asarray(x, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2D sample array, received shape {samples.shape}.")
    if len(multi_indices) == 0:
        return np.zeros((samples.shape[0], 0), dtype=float)

    input_dim = samples.shape[1]
    if any(len(multi_index) != input_dim for multi_index in multi_indices):
        raise ValueError("All multi-indices must have the same dimension as x.")

    unique_orders = [sorted({multi_index[axis] for multi_index in multi_indices}) for axis in range(input_dim)]
    axis_tables: list[dict[int, np.ndarray]] = []
    for axis, orders in enumerate(unique_orders):
        axis_table, _ = _eval_1d_tables(orders, samples[:, axis], sigma)
        axis_tables.append(axis_table)

    matrix = np.ones((samples.shape[0], len(multi_indices)), dtype=float)
    for column, multi_index in enumerate(multi_indices):
        values = np.ones(samples.shape[0], dtype=float)
        for axis, order in enumerate(multi_index):
            values *= axis_tables[axis][order]
        matrix[:, column] = values * normalization[column]
    return matrix


def evaluate_multiindex_derivative_matrix(
    x: np.ndarray,
    multi_indices: Sequence[tuple[int, ...]],
    normalization: np.ndarray,
    sigma: float,
    axis: int,
) -> np.ndarray:
    """Evaluate derivatives of a multivariate Hermite basis matrix."""
    samples = np.asarray(x, dtype=float)
    if samples.ndim != 2:
        raise ValueError(f"Expected a 2D sample array, received shape {samples.shape}.")
    if len(multi_indices) == 0:
        return np.zeros((samples.shape[0], 0), dtype=float)

    input_dim = samples.shape[1]
    if axis < 0 or axis >= input_dim:
        raise ValueError(f"axis must be in [0, {input_dim}), received {axis}.")
    if any(len(multi_index) != input_dim for multi_index in multi_indices):
        raise ValueError("All multi-indices must have the same dimension as x.")

    unique_orders = [sorted({multi_index[idx] for multi_index in multi_indices}) for idx in range(input_dim)]
    axis_tables: list[dict[int, np.ndarray]] = []
    derivative_table: dict[int, np.ndarray] = {}
    for idx, orders in enumerate(unique_orders):
        axis_table, derivatives = _eval_1d_tables(
            orders,
            samples[:, idx],
            sigma,
            include_derivatives=(idx == axis),
        )
        axis_tables.append(axis_table)
        if idx == axis:
            assert derivatives is not None
            derivative_table = derivatives

    matrix = np.ones((samples.shape[0], len(multi_indices)), dtype=float)
    for column, multi_index in enumerate(multi_indices):
        values = np.ones(samples.shape[0], dtype=float)
        for idx, order in enumerate(multi_index):
            if idx == axis:
                values *= derivative_table[order]
            else:
                values *= axis_tables[idx][order]
        matrix[:, column] = values * normalization[column]
    return matrix


@dataclass
class MultivariateHermiteFunction:
    """Wrap one multivariate Hermite basis function as an evaluable object."""
    multi_index: tuple[int, ...]
    sigma: float = 30.0

    def __post_init__(self) -> None:
        """Precompute the normalization constant for the basis function."""
        self._normalization = compute_normalization([self.multi_index], sigma=self.sigma)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the basis function on a batch of samples."""
        samples = np.asarray(x, dtype=float)
        if samples.ndim == 1:
            samples = samples[None, :]
        if samples.ndim != 2:
            raise ValueError(f"Expected a 2D sample array, received shape {samples.shape}.")
        if samples.shape[1] != len(self.multi_index):
            raise ValueError("Sample dimension must match the multi-index dimension.")

        values = np.ones(samples.shape[0], dtype=float)
        for axis, order in enumerate(self.multi_index):
            values *= _eval_1d(order, samples[:, axis], self.sigma)
        return values * self._normalization[0]
