"""Compute Gaussianity diagnostics for pushed-forward transport-map samples."""

from __future__ import annotations

from typing import Any

import numpy as np


def _validate_samples(samples: np.ndarray) -> np.ndarray:
    """Validate and return a 2D sample array."""
    array = np.asarray(samples, dtype=float)
    if array.ndim != 2:
        raise ValueError(f"samples must be a 2D array, received shape {array.shape}.")
    if array.shape[0] == 0:
        raise ValueError("samples must contain at least one sample.")
    return array


def _symmetrize(matrix: np.ndarray) -> np.ndarray:
    """Return the symmetric part of a square matrix."""
    return 0.5 * (matrix + matrix.T)


def _validate_target_mean(target_mean: np.ndarray | None, input_dim: int) -> np.ndarray:
    """Return a validated target mean vector."""
    if target_mean is None:
        return np.zeros(input_dim, dtype=float)
    mean = np.asarray(target_mean, dtype=float).reshape(-1)
    if mean.shape != (input_dim,):
        raise ValueError(f"target_mean must have shape ({input_dim},), received shape {mean.shape}.")
    return mean


def _validate_target_covariance(target_cov: np.ndarray | None, input_dim: int) -> np.ndarray:
    """Return a validated target covariance matrix."""
    if target_cov is None:
        return np.eye(input_dim, dtype=float)
    covariance = np.asarray(target_cov, dtype=float)
    if covariance.shape != (input_dim, input_dim):
        raise ValueError(
            f"target_cov must have shape ({input_dim}, {input_dim}), received shape {covariance.shape}."
        )
    return _symmetrize(covariance)


def _pairwise_squared_distances(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Compute pairwise squared Euclidean distances."""
    x_sq = np.sum(x * x, axis=1)[:, None]
    y_sq = np.sum(y * y, axis=1)[None, :]
    squared = x_sq + y_sq - 2.0 * np.dot(x, y.T)
    return np.maximum(squared, 0.0)


def _gaussian_kernel(squared_distances: np.ndarray, bandwidth: float) -> np.ndarray:
    """Evaluate a Gaussian kernel matrix from squared distances."""
    if bandwidth <= 0.0:
        raise ValueError("bandwidth must be positive.")
    return np.exp(-squared_distances / (2.0 * bandwidth * bandwidth))


def _select_bandwidth_subset(
    pooled: np.ndarray,
    subset_size: int,
    random_seed: int | None,
) -> np.ndarray:
    """Select a deterministic subset of pooled samples for bandwidth estimation."""
    if subset_size <= 0:
        raise ValueError("bandwidth_subset_size must be positive.")
    if pooled.shape[0] <= subset_size:
        return pooled
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(pooled.shape[0], size=subset_size, replace=False)
    return pooled[indices]


def _median_heuristic_bandwidth(
    samples: np.ndarray,
    reference_samples: np.ndarray,
    subset_size: int = 2000,
    random_seed: int | None = 0,
) -> float:
    """Estimate a Gaussian-kernel bandwidth from pooled pairwise distances."""
    pooled = np.vstack([samples, reference_samples])
    pooled = _select_bandwidth_subset(pooled, subset_size=subset_size, random_seed=random_seed)
    pairwise_sq = _pairwise_squared_distances(pooled, pooled)
    pairwise_dist = np.sqrt(pairwise_sq)
    mask = ~np.eye(pairwise_dist.shape[0], dtype=bool)
    off_diagonal = pairwise_dist[mask]
    positive = off_diagonal[off_diagonal > 0.0]
    if positive.size == 0:
        return 1.0
    bandwidth = float(np.median(positive))
    if not np.isfinite(bandwidth) or bandwidth <= 0.0:
        return 1.0
    return bandwidth


def median_heuristic_bandwidth(
    samples: np.ndarray,
    reference_samples: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
    random_seed: int | None = 0,
    bandwidth_subset_size: int = 2000,
) -> float:
    """Estimate a Gaussian-kernel bandwidth from samples and reference data.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud used to estimate the bandwidth.
    reference_samples : array-like, optional
        Reference cloud with the same coordinate dimension. When omitted, a
        Gaussian cloud is drawn from ``target_mean`` and ``target_cov``.
    target_mean : array-like, shape (input_dim,), optional
        Mean for generated Gaussian reference samples; defaults to zero.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Covariance for generated Gaussian reference samples; defaults to the
        identity.
    random_seed : int, optional, default=0
        Seed used for generated reference samples and subset selection.
    bandwidth_subset_size : int, default=2000
        Maximum number of rows used by the median-distance calculation.

    Returns
    -------
    float
        Positive median-heuristic bandwidth.
    """
    validated = _validate_samples(samples)
    input_dim = validated.shape[1]
    mean = _validate_target_mean(target_mean, input_dim)
    covariance = _validate_target_covariance(target_cov, input_dim)
    reference = _resolve_reference_samples(
        validated,
        reference_samples=reference_samples,
        target_mean=mean,
        target_cov=covariance,
        random_seed=random_seed,
    )
    return _median_heuristic_bandwidth(
        validated,
        reference,
        subset_size=bandwidth_subset_size,
        random_seed=random_seed,
    )


def _inverse_square_root(matrix: np.ndarray, jitter: float) -> np.ndarray:
    """Compute an inverse square root for a symmetric positive matrix."""
    regularized = _symmetrize(matrix) + jitter * np.eye(matrix.shape[0], dtype=float)
    eigenvalues, eigenvectors = np.linalg.eigh(regularized)
    eigenvalues = np.clip(eigenvalues, jitter, None)
    inv_sqrt = eigenvectors @ np.diag(1.0 / np.sqrt(eigenvalues)) @ eigenvectors.T
    return _symmetrize(inv_sqrt)


def _draw_gaussian_reference_samples(
    num_samples: int,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    random_seed: int | None,
) -> np.ndarray:
    """Draw Gaussian reference samples for MMD comparisons."""
    rng = np.random.default_rng(random_seed)
    return rng.multivariate_normal(mean=target_mean, cov=target_cov, size=num_samples)


def _sum_off_diagonal_kernel_blockwise(samples: np.ndarray, bandwidth: float, block_size: int) -> float:
    """Accumulate an exact off-diagonal kernel sum without materializing a full matrix."""
    total = 0.0
    num_samples = samples.shape[0]
    for start_i in range(0, num_samples, block_size):
        stop_i = min(start_i + block_size, num_samples)
        x_block = samples[start_i:stop_i]
        for start_j in range(0, num_samples, block_size):
            stop_j = min(start_j + block_size, num_samples)
            y_block = samples[start_j:stop_j]
            kernel_block = _gaussian_kernel(_pairwise_squared_distances(x_block, y_block), bandwidth)
            if start_i == start_j:
                total += float(np.sum(kernel_block) - np.trace(kernel_block))
            else:
                total += float(np.sum(kernel_block))
    return total


def _sum_cross_kernel_blockwise(x: np.ndarray, y: np.ndarray, bandwidth: float, block_size: int) -> float:
    """Accumulate an exact cross-kernel sum without materializing a full matrix."""
    total = 0.0
    for start_i in range(0, x.shape[0], block_size):
        stop_i = min(start_i + block_size, x.shape[0])
        x_block = x[start_i:stop_i]
        for start_j in range(0, y.shape[0], block_size):
            stop_j = min(start_j + block_size, y.shape[0])
            y_block = y[start_j:stop_j]
            kernel_block = _gaussian_kernel(_pairwise_squared_distances(x_block, y_block), bandwidth)
            total += float(np.sum(kernel_block))
    return total


def _dense_squared_mmd_gaussian_kernel(samples: np.ndarray, reference: np.ndarray, bandwidth: float) -> float:
    """Compute the exact unbiased squared MMD using dense kernel matrices."""
    n = samples.shape[0]
    m = reference.shape[0]
    k_xx = _gaussian_kernel(_pairwise_squared_distances(samples, samples), bandwidth)
    k_yy = _gaussian_kernel(_pairwise_squared_distances(reference, reference), bandwidth)
    k_xy = _gaussian_kernel(_pairwise_squared_distances(samples, reference), bandwidth)
    term_xx = (np.sum(k_xx) - np.trace(k_xx)) / (n * (n - 1))
    term_yy = (np.sum(k_yy) - np.trace(k_yy)) / (m * (m - 1))
    term_xy = np.mean(k_xy)
    return float(term_xx + term_yy - 2.0 * term_xy)


def _blockwise_squared_mmd_gaussian_kernel(
    samples: np.ndarray,
    reference: np.ndarray,
    bandwidth: float,
    block_size: int,
) -> float:
    """Compute the exact unbiased squared MMD by blockwise kernel accumulation."""
    n = samples.shape[0]
    m = reference.shape[0]
    term_xx = _sum_off_diagonal_kernel_blockwise(samples, bandwidth, block_size) / (n * (n - 1))
    term_yy = _sum_off_diagonal_kernel_blockwise(reference, bandwidth, block_size) / (m * (m - 1))
    term_xy = _sum_cross_kernel_blockwise(samples, reference, bandwidth, block_size) / (n * m)
    return float(term_xx + term_yy - 2.0 * term_xy)


def _resolve_reference_samples(
    samples: np.ndarray,
    reference_samples: np.ndarray | None,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    random_seed: int | None,
) -> np.ndarray:
    """Return validated or generated reference samples for MMD comparisons."""
    input_dim = samples.shape[1]
    if reference_samples is None:
        return _draw_gaussian_reference_samples(
            num_samples=samples.shape[0],
            target_mean=target_mean,
            target_cov=target_cov,
            random_seed=random_seed,
        )
    reference = _validate_samples(reference_samples)
    if reference.shape[1] != input_dim:
        raise ValueError(
            f"reference_samples must have shape (n, {input_dim}), received shape {reference.shape}."
        )
    return reference


def _squared_mmd_and_bandwidth(
    samples: np.ndarray,
    reference_samples: np.ndarray | None,
    target_mean: np.ndarray,
    target_cov: np.ndarray,
    bandwidth: float | None,
    bandwidth_mode: str,
    random_seed: int | None,
    bandwidth_subset_size: int,
    full_matrix_threshold: int,
    block_size: int,
) -> tuple[float, float]:
    """Compute squared MMD and report the bandwidth actually used."""
    if full_matrix_threshold < 1:
        raise ValueError("full_matrix_threshold must be positive.")
    if block_size < 1:
        raise ValueError("block_size must be positive.")
    reference = _resolve_reference_samples(
        samples,
        reference_samples=reference_samples,
        target_mean=target_mean,
        target_cov=target_cov,
        random_seed=random_seed,
    )

    n = samples.shape[0]
    m = reference.shape[0]
    if n < 2 or m < 2:
        raise ValueError("Need at least two samples in each sample cloud for unbiased squared MMD.")

    if bandwidth is None:
        if bandwidth_mode != "median_heuristic":
            raise ValueError(f"Unsupported bandwidth_mode '{bandwidth_mode}'.")
        kernel_bandwidth = _median_heuristic_bandwidth(
            samples,
            reference,
            subset_size=bandwidth_subset_size,
            random_seed=random_seed,
        )
    else:
        kernel_bandwidth = float(bandwidth)
        if kernel_bandwidth <= 0.0:
            raise ValueError("bandwidth must be positive.")

    if max(n, m) <= full_matrix_threshold:
        mmd = _dense_squared_mmd_gaussian_kernel(samples, reference, kernel_bandwidth)
    else:
        mmd = _blockwise_squared_mmd_gaussian_kernel(samples, reference, kernel_bandwidth, block_size)

    return mmd, kernel_bandwidth


def sample_mean(samples: np.ndarray) -> np.ndarray:
    """Compute the empirical mean of a sample cloud.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Samples in a common coordinate system.

    Returns
    -------
    numpy.ndarray, shape (input_dim,)
        Empirical mean.
    """
    validated = _validate_samples(samples)
    return np.mean(validated, axis=0)


def sample_covariance(samples: np.ndarray, ddof: int = 1) -> np.ndarray:
    """Compute the empirical covariance matrix of a sample cloud.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Samples in a common coordinate system.
    ddof : int, default=1
        Delta degrees of freedom used in the covariance denominator.

    Returns
    -------
    numpy.ndarray, shape (input_dim, input_dim)
        Empirical covariance.

    Raises
    ------
    ValueError
        If fewer than ``ddof + 1`` rows are supplied.
    """
    validated = _validate_samples(samples)
    num_samples, input_dim = validated.shape
    if num_samples <= ddof:
        raise ValueError(f"Need more than ddof={ddof} samples to compute a covariance matrix.")
    centered = validated - np.mean(validated, axis=0)
    covariance = centered.T @ centered / (num_samples - ddof)
    return covariance.reshape(input_dim, input_dim)


def first_moment_error(samples: np.ndarray, target_mean: np.ndarray | None = None) -> float:
    """Compute the Euclidean error in the first moment.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    target_mean : array-like, shape (input_dim,), optional
        Target mean; defaults to the zero vector.

    Returns
    -------
    float
        Euclidean norm of the mean error.
    """
    validated = _validate_samples(samples)
    mean = sample_mean(validated)
    target = _validate_target_mean(target_mean, validated.shape[1])
    return float(np.linalg.norm(mean - target, ord=2))


def second_moment_error(
    samples: np.ndarray,
    target_cov: np.ndarray | None = None,
    ddof: int = 1,
) -> float:
    """Compute the Frobenius error in the covariance.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Target covariance; defaults to identity.
    ddof : int, default=1
        Delta degrees of freedom used for the sample covariance.

    Returns
    -------
    float
        Frobenius norm of the covariance error.
    """
    validated = _validate_samples(samples)
    covariance = sample_covariance(validated, ddof=ddof)
    target = _validate_target_covariance(target_cov, validated.shape[1])
    return float(np.linalg.norm(covariance - target, ord="fro"))


def forstner_distance(
    samples: np.ndarray,
    target_cov: np.ndarray | None = None,
    ddof: int = 1,
    jitter: float = 1e-10,
) -> float:
    """Compute the Forstner distance from a sample to a target covariance.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Target covariance; defaults to identity.
    ddof : int, default=1
        Delta degrees of freedom used for the sample covariance.
    jitter : float, default=1e-10
        Positive regularization used in covariance square roots.

    Returns
    -------
    float
        Forstner distance between covariance matrices.
    """
    validated = _validate_samples(samples)
    covariance = sample_covariance(validated, ddof=ddof)
    target = _validate_target_covariance(target_cov, validated.shape[1])
    sample_inv_sqrt = _inverse_square_root(covariance, jitter=jitter)
    congruence = sample_inv_sqrt @ (target + jitter * np.eye(target.shape[0], dtype=float)) @ sample_inv_sqrt
    congruence = _symmetrize(congruence)
    eigenvalues = np.linalg.eigvalsh(congruence)
    eigenvalues = np.clip(eigenvalues, jitter, None)
    return float(np.sqrt(np.sum(np.log(eigenvalues) ** 2)))


def squared_mmd_gaussian_kernel(
    samples: np.ndarray,
    reference_samples: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
    bandwidth: float | None = None,
    bandwidth_mode: str = "median_heuristic",
    random_seed: int | None = 0,
    bandwidth_subset_size: int = 2000,
    full_matrix_threshold: int = 4000,
    block_size: int = 1024,
) -> float:
    """Compute an unbiased squared MMD against a Gaussian reference.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    reference_samples : array-like, optional
        Explicit Gaussian-reference samples. If omitted, samples are drawn
        from ``target_mean`` and ``target_cov``.
    target_mean : array-like, shape (input_dim,), optional
        Mean of generated reference samples; defaults to zero.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Covariance of generated reference samples; defaults to identity.
    bandwidth : float, optional
        Positive kernel bandwidth. When omitted, the median heuristic is used.
    bandwidth_mode : {"median_heuristic"}, default="median_heuristic"
        Rule used when ``bandwidth`` is omitted.
    random_seed : int, optional, default=0
        Seed for generated reference data and median-heuristic subsets.
    bandwidth_subset_size : int, default=2000
        Maximum rows used by the median-heuristic calculation.
    full_matrix_threshold : int, default=4000
        Use dense kernel matrices at or below this cloud size.
    block_size : int, default=1024
        Block size for exact large-cloud kernel accumulation.

    Returns
    -------
    float
        Unbiased squared Gaussian-kernel MMD.
    """
    validated = _validate_samples(samples)
    input_dim = validated.shape[1]
    mean = _validate_target_mean(target_mean, input_dim)
    covariance = _validate_target_covariance(target_cov, input_dim)
    mmd, _ = _squared_mmd_and_bandwidth(
        validated,
        reference_samples=reference_samples,
        target_mean=mean,
        target_cov=covariance,
        bandwidth=bandwidth,
        bandwidth_mode=bandwidth_mode,
        random_seed=random_seed,
        bandwidth_subset_size=bandwidth_subset_size,
        full_matrix_threshold=full_matrix_threshold,
        block_size=block_size,
    )
    return mmd


def multiscale_squared_mmd_gaussian_kernel(
    samples: np.ndarray,
    bandwidths: np.ndarray,
    reference_samples: np.ndarray | None = None,
    target_mean: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
    random_seed: int | None = 0,
    full_matrix_threshold: int = 4000,
    block_size: int = 1024,
) -> np.ndarray:
    """Compute unbiased squared MMD values for supplied bandwidths.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    bandwidths : array-like, shape (n_bandwidths,)
        Positive Gaussian-kernel bandwidths.
    reference_samples : array-like, optional
        Explicit reference cloud; otherwise a Gaussian cloud is generated.
    target_mean : array-like, shape (input_dim,), optional
        Mean of generated reference samples; defaults to zero.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Covariance of generated reference samples; defaults to identity.
    random_seed : int, optional, default=0
        Seed for generated reference samples.
    full_matrix_threshold : int, default=4000
        Dense-kernel cutoff.
    block_size : int, default=1024
        Block size for exact large-cloud accumulation.

    Returns
    -------
    numpy.ndarray, shape (n_bandwidths,)
        One unbiased squared MMD value per supplied bandwidth.
    """
    validated = _validate_samples(samples)
    input_dim = validated.shape[1]
    mean = _validate_target_mean(target_mean, input_dim)
    covariance = _validate_target_covariance(target_cov, input_dim)
    reference = _resolve_reference_samples(
        validated,
        reference_samples=reference_samples,
        target_mean=mean,
        target_cov=covariance,
        random_seed=random_seed,
    )
    resolved_bandwidths = np.asarray(bandwidths, dtype=float).reshape(-1)
    if resolved_bandwidths.size == 0:
        raise ValueError("bandwidths must contain at least one positive value.")
    if np.any(~np.isfinite(resolved_bandwidths)) or np.any(resolved_bandwidths <= 0.0):
        raise ValueError("bandwidths must contain only positive finite values.")

    values = np.zeros(resolved_bandwidths.shape[0], dtype=float)
    for idx, bandwidth in enumerate(resolved_bandwidths):
        if max(validated.shape[0], reference.shape[0]) <= full_matrix_threshold:
            values[idx] = _dense_squared_mmd_gaussian_kernel(validated, reference, float(bandwidth))
        else:
            values[idx] = _blockwise_squared_mmd_gaussian_kernel(validated, reference, float(bandwidth), block_size)
    return values


def gaussian_diagnostics(
    samples: np.ndarray,
    target_mean: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
    ddof: int = 1,
    jitter: float = 1e-10,
    random_seed: int | None = 0,
    bandwidth_subset_size: int = 2000,
    full_matrix_threshold: int = 4000,
    block_size: int = 1024,
) -> dict[str, Any]:
    """Summarize how closely a sample cloud matches a Gaussian target.

    Parameters
    ----------
    samples : array-like, shape (n_samples, input_dim)
        Sample cloud to assess.
    target_mean : array-like, shape (input_dim,), optional
        Gaussian target mean; defaults to zero.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Gaussian target covariance; defaults to identity.
    ddof : int, default=1
        Delta degrees of freedom for sample covariance.
    jitter : float, default=1e-10
        Covariance regularization for the Forstner distance.
    random_seed : int, optional, default=0
        Seed for MMD reference sampling.
    bandwidth_subset_size : int, default=2000
        Median-heuristic subset limit.
    full_matrix_threshold : int, default=4000
        Dense-kernel cutoff.
    block_size : int, default=1024
        Block size for exact large-cloud MMD.

    Returns
    -------
    dict
        Sample moments, target moments, moment errors, Forstner distance, MMD,
        and the bandwidth used.
    """
    validated = _validate_samples(samples)
    input_dim = validated.shape[1]
    mean = sample_mean(validated)
    covariance = sample_covariance(validated, ddof=ddof)
    target_mean_array = _validate_target_mean(target_mean, input_dim)
    target_covariance = _validate_target_covariance(target_cov, input_dim)
    mmd, bandwidth = _squared_mmd_and_bandwidth(
        validated,
        reference_samples=None,
        target_mean=target_mean_array,
        target_cov=target_covariance,
        bandwidth=None,
        bandwidth_mode="median_heuristic",
        random_seed=random_seed,
        bandwidth_subset_size=bandwidth_subset_size,
        full_matrix_threshold=full_matrix_threshold,
        block_size=block_size,
    )
    return {
        "num_samples": validated.shape[0],
        "input_dim": input_dim,
        "sample_mean": mean,
        "sample_covariance": covariance,
        "target_mean": target_mean_array,
        "target_covariance": target_covariance,
        "first_moment_error": float(np.linalg.norm(mean - target_mean_array, ord=2)),
        "second_moment_error": float(np.linalg.norm(covariance - target_covariance, ord="fro")),
        "forstner_distance": forstner_distance(validated, target_cov=target_covariance, ddof=ddof, jitter=jitter),
        "squared_mmd_gaussian_kernel": mmd,
        "gaussian_kernel_bandwidth": bandwidth,
    }


def map_gaussian_diagnostics(
    map_obj,
    target_samples: np.ndarray,
    target_mean: np.ndarray | None = None,
    target_cov: np.ndarray | None = None,
    ddof: int = 1,
    jitter: float = 1e-10,
    random_seed: int | None = 0,
    bandwidth_subset_size: int = 2000,
    full_matrix_threshold: int = 4000,
    block_size: int = 1024,
) -> dict[str, Any]:
    """Push target samples through a trained map and summarize Gaussianity.

    Parameters
    ----------
    map_obj : TriangularMap, HierarchicalTriangularMap, or NonHierarchicalTriangularMap
        Trained map exposing ``evaluate`` from target to reference coordinates.
    target_samples : array-like, shape (n_samples, input_dim)
        Raw high-fidelity target samples evaluated by ``map_obj``.
    target_mean : array-like, shape (input_dim,), optional
        Gaussian reference mean; defaults to zero.
    target_cov : array-like, shape (input_dim, input_dim), optional
        Gaussian reference covariance; defaults to identity.
    ddof : int, default=1
        Delta degrees of freedom for sample covariance.
    jitter : float, default=1e-10
        Covariance regularization for the Forstner distance.
    random_seed : int, optional, default=0
        Seed for MMD reference sampling.
    bandwidth_subset_size : int, default=2000
        Median-heuristic subset limit.
    full_matrix_threshold : int, default=4000
        Dense-kernel cutoff.
    block_size : int, default=1024
        Block size for exact large-cloud MMD.

    Returns
    -------
    dict
        Gaussian diagnostics of mapped reference-space samples.
    """
    pushed = np.asarray(map_obj.evaluate(target_samples), dtype=float)
    return gaussian_diagnostics(
        pushed,
        target_mean=target_mean,
        target_cov=target_cov,
        ddof=ddof,
        jitter=jitter,
        random_seed=random_seed,
        bandwidth_subset_size=bandwidth_subset_size,
        full_matrix_threshold=full_matrix_threshold,
        block_size=block_size,
    )


__all__ = [
    "first_moment_error",
    "forstner_distance",
    "gaussian_diagnostics",
    "map_gaussian_diagnostics",
    "median_heuristic_bandwidth",
    "multiscale_squared_mmd_gaussian_kernel",
    "sample_covariance",
    "sample_mean",
    "second_moment_error",
    "squared_mmd_gaussian_kernel",
]
