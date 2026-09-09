"""Checks single-fidelity basis values, derivatives, and normalization caching."""

import numpy as np

from mftt.single_fidelity import basis as basis_module


def test_evaluate_multiindex_matrix_matches_closed_form_1d_terms():
    sigma = 5.0
    samples = np.array([[-1.5], [0.0], [2.0]])
    multi_indices = [(0,), (1,), (2,), (3,)]
    normalization = np.ones(len(multi_indices), dtype=float)

    matrix = basis_module.evaluate_multiindex_matrix(samples, multi_indices, normalization, sigma)
    x = samples[:, 0]
    gaussian = np.exp(-(x ** 2) / sigma)
    expected = np.column_stack(
        [
            np.ones_like(x),
            x,
            (x ** 2 - 1.0) * gaussian,
            (x ** 3 - 3.0 * x) * gaussian,
        ]
    )
    assert np.allclose(matrix, expected)


def test_evaluate_multiindex_derivative_matrix_matches_finite_difference():
    sigma = 7.0
    samples = np.array([[-1.25, 0.4], [0.5, -0.2], [1.75, 0.8]])
    multi_indices = [(0, 0), (1, 0), (0, 2), (1, 1)]
    normalization = np.ones(len(multi_indices), dtype=float)
    eps = 1e-6

    analytic = basis_module.evaluate_multiindex_derivative_matrix(
        samples,
        multi_indices,
        normalization,
        sigma,
        axis=1,
    )
    numeric = (
        basis_module.evaluate_multiindex_matrix(samples + np.array([[0.0, eps]]), multi_indices, normalization, sigma)
        - basis_module.evaluate_multiindex_matrix(samples - np.array([[0.0, eps]]), multi_indices, normalization, sigma)
    ) / (2.0 * eps)
    assert np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5)


def test_compute_normalization_cache_is_stable():
    multi_indices = [(0, 0), (1, 0), (0, 2)]
    sigma = 30.0
    basis_module._NORMALIZATION_CACHE.clear()

    first = basis_module.compute_normalization(multi_indices, sigma=sigma)
    cache_size_after_first = len(basis_module._NORMALIZATION_CACHE)
    second = basis_module.compute_normalization(multi_indices, sigma=sigma)
    cache_size_after_second = len(basis_module._NORMALIZATION_CACHE)

    assert np.allclose(first, second)
    assert cache_size_after_first == 1
    assert cache_size_after_second == cache_size_after_first
