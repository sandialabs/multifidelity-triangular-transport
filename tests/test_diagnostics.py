"""Checks Gaussian diagnostics, MMD, moment errors, and their validation paths."""

import numpy as np
import pytest

from mftt import (
    MapParams,
    TriangularMap,
    first_moment_error,
    forstner_distance,
    gaussian_diagnostics,
    map_gaussian_diagnostics,
    sample_covariance,
    sample_mean,
    second_moment_error,
    squared_mmd_gaussian_kernel,
)


def test_gaussian_diagnostics_standard_normal_are_small():
    rng = np.random.default_rng(10)
    samples = rng.normal(size=(2000, 2))
    summary = gaussian_diagnostics(samples, random_seed=0)

    assert summary["num_samples"] == 2000
    assert summary["input_dim"] == 2
    assert summary["sample_mean"].shape == (2,)
    assert summary["sample_covariance"].shape == (2, 2)
    assert summary["first_moment_error"] < 0.1
    assert summary["second_moment_error"] < 0.2
    assert summary["forstner_distance"] < 0.2
    assert summary["gaussian_kernel_bandwidth"] > 0.0


def test_moment_and_forstner_errors_increase_under_shift_and_scale():
    rng = np.random.default_rng(11)
    baseline = rng.normal(size=(1500, 2))
    shifted_scaled = rng.normal(size=(1500, 2)) @ np.array([[1.8, 0.3], [0.0, 0.6]]).T + np.array([1.0, -0.7])

    assert first_moment_error(shifted_scaled) > first_moment_error(baseline)
    assert second_moment_error(shifted_scaled) > second_moment_error(baseline)
    assert forstner_distance(shifted_scaled) > forstner_distance(baseline)


def test_squared_mmd_is_small_for_matching_gaussian_and_larger_for_mismatch():
    rng = np.random.default_rng(12)
    matching = rng.normal(size=(180, 2))
    mismatched = rng.uniform(low=-2.5, high=2.5, size=(180, 2))

    matching_mmd = squared_mmd_gaussian_kernel(matching, random_seed=0)
    mismatched_mmd = squared_mmd_gaussian_kernel(mismatched, random_seed=0)

    assert mismatched_mmd > matching_mmd


def test_median_heuristic_bandwidth_is_deterministic():
    rng = np.random.default_rng(13)
    samples = rng.normal(size=(120, 3))
    first = gaussian_diagnostics(samples, random_seed=5)["gaussian_kernel_bandwidth"]
    second = gaussian_diagnostics(samples, random_seed=5)["gaussian_kernel_bandwidth"]

    assert first > 0.0
    assert np.isclose(first, second)


def test_dense_and_blockwise_mmd_paths_agree():
    rng = np.random.default_rng(17)
    samples = rng.normal(size=(96, 2))
    reference = rng.normal(size=(104, 2))

    dense = squared_mmd_gaussian_kernel(
        samples,
        reference_samples=reference,
        full_matrix_threshold=10_000,
        random_seed=4,
    )
    blockwise = squared_mmd_gaussian_kernel(
        samples,
        reference_samples=reference,
        full_matrix_threshold=16,
        block_size=32,
        random_seed=4,
    )

    assert np.isclose(dense, blockwise, atol=1e-12)


def test_map_gaussian_diagnostics_matches_sample_based_diagnostics():
    rng = np.random.default_rng(14)
    train_data = rng.normal(size=(80, 2))
    target_samples = rng.normal(size=(40, 2))
    model = TriangularMap(train_data=train_data,
        params=MapParams(total_order=2),
    )

    direct = gaussian_diagnostics(model.evaluate(target_samples), random_seed=3)
    wrapped = map_gaussian_diagnostics(model, target_samples, random_seed=3)

    assert np.allclose(direct["sample_mean"], wrapped["sample_mean"])
    assert np.allclose(direct["sample_covariance"], wrapped["sample_covariance"])
    assert np.isclose(direct["first_moment_error"], wrapped["first_moment_error"])
    assert np.isclose(direct["second_moment_error"], wrapped["second_moment_error"])
    assert np.isclose(direct["forstner_distance"], wrapped["forstner_distance"])
    assert np.isclose(direct["squared_mmd_gaussian_kernel"], wrapped["squared_mmd_gaussian_kernel"])


def test_general_gaussian_target_gives_small_errors_for_matching_samples():
    rng = np.random.default_rng(15)
    target_mean = np.array([1.2, -0.4])
    target_cov = np.array([[1.7, 0.25], [0.25, 0.8]])
    samples = rng.multivariate_normal(target_mean, target_cov, size=2500)

    summary = gaussian_diagnostics(samples, target_mean=target_mean, target_cov=target_cov, random_seed=2)

    assert summary["first_moment_error"] < 0.1
    assert summary["second_moment_error"] < 0.2
    assert summary["forstner_distance"] < 0.2


def test_basic_helpers_return_expected_shapes():
    rng = np.random.default_rng(16)
    samples = rng.normal(size=(50, 3))

    assert sample_mean(samples).shape == (3,)
    assert sample_covariance(samples).shape == (3, 3)


def test_invalid_shapes_raise_clear_errors():
    with pytest.raises(ValueError, match="2D"):
        gaussian_diagnostics(np.array([1.0, 2.0]))

    with pytest.raises(ValueError, match="target_mean"):
        gaussian_diagnostics(np.ones((10, 2)), target_mean=np.ones(3))

    with pytest.raises(ValueError, match="target_cov"):
        gaussian_diagnostics(np.ones((10, 2)), target_cov=np.eye(3))


def test_forstner_distance_handles_near_singular_covariance():
    samples = np.ones((6, 2))
    distance = forstner_distance(samples, jitter=1e-8)
    assert np.isfinite(distance)


def test_mmd_requires_two_samples_per_cloud_for_unbiased_estimate():
    with pytest.raises(ValueError, match="at least two"):
        squared_mmd_gaussian_kernel(np.zeros((1, 2)), random_seed=0)


def test_large_sample_mmd_blockwise_path_returns_finite_value():
    rng = np.random.default_rng(18)
    samples = rng.normal(size=(4500, 2))
    value = squared_mmd_gaussian_kernel(
        samples,
        full_matrix_threshold=512,
        block_size=256,
        bandwidth_subset_size=800,
        random_seed=6,
    )
    assert np.isfinite(value)
