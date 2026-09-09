"""Checks monotone inversion, Newton fallbacks, and per-row failure safeguards."""

import numpy as np

from mftt.single_fidelity import inversion
from mftt.single_fidelity.inversion import InverseOptions, solve_monotone_inverse


def test_monotone_inverse_uses_newton_without_fallback_when_well_conditioned():
    target = np.array([3.0, -1.0, 7.0])
    result = solve_monotone_inverse(
        evaluate=lambda x: 2.0 * x + 1.0,
        derivative=lambda x: np.full_like(x, 2.0),
        target=target,
        initial=target.copy(),
    )

    assert np.allclose(result.values, np.array([1.0, -1.0, 3.0]))
    assert result.stats.midpoint_steps == 0
    assert result.stats.bracket_failure_count == 0


def test_monotone_inverse_uses_midpoint_when_newton_derivative_is_too_small():
    target = np.array([2.0, -1.5])
    result = solve_monotone_inverse(
        evaluate=lambda x: 0.01 * x,
        derivative=lambda x: np.full_like(x, 1e-12),
        target=target,
        initial=target.copy(),
    )

    assert np.allclose(result.values, np.array([200.0, -150.0]), atol=1e-8)
    assert result.stats.midpoint_steps > 0
    assert result.stats.bracket_failure_count == 0


def test_monotone_inverse_midpoint_steps_preserve_subset_row_context():
    target = np.array([1.0, 2.0, 3.0])

    def derivative(x):
        return np.array([0.5, 1e-12, 0.5])

    result = solve_monotone_inverse(
        evaluate=lambda x: 0.5 * x,
        derivative=derivative,
        target=target,
        initial=target.copy(),
    )

    assert np.allclose(result.values, 2.0 * target, atol=1e-8)
    assert result.stats.midpoint_steps > 0
    assert result.stats.bracket_failure_count == 0


def test_monotone_inverse_recovers_nonfinite_initial_rows_independently():
    target = np.array([1.0, 2.0])
    result = solve_monotone_inverse(
        evaluate=lambda x: x,
        derivative=lambda x: np.ones_like(x),
        target=target,
        initial=np.array([0.0, np.inf]),
    )

    assert np.allclose(result.values, target, atol=1e-8)
    assert result.stats.midpoint_steps == 0
    assert result.stats.bracket_failure_count == 0


def test_monotone_inverse_mixed_brackets_preserve_subset_row_context():
    slopes = np.asarray([1.0, 0.0])
    evaluated_scalars = 0

    def evaluate(values):
        nonlocal evaluated_scalars
        evaluated_scalars += values.size
        return slopes * values

    def evaluate_subset(values, rows):
        nonlocal evaluated_scalars
        evaluated_scalars += values.size
        return slopes[rows] * values

    def derivative(values):
        nonlocal evaluated_scalars
        evaluated_scalars += values.size
        return np.zeros_like(values)

    result = solve_monotone_inverse(
        evaluate=evaluate,
        evaluate_subset=evaluate_subset,
        derivative=derivative,
        target=np.ones(2),
        initial=np.zeros(2),
        max_iter=8,
        tol=2.0e-2,
        inverse_options=InverseOptions(max_bracket_expansions=1),
    )

    assert np.isclose(result.values[0], 1.0, atol=2.0e-2)
    assert np.isnan(result.values[1])
    assert result.failed_mask.tolist() == [False, True]
    assert result.failure_reasons.tolist() == ["", "bracket_not_found"]
    assert result.stats.bracket_failure_count == 1
    assert result.stats.evaluation_count == evaluated_scalars


def test_monotone_inverse_robust_bracket_cap_fails_with_nan():
    target = np.array([1.0])
    result = solve_monotone_inverse(
        evaluate=lambda x: np.zeros_like(x),
        derivative=lambda x: np.zeros_like(x),
        target=target,
        initial=np.array([0.0]),
        max_iter=1,
        inverse_options=InverseOptions(max_bracket_expansions=1),
    )

    assert np.isnan(result.values[0])
    assert result.failed_mask is not None
    assert result.failed_mask.tolist() == [True]
    assert result.failure_reasons[0] == "bracket_not_found"
    assert result.stats.bracket_failure_count == 1


def test_monotone_inverse_robust_coordinate_radius_cap_fails_fast():
    target = np.array([10.0])
    result = solve_monotone_inverse(
        evaluate=lambda x: np.zeros_like(x),
        derivative=lambda x: np.zeros_like(x),
        target=target,
        initial=np.array([0.0]),
        max_iter=1,
        inverse_options=InverseOptions(max_standardized_abs=2.0),
    )

    assert np.isnan(result.values[0])
    assert result.failed_mask[0]
    assert result.failure_reasons[0] == "coordinate_radius_exceeded"


def test_monotone_inverse_robust_evaluation_budget_cap_fails_fast():
    target = np.array([1.0, 2.0])
    result = solve_monotone_inverse(
        evaluate=lambda x: x,
        derivative=lambda x: np.ones_like(x),
        target=target,
        initial=np.array([0.0, 0.0]),
        inverse_options=InverseOptions(max_evaluations=1),
    )

    assert np.all(np.isnan(result.values))
    assert result.failed_mask.tolist() == [True, True]
    assert set(result.failure_reasons.tolist()) == {"evaluation_budget_exceeded"}


def test_monotone_inverse_evaluation_budget_scales_per_initial_row():
    def solve(target):
        return solve_monotone_inverse(
            evaluate=lambda x: x,
            derivative=lambda x: np.ones_like(x),
            target=np.asarray(target, dtype=float),
            initial=np.zeros(len(target), dtype=float),
            inverse_options=InverseOptions(max_evaluations=1),
        )

    batched = solve([1.0, 2.0, 3.0, 4.0])
    individual = [solve([value]) for value in [1.0, 2.0, 3.0, 4.0]]

    assert batched.failed_mask.tolist() == [result.failed_mask[0] for result in individual]
    assert batched.failure_reasons.tolist() == [result.failure_reasons[0] for result in individual]
    assert batched.stats.evaluation_count == sum(result.stats.evaluation_count for result in individual)


def test_monotone_inverse_fallback_uses_per_row_evaluation_budget():
    def solve(target):
        return solve_monotone_inverse(
            evaluate=lambda x: np.zeros_like(x),
            derivative=lambda x: np.zeros_like(x),
            target=np.asarray(target, dtype=float),
            initial=np.zeros(len(target), dtype=float),
            max_iter=1,
            inverse_options=InverseOptions(max_evaluations=10, max_bracket_expansions=10),
        )

    batched = solve([1.0, 2.0, 3.0])
    individual = [solve([value]) for value in [1.0, 2.0, 3.0]]

    assert batched.failed_mask.tolist() == [result.failed_mask[0] for result in individual]
    assert batched.failure_reasons.tolist() == [result.failure_reasons[0] for result in individual]
    assert set(batched.failure_reasons.tolist()) == {"evaluation_budget_exceeded"}


def test_monotone_inverse_wall_time_limit_remains_explicit(monkeypatch):
    clock = iter([0.0, 1.0])
    monkeypatch.setattr(inversion.time, "monotonic", lambda: next(clock))
    result = solve_monotone_inverse(
        evaluate=lambda x: x,
        derivative=lambda x: np.ones_like(x),
        target=np.asarray([1.0]),
        initial=np.asarray([0.0]),
        inverse_options=InverseOptions(max_batch_seconds=0.5),
    )

    assert result.failed_mask.tolist() == [True]
    assert result.failure_reasons.tolist() == ["time_budget_exceeded"]
    assert InverseOptions().max_batch_seconds is None


def test_monotone_inverse_active_rows_stop_evaluating_converged_rows():
    slopes = np.asarray([1.0, 0.01])
    result = solve_monotone_inverse(
        evaluate=lambda values: slopes * values,
        derivative=lambda values: slopes.copy(),
        evaluate_subset=lambda values, rows: slopes[rows] * values,
        derivative_subset=lambda values, rows: slopes[rows],
        target=np.ones(2),
        initial=np.ones(2),
        inverse_options=InverseOptions(max_evaluations=500),
    )

    assert np.allclose(result.values, [1.0, 100.0])
    assert result.evaluation_counts[0] == 1
    assert 1 < result.evaluation_counts[1] <= 500


def test_monotone_inverse_active_budget_is_enforced_per_row():
    result = solve_monotone_inverse(
        evaluate=lambda values: np.zeros_like(values),
        derivative=lambda values: np.zeros_like(values),
        evaluate_subset=lambda values, rows: np.zeros_like(values),
        derivative_subset=lambda values, rows: np.zeros_like(values),
        target=np.ones(3),
        initial=np.zeros(3),
        inverse_options=InverseOptions(max_evaluations=20, max_bracket_expansions=20),
    )

    assert result.failed_mask.tolist() == [True, True, True]
    assert np.all(result.evaluation_counts <= 20)
    assert set(result.failure_reasons.tolist()) == {"evaluation_budget_exceeded"}
