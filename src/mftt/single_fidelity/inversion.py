"""Shared vectorized inversion helpers for monotone triangular components."""

from __future__ import annotations

from dataclasses import dataclass
import time
from typing import Callable

import numpy as np


@dataclass(frozen=True)
class InverseOptions:
    """Optional resource limits for monotone inverse workloads.

    Parameters
    ----------
    max_bracket_expansions : int, default=60
        Maximum geometric expansions used to bracket one scalar inverse root.
    max_standardized_abs : float, optional
        Maximum absolute standardized coordinate considered during inversion.
    max_evaluations : int, optional
        Per-row limit on component-function evaluations.
    max_batch_seconds : float, optional
        Wall-clock limit for a row batch.

    Notes
    -----
    Limits turn difficult root solves into structured diagnostics rather than
    silently returning inaccurate values.
    """

    max_bracket_expansions: int = 60
    max_standardized_abs: float | None = None
    max_evaluations: int | None = None
    max_batch_seconds: float | None = None


@dataclass(frozen=True)
class InverseFailure:
    """Describe one failed inverse component solve.

    Parameters
    ----------
    row : int
        Zero-based row index in the requested inverse batch.
    component : int
        Zero-based triangular component index that failed.
    reason : str
        Machine-readable failure category.
    residual : float
        Final absolute scalar residual.
    bracket_radius : float
        Final standardized bracketing radius.
    evaluations : int
        Number of component evaluations consumed for the row.
    stage : int, optional
        Hierarchical fidelity-stage index, when applicable.
    """

    row: int
    component: int
    reason: str
    residual: float
    bracket_radius: float
    evaluations: int
    stage: int | None = None


@dataclass(frozen=True)
class InverseDiagnostics:
    """Aggregate inverse diagnostics for a full map inverse call.

    Parameters
    ----------
    failed_mask : numpy.ndarray, shape (n_samples,)
        Boolean mask identifying rows with at least one failed component.
    failures : tuple of InverseFailure
        Detailed failure records.
    component_failure_counts : numpy.ndarray, shape (input_dim,)
        Number of failed rows for each triangular component.
    max_residual : float
        Largest finite residual, or infinity when no finite residual exists.
    evaluation_counts : numpy.ndarray, shape (n_samples,), optional
        Total component evaluations consumed for each row.
    """

    failed_mask: np.ndarray
    failures: tuple[InverseFailure, ...]
    component_failure_counts: np.ndarray
    max_residual: float
    evaluation_counts: np.ndarray | None = None


@dataclass(frozen=True)
class InverseResult:
    """Return inverse values together with diagnostics.

    Parameters
    ----------
    values : numpy.ndarray, shape (n_samples, input_dim)
        Recovered target-space coordinates; failed rows may be incomplete.
    diagnostics : InverseDiagnostics
        Per-row and per-component inversion outcome information.
    """

    values: np.ndarray
    diagnostics: InverseDiagnostics


def combine_inverse_results(batches: list[InverseResult]) -> InverseResult:
    """Concatenate row-batched inverse results and reindex failures once."""
    if not batches:
        raise ValueError("At least one inverse batch is required.")
    values = np.concatenate([batch.values for batch in batches], axis=0)
    failed_mask = np.concatenate([batch.diagnostics.failed_mask for batch in batches], axis=0)
    failures: list[InverseFailure] = []
    offset = 0
    for batch in batches:
        failures.extend(
            InverseFailure(
                row=int(failure.row) + offset,
                component=int(failure.component),
                reason=failure.reason,
                residual=float(failure.residual),
                bracket_radius=float(failure.bracket_radius),
                evaluations=int(failure.evaluations),
                stage=failure.stage,
            )
            for failure in batch.diagnostics.failures
        )
        offset += batch.values.shape[0]
    component_counts = np.sum(
        [batch.diagnostics.component_failure_counts for batch in batches], axis=0, dtype=int
    )
    evaluation_counts = np.concatenate(
        [
            np.zeros(batch.values.shape[0], dtype=int)
            if batch.diagnostics.evaluation_counts is None
            else np.asarray(batch.diagnostics.evaluation_counts, dtype=int)
            for batch in batches
        ]
    )
    residuals = [batch.diagnostics.max_residual for batch in batches]
    finite = [value for value in residuals if np.isfinite(value)]
    return InverseResult(
        values=values,
        diagnostics=InverseDiagnostics(
            failed_mask=failed_mask,
            failures=tuple(failures),
            component_failure_counts=np.asarray(component_counts, dtype=int),
            max_residual=float(max(finite)) if finite else float("inf"),
            evaluation_counts=evaluation_counts,
        ),
    )


def validate_max_batch_rows(max_batch_rows: int | None) -> int | None:
    """Validate and normalize the optional inverse row-batch limit."""
    if max_batch_rows is None:
        return None
    value = int(max_batch_rows)
    if value <= 0 or value != max_batch_rows:
        raise ValueError("max_batch_rows must be a positive integer.")
    return value


def batch_fixed_values(fixed_values, start: int, stop: int):
    """Select row-wise fixed values for one inverse batch."""
    if fixed_values is None:
        return None
    array = np.asarray(fixed_values, dtype=float)
    return array[start:stop] if array.ndim == 2 else fixed_values


def unwrap_inverse_result(result: InverseResult, return_diagnostics: bool):
    """Return diagnostics on request, otherwise require every row to converge."""
    if return_diagnostics:
        return result
    failed = np.flatnonzero(result.diagnostics.failed_mask)
    if failed.size:
        first = result.diagnostics.failures[0].reason if result.diagnostics.failures else "unknown"
        raise RuntimeError(
            f"Inverse failed for {failed.size} row(s) (first reason: {first}). "
            "Rerun with return_diagnostics=True for row-level details."
        )
    return result.values


@dataclass(frozen=True)
class InverseSolveStats:
    """Summarize one vectorized safeguarded-Newton solve."""

    iterations: int
    midpoint_steps: int
    bracket_failure_count: int
    max_residual: float
    evaluation_count: int = 0

@dataclass(frozen=True)
class InverseSolveResult:
    """Store vectorized inverse values and lightweight diagnostics."""

    values: np.ndarray
    stats: InverseSolveStats
    failed_mask: np.ndarray
    failure_reasons: np.ndarray
    residuals: np.ndarray
    bracket_radii: np.ndarray
    evaluation_counts: np.ndarray


def solve_monotone_inverse(
    evaluate: Callable[[np.ndarray], np.ndarray],
    derivative: Callable[[np.ndarray], np.ndarray],
    target: np.ndarray,
    initial: np.ndarray,
    *,
    max_iter: int = 100,
    tol: float = 1e-8,
    derivative_floor: float = 1e-8,
    max_bracket_expansions: int = 60,
    evaluate_subset: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    derivative_subset: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    inverse_options: InverseOptions | None = None,
) -> InverseSolveResult:
    """Solve increasing scalar roots with bracketed Newton iterations.

    The callbacks may optionally accept active row indices through the
    ``*_subset`` variants. This lets triangular map components stop evaluating
    rows that have already converged or failed without changing the scalar
    algorithm.
    """
    target = np.asarray(target, dtype=float).reshape(-1)
    initial = np.asarray(initial, dtype=float).reshape(-1)
    if target.shape != initial.shape:
        raise ValueError("target and initial must have matching 1D shapes.")
    options = inverse_options or InverseOptions(max_bracket_expansions=max_bracket_expansions)
    max_bracket_expansions = int(options.max_bracket_expansions)
    n_rows = target.size
    counts = np.zeros(n_rows, dtype=int)
    reasons = np.full(n_rows, "", dtype=object)
    radii = np.zeros(n_rows, dtype=float)
    failed = ~np.isfinite(target)
    reasons[failed] = "nonfinite_target"
    residual = np.full(n_rows, np.nan, dtype=float)
    values = np.where(np.isfinite(initial), initial, target)
    start_time = time.monotonic()
    midpoint_steps = 0
    bracket_failure_count = 0

    def call_evaluate(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        counts[rows] += 1
        if evaluate_subset is not None and rows.size != n_rows:
            result = evaluate_subset(x, rows)
        else:
            result = evaluate(x)
        result = np.asarray(result, dtype=float).reshape(-1)
        if result.shape != x.shape and result.size == n_rows:
            result = result[rows]
        if result.shape != x.shape:
            raise ValueError("evaluate callback returned an unexpected shape.")
        return result

    def call_derivative(x: np.ndarray, rows: np.ndarray) -> np.ndarray:
        counts[rows] += 1
        if derivative_subset is not None and rows.size != n_rows:
            result = derivative_subset(x, rows)
        else:
            result = derivative(x)
        result = np.asarray(result, dtype=float).reshape(-1)
        if result.shape != x.shape and result.size == n_rows:
            result = result[rows]
        if result.shape != x.shape:
            raise ValueError("derivative callback returned an unexpected shape.")
        return result

    def limit_rows(rows: np.ndarray, candidates: np.ndarray, needed: int = 1) -> np.ndarray:
        limited = np.zeros(rows.size, dtype=bool)
        if options.max_standardized_abs is not None:
            limited |= np.abs(candidates) > float(options.max_standardized_abs)
            reasons[rows[limited]] = "coordinate_radius_exceeded"
        if options.max_evaluations is not None:
            limited |= counts[rows] + int(needed) > int(options.max_evaluations)
            reasons[rows[limited]] = "evaluation_budget_exceeded"
        if options.max_batch_seconds is not None and time.monotonic() - start_time >= float(options.max_batch_seconds):
            limited[:] = True
            reasons[rows] = "time_budget_exceeded"
        failed[rows[limited]] = True
        return limited

    if n_rows == 0:
        return InverseSolveResult(
            values=values,
            stats=InverseSolveStats(0, 0, 0, 0.0, 0),
            failed_mask=failed,
            failure_reasons=reasons,
            residuals=residual,
            bracket_radii=radii,
            evaluation_counts=counts,
        )

    all_rows = np.arange(n_rows, dtype=int)
    active = all_rows[~failed]
    if options.max_evaluations is not None and int(options.max_evaluations) < 1:
        failed[active] = True
        reasons[active] = "evaluation_budget_exceeded"
        active = np.empty(0, dtype=int)
    if options.max_batch_seconds is not None and time.monotonic() - start_time >= float(options.max_batch_seconds):
        failed[active] = True
        reasons[active] = "time_budget_exceeded"
        active = np.empty(0, dtype=int)
    if active.size:
        residual[active] = target[active] - call_evaluate(values[active], active)
    converged = np.isfinite(residual) & (np.abs(residual) < tol)
    failed |= ~np.isfinite(residual) & ~converged
    reasons[(~np.isfinite(residual)) & ~converged & (reasons == "")] = "nonfinite_evaluation"

    center = np.where(np.isfinite(values), values, target)
    lower = center - 1.0
    upper = center + 1.0
    radius = np.ones(n_rows, dtype=float)
    bracketed = converged.copy()
    for expansion in range(max_bracket_expansions + 1):
        rows = np.flatnonzero(~(failed | converged | bracketed))
        if rows.size == 0:
            break
        if expansion:
            radius[rows] *= 2.0
            lower[rows] = center[rows] - radius[rows]
            upper[rows] = center[rows] + radius[rows]
        limits = limit_rows(rows, np.maximum(np.abs(lower[rows]), np.abs(upper[rows])), needed=2)
        rows = rows[~limits]
        if rows.size == 0:
            continue
        lower_values = call_evaluate(lower[rows], rows)
        upper_values = call_evaluate(upper[rows], rows)
        good = (
            np.isfinite(lower_values)
            & np.isfinite(upper_values)
            & (lower_values <= target[rows])
            & (upper_values >= target[rows])
        )
        bracketed[rows[good]] = True
    missing = ~(failed | converged | bracketed)
    if np.any(missing):
        failed[missing] = True
        reasons[missing] = np.where(reasons[missing] == "", "bracket_not_found", reasons[missing])
        bracket_failure_count = int(np.count_nonzero(missing))

    iterations = 0
    for iterations in range(1, int(max_iter) + 1):
        rows = np.flatnonzero(bracketed & ~failed & ~converged)
        if rows.size == 0:
            break
        derivatives = call_derivative(values[rows], rows)
        newton = np.full(rows.size, np.nan, dtype=float)
        np.divide(
            residual[rows],
            derivatives,
            out=newton,
            where=np.isfinite(derivatives) & (np.abs(derivatives) > 0.0),
        )
        newton += values[rows]
        midpoint = 0.5 * (lower[rows] + upper[rows])
        use_newton = (
            np.isfinite(newton)
            & np.isfinite(derivatives)
            & (derivatives > derivative_floor)
            & (newton >= lower[rows])
            & (newton <= upper[rows])
        )
        candidates = np.where(use_newton, newton, midpoint)
        midpoint_steps += int(np.count_nonzero(~use_newton))
        limits = limit_rows(rows, candidates, needed=1)
        rows = rows[~limits]
        candidates = candidates[~limits]
        if rows.size == 0:
            continue
        candidate_values = call_evaluate(candidates, rows)
        bad = ~np.isfinite(candidate_values)
        if np.any(bad):
            bad_rows = rows[bad]
            retry = midpoint[~limits][bad]
            retry_limits = limit_rows(bad_rows, retry, needed=1)
            retry_rows = bad_rows[~retry_limits]
            if retry_rows.size:
                retry_values = call_evaluate(retry[~retry_limits], retry_rows)
                candidate_values[bad] = np.nan
                candidate_values[np.flatnonzero(bad)[~retry_limits]] = retry_values
                candidates[bad] = retry[~retry_limits]
                retry_bad = ~np.isfinite(retry_values)
                if np.any(retry_bad):
                    failed[retry_rows[retry_bad]] = True
                    reasons[retry_rows[retry_bad]] = "nonfinite_evaluation"
            failed_bad = bad_rows[retry_limits]
            failed[failed_bad] = True
            reasons[failed_bad] = "nonfinite_evaluation"
        finite = np.isfinite(candidate_values)
        valid_rows = rows[finite]
        valid_values = candidates[finite]
        valid_outputs = candidate_values[finite]
        values[valid_rows] = valid_values
        residual[valid_rows] = target[valid_rows] - valid_outputs
        converged[valid_rows] = np.abs(residual[valid_rows]) < tol
        above = valid_outputs > target[valid_rows]
        lower[valid_rows] = np.where(above, lower[valid_rows], valid_values)
        upper[valid_rows] = np.where(above, valid_values, upper[valid_rows])

    unresolved = bracketed & ~failed & ~converged
    failed[unresolved] = True
    reasons[unresolved] = "residual_not_converged"
    values[failed] = np.nan
    finite_residual = np.abs(residual[np.isfinite(residual)])
    max_residual = float(np.max(finite_residual)) if finite_residual.size else float("inf")
    return InverseSolveResult(
        values=values,
        stats=InverseSolveStats(
            iterations=iterations,
            midpoint_steps=midpoint_steps,
            bracket_failure_count=bracket_failure_count,
            max_residual=max_residual,
            evaluation_count=int(np.sum(counts)),
        ),
        failed_mask=failed,
        failure_reasons=reasons,
        residuals=residual,
        bracket_radii=radius,
        evaluation_counts=counts,
    )
