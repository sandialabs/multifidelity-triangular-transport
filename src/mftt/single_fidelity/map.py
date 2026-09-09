"""Implement refactored single-fidelity triangular transport maps."""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.optimize import minimize

from .._describe import format_triangular_map_lines

from .coefficients import MapCoefficients
from .components import ComponentCache, MapComponent
from .inversion import (
    InverseDiagnostics,
    InverseFailure,
    InverseOptions,
    InverseResult,
    batch_fixed_values,
    combine_inverse_results,
    solve_monotone_inverse,
    unwrap_inverse_result,
    validate_max_batch_rows,
)
from .params import (
    MapParams,
    OptimizationParams,
    StandardizationParams,
    component_params_from_multi_index_sets,
)
from .reference import Reference
from .utils import (
    apply_standardization,
    ensure_2d,
    invert_standardization,
    validate_fixed_indices,
    validate_leading_fixed_indices,
    summarize_minimize_result,
)


class TriangularMap:
    """Represent a triangular transport from a target density to a reference.

    Parameters
    ----------
    train_data : array-like, shape (n_samples, input_dim)
        Raw target samples from the density to be transported.
    params : MapParams
        Basis and optimization configuration.
    reference : Reference, optional
        Reference density ``eta``. The standard normal is used when omitted.
    standardization : StandardizationParams, optional
        Precomputed raw-coordinate standardization. By default it is fitted
        from ``train_data``.

    Notes
    -----
    Public methods accept raw coordinates. ``evaluate`` maps the target to
    ``eta`` and ``inverse`` pulls reference samples back to the target.
    """

    @staticmethod
    def _copy_standardization_params(
        params: StandardizationParams,
        *,
        expected_dim: int,
    ) -> StandardizationParams:
        """Validate and copy externally supplied standardization parameters."""
        copied = StandardizationParams(mean=params.mean.copy(), std=params.std.copy())
        if copied.mean.shape[0] != expected_dim:
            raise ValueError("StandardizationParams must match the map input dimension.")
        return copied

    @classmethod
    def from_multi_index_sets(
        cls,
        train_data,
        multi_index_sets,
        reference: Reference | None = None,
        optimization: OptimizationParams | None = None,
        sigma: float = 30.0,
        rectifier_epsilon: float = 1e-4,
        quadrature_rule=None,
    ) -> "TriangularMap":
        """Construct a map from explicit per-component multi-index sets.

        Parameters
        ----------
        train_data : array-like, shape (n_samples, input_dim)
            Raw target training samples.
        multi_index_sets : sequence of sequence of tuple of int
            One Hermite multi-index collection for each triangular component.
        reference : Reference, optional
            Reference density; defaults to the standard normal.
        optimization : OptimizationParams, optional
            Component-optimization configuration.
        sigma : float, default=30.0
            Hermite normalization scale.
        rectifier_epsilon : float, default=1e-4
            Positive monotonicity-rectifier floor.
        quadrature_rule : QuadratureRule, optional
            Quadrature rule for monotone component integrals.

        Returns
        -------
        TriangularMap
            Untrained map configured with the supplied bases.
        """
        input_dim = len(multi_index_sets)
        component_params = component_params_from_multi_index_sets(
            multi_index_sets=multi_index_sets,
            sigma=sigma,
            rectifier_epsilon=rectifier_epsilon,
            quadrature_rule=quadrature_rule,
        )
        map_params = MapParams(
            component_params=component_params,
            optimization=optimization or OptimizationParams(),
        )
        return cls(train_data, map_params, reference=reference)

    def __init__(
        self,
        train_data,
        params: MapParams,
        *,
        reference: Reference | None = None,
        standardization: StandardizationParams | None = None,
    ) -> None:
        """Build a map from one dataset and one parameter object.

        Parameters
        ----------
        train_data : array-like, shape (n_samples, input_dim)
            Raw target training samples.
        params : MapParams
            Map basis and optimization settings.
        reference : Reference, optional
            Reference density; defaults to a standard normal.
        standardization : StandardizationParams, optional
            Raw-coordinate standardization to retain as fitted map state.
        """
        data = ensure_2d(train_data)
        self.input_dim = data.shape[1]
        params.bind(self.input_dim)
        self.params = params
        self.total_order = self.params.total_order
        self.reference = reference or Reference.standard_normal(self.input_dim)
        self.train_data = data

        if standardization is None:
            mean = np.mean(data, axis=0)
            std = np.std(data, axis=0)
            if np.any(std <= 0.0):
                raise ValueError("Training data contains a zero-variance feature.")
            self.standardization_params = StandardizationParams(mean=mean, std=std)
        else:
            self.standardization_params = self._copy_standardization_params(
                standardization, expected_dim=self.input_dim
            )
        self.standardized_train_data = apply_standardization(self.train_data, self.standardization_params)

        self.components = [MapComponent(component_params) for component_params in self.params.component_params]
        self._component_cache: list[ComponentCache] | None = None
        self._last_aao_optimization_result: dict[str, object] | None = None

    def __str__(self) -> str:
        """Return a human-readable summary of the configured map."""
        return "\n".join(format_triangular_map_lines(self, title="TriangularMap"))

    __repr__ = __str__

    @property
    def coeffs(self) -> np.ndarray:
        """Return all component coefficients as one flat vector."""
        return self.coefficients.flatten()

    @property
    def coefficients(self) -> MapCoefficients:
        """Expose the full map coefficients in structured form."""
        return MapCoefficients(components=[component.coefficients for component in self.components])

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        """Assign a flattened coefficient vector to all components.

        Parameters
        ----------
        coeffs : array-like, shape (n_coefficients,)
            Concatenated component coefficients in map order.
        """
        self.coefficients.set_from_flat(coeffs)

    def precompute_training_data(self) -> list[ComponentCache]:
        """Build or reuse per-component caches for the training dataset."""
        if self._component_cache is None:
            standardized = self.standardized_train_data
            caches = []
            for k, component in enumerate(self.components, start=1):
                caches.append(component.precompute(standardized[:, :k]))
            self._component_cache = caches
        return self._component_cache

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Map raw target samples to reference samples.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Target-space samples.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Reference-space coordinates.
        """
        params = self.standardization_params
        x_scaled = apply_standardization(ensure_2d(x, expected_dim=self.input_dim), params)
        return self.evaluate_standardized(x_scaled)

    def evaluate_standardized(self, x_std: np.ndarray) -> np.ndarray:
        """Evaluate the map on already-standardized target inputs.

        Parameters
        ----------
        x_std : array-like, shape (n_samples, input_dim)
            Target samples standardized with this map's fitted parameters.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Reference-space coordinates.
        """
        standardized = ensure_2d(x_std, expected_dim=self.input_dim)
        output = np.zeros_like(standardized)
        for k, component in enumerate(self.components, start=1):
            output[:, k - 1] = component.evaluate(standardized[:, :k])
        return output

    def objective_aao_precomp(self, coeffs: np.ndarray, reg_cst: float | None = None) -> float:
        """Evaluate the all-at-once KL objective with cached bases.

        Parameters
        ----------
        coeffs : array-like, shape (n_coefficients,)
            Candidate flattened map coefficients.
        reg_cst : float, optional
            Quadratic regularization constant; defaults to configured value.

        Returns
        -------
        float
            Regularized training objective.
        """
        reg = self.params.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        caches = self.precompute_training_data()
        z = np.zeros_like(self.standardized_train_data)
        for k, (component, cache) in enumerate(zip(self.components, caches), start=1):
            z[:, k - 1] = component.evaluate(cache.x, cache=cache)
        log_det = self.log_det_aao_precomp()
        total = np.mean(-self.reference.evaluate_logpdf(z) - log_det)
        return float(total + reg * np.dot(coeffs, coeffs))

    def gradient_aao_precomp(self, coeffs: np.ndarray, reg_cst: float | None = None) -> np.ndarray:
        """Evaluate the analytic AAO gradient using cached training data.

        Parameters
        ----------
        coeffs : array-like, shape (n_coefficients,)
            Candidate flattened map coefficients.
        reg_cst : float, optional
            Quadratic regularization constant; defaults to configured value.

        Returns
        -------
        numpy.ndarray, shape (n_coefficients,)
            Gradient of the regularized all-at-once objective.
        """
        reg = self.params.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        caches = self.precompute_training_data()
        z = np.zeros_like(self.standardized_train_data)
        for k, (component, cache) in enumerate(zip(self.components, caches), start=1):
            z[:, k - 1] = component.evaluate(cache.x, cache=cache)
        score = self.reference.evaluate_score(z)
        grads = []
        for k, (component, cache) in enumerate(zip(self.components, caches), start=1):
            component_values = component.evaluate(cache.x, cache=cache)
            deriv = component.evaluate_derivative_xk(cache.x, cache=cache)
            coeff_grad = component.evaluate_coeff_gradient(cache.x, cache=cache)
            deriv_coeff_grad = component.evaluate_derivative_xk_coeff_gradient(cache.x, cache=cache)
            total = np.mean(
                -score[:, k - 1][:, None] * coeff_grad - (1.0 / deriv)[:, None] * deriv_coeff_grad,
                axis=0,
            )
            grads.append(total)
        flat = np.concatenate(grads)
        return flat + 2.0 * reg * coeffs

    def train(self, optimization: OptimizationParams | None = None, verbose: bool = False) -> None:
        """Train components sequentially on one dataset.

        Parameters
        ----------
        optimization : OptimizationParams, optional
            Optimizer settings overriding those stored in ``params``.
        verbose : bool, default=False
            Print one completion message per triangular component.
        """
        optimization = optimization or self.params.optimization
        standardized = self.standardized_train_data
        # Basis evaluations are cheap relative to optimization; keep one
        # cached training path so sequential and AAO training share the same
        # algebra.
        caches = self.precompute_training_data()
        self._last_aao_optimization_result = None
        for k, component in enumerate(self.components, start=1):
            cache = caches[k - 1]
            local_x = standardized[:, :k]
            component.train(local_x, optimization=optimization, cache=cache)
            if verbose:
                print(f"Completed training component {k}/{self.input_dim}.")
        self._component_cache = None

    def train_aao(self, optimization: OptimizationParams | None = None, verbose: bool = False) -> None:
        """Train the whole map jointly with analytic AAO gradients.

        Parameters
        ----------
        optimization : OptimizationParams, optional
            Optimizer settings overriding the configured values.
        verbose : bool, default=False
            Print a completion message after optimization.
        """
        optimization = optimization or self.params.optimization
        objective = self.objective_aao_precomp
        gradient = self.gradient_aao_precomp
        result = minimize(
            objective,
            self.coeffs,
            args=(optimization.reg_cst,),
            jac=gradient,
            method=optimization.optimizer,
            options={"gtol": optimization.gtol, "maxiter": optimization.maxiter},
        )
        self.set_coeffs(result.x)
        self._last_aao_optimization_result = summarize_minimize_result(result, optimization)
        self._component_cache = None
        if verbose:
            print("Completed all-at-once map training.")

    def optimization_diagnostics(self) -> dict[str, object]:
        """Return the most recent optimizer summaries for the trained map."""
        return {
            "components": [component.last_optimization_result for component in self.components],
            "aao": self._last_aao_optimization_result,
        }

    def log_det(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the log absolute determinant on raw target inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            ``log |det dS/dx|`` including raw-coordinate standardization.
        """
        params = self.standardization_params
        x_scaled = apply_standardization(ensure_2d(x, expected_dim=self.input_dim), params)
        return self.log_det_standardized(x_scaled) - np.sum(np.log(params.std))

    def log_det_standardized(self, x_std: np.ndarray) -> np.ndarray:
        """Evaluate the log determinant on standardized target inputs.

        Parameters
        ----------
        x_std : array-like, shape (n_samples, input_dim)
            Target coordinates standardized with this map's fitted parameters.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Log determinant in standardized target coordinates.
        """
        standardized = ensure_2d(x_std, expected_dim=self.input_dim)
        log_det = np.zeros(standardized.shape[0], dtype=float)
        for k, component in enumerate(self.components, start=1):
            log_det += np.log(component.evaluate_derivative_xk(standardized[:, :k]))
        return log_det

    def component_log_det(self, x: np.ndarray, component_indices) -> np.ndarray:
        """Evaluate selected diagonal log-determinant terms on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates.
        component_indices : iterable of int
            Zero-based triangular component indices whose diagonal terms are
            summed. Indices must lie in ``[0, input_dim)``.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Sum of requested raw-coordinate diagonal log-Jacobian terms.
        """
        indices = validate_fixed_indices(component_indices, self.input_dim)
        samples = ensure_2d(x, expected_dim=self.input_dim)
        if not indices:
            return np.zeros(samples.shape[0], dtype=float)
        params = self.standardization_params
        x_scaled = apply_standardization(samples, params)
        return self.component_log_det_standardized(x_scaled, indices) - np.sum(np.log(params.std[indices]))

    def component_log_det_standardized(self, x_std: np.ndarray, component_indices) -> np.ndarray:
        """Evaluate selected diagonal log-determinant terms on standardized inputs.

        Parameters
        ----------
        x_std : array-like, shape (n_samples, input_dim)
            Standardized target coordinates.
        component_indices : iterable of int
            Zero-based component indices whose diagonal terms are summed.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Sum of requested standardized-coordinate log-Jacobian terms.
        """
        indices = validate_fixed_indices(component_indices, self.input_dim)
        standardized = ensure_2d(x_std, expected_dim=self.input_dim)
        log_det = np.zeros(standardized.shape[0], dtype=float)
        for idx in indices:
            component = self.components[idx]
            log_det += np.log(component.evaluate_derivative_xk(standardized[:, : idx + 1]))
        return log_det

    def log_det_aao_precomp(self) -> np.ndarray:
        """Evaluate the training-data log determinant from caches."""
        log_det = np.zeros(self.standardized_train_data.shape[0], dtype=float)
        for component, cache in zip(self.components, self.precompute_training_data()):
            log_det += np.log(component.evaluate_derivative_xk(cache.x, cache=cache))
        return log_det

    def inverse(
        self,
        z: np.ndarray,
        max_iter: int = 100,
        tol: float = 1e-8,
        fixed_indices=None,
        fixed_values=None,
        max_batch_rows: int | None = None,
        inverse_options: InverseOptions | None = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray | InverseResult:
        """Map reference-space samples back to target space.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        max_iter : int, default=100
            Maximum safeguarded-Newton iterations per component and row.
        tol : float, default=1e-8
            Absolute scalar inverse residual tolerance.
        fixed_indices : iterable of int, optional
            Leading target-coordinate prefix ``[0, ..., m - 1]`` to clamp.
        fixed_values : array-like, shape (m,) or (n_samples, m), optional
            Raw target values for ``fixed_indices``. Required when indices are
            supplied and broadcast from one row when one-dimensional.
        max_batch_rows : int, optional
            Maximum rows solved in one inverse batch; bounds workspace memory.
        inverse_options : InverseOptions, optional
            Additional bracket, evaluation, and time limits.
        return_diagnostics : bool, default=False
            Return an ``InverseResult`` instead of only values.

        Returns
        -------
        numpy.ndarray or InverseResult
            Raw target coordinates, or values together with inversion
            diagnostics when requested.

        Raises
        ------
        ValueError
            If coordinate shapes are invalid or conditioned indices are not a
            leading prefix.
        RuntimeError
            If a solve fails and diagnostics were not requested.

        Notes
        -----
        When ``fixed_indices`` are supplied, the inverse solve clamps those
        target-space coordinates and samples the remaining coordinates
        conditionally under the learned triangular map. They must form the
        leading prefix ``[0, ..., m - 1]``.

        ``max_batch_rows`` bounds inversion workspace memory by solving
        independent row batches. Chunked and unchunked solves can differ at
        the requested numerical tolerance because convergence is checked per
        batch.
        """
        z = ensure_2d(z, expected_dim=self.input_dim)
        options = inverse_options or InverseOptions()
        if max_batch_rows is None or z.shape[0] <= validate_max_batch_rows(max_batch_rows):
            return unwrap_inverse_result(self._inverse_rows_result(z, max_iter, tol, fixed_indices, fixed_values, options), return_diagnostics)
        batch_rows = validate_max_batch_rows(max_batch_rows)
        batches: list[InverseResult] = []
        for start in range(0, z.shape[0], batch_rows):
            stop = min(start + batch_rows, z.shape[0])
            fixed_for_batch = (
                batch_fixed_values(fixed_values, start, stop)
            )
            batches.append(
                self._inverse_rows_result(
                    z[start:stop],
                    max_iter,
                    tol,
                    fixed_indices,
                    fixed_for_batch,
                    options,
                )
            )
        result = self._combine_inverse_results(batches)
        return unwrap_inverse_result(result, return_diagnostics)

    def _combine_inverse_results(self, batches: list[InverseResult]) -> InverseResult:
        """Combine row-batched inverse results."""
        return combine_inverse_results(batches)

    def _inverse_rows_result(
        self,
        z: np.ndarray,
        max_iter: int,
        tol: float,
        fixed_indices,
        fixed_values,
        inverse_options: InverseOptions,
    ) -> InverseResult:
        """Invert one row batch; conditioned indices must be a leading prefix."""
        params = self.standardization_params
        fixed_mask = np.zeros(self.input_dim, dtype=bool)
        fixed_std = np.zeros((z.shape[0], self.input_dim), dtype=float)
        options = inverse_options

        if fixed_indices is not None and len(fixed_indices) > 0:
            indices = validate_leading_fixed_indices(fixed_indices, self.input_dim)
            if fixed_values is None:
                raise ValueError("fixed_values must be provided when fixed_indices are specified.")
            fixed_values = np.asarray(fixed_values, dtype=float)
            if fixed_values.ndim == 1:
                fixed_values = np.broadcast_to(fixed_values, (z.shape[0], fixed_values.shape[0]))
            if fixed_values.shape != (z.shape[0], len(indices)):
                raise ValueError("fixed_values must have shape (n_samples, len(fixed_indices)) or be 1D.")
            for col, idx in enumerate(indices):
                fixed_mask[idx] = True
                fixed_std[:, idx] = (fixed_values[:, col] - params.mean[idx]) / params.std[idx]

        x_std = np.zeros_like(z)
        row_failed = np.zeros(z.shape[0], dtype=bool)
        failures: list[InverseFailure] = []
        component_failure_counts = np.zeros(self.input_dim, dtype=int)
        evaluation_counts = np.zeros(z.shape[0], dtype=int)
        max_residual = 0.0
        for k, component in enumerate(self.components):
            if fixed_mask[k]:
                x_std[:, k] = fixed_std[:, k]
                continue
            active_rows = np.flatnonzero(~row_failed)
            if active_rows.size == 0:
                x_std[:, k] = np.nan
                continue
            target = z[active_rows, k]
            x_curr = target.copy()
            x_prev = x_std[active_rows, :k]
            result = solve_monotone_inverse(
                evaluate=lambda active, component=component, x_prev=x_prev: component.evaluate(
                    self._stack_inputs(x_prev, active)
                ),
                derivative=lambda active, component=component, x_prev=x_prev: component.evaluate_derivative_xk(
                    self._stack_inputs(x_prev, active)
                ),
                evaluate_subset=lambda active, rows, component=component, x_prev=x_prev: component.evaluate(
                    self._stack_inputs(x_prev[rows], active)
                ),
                derivative_subset=lambda active, rows, component=component, x_prev=x_prev: component.evaluate_derivative_xk(
                    self._stack_inputs(x_prev[rows], active)
                ),
                target=target,
                initial=x_curr,
                max_iter=max_iter,
                tol=tol,
                inverse_options=options,
            )
            x_curr = result.values
            x_std[active_rows, k] = x_curr
            if result.evaluation_counts is not None:
                evaluation_counts[active_rows] += np.asarray(result.evaluation_counts, dtype=int)
            if result.stats.max_residual > max_residual:
                max_residual = float(result.stats.max_residual)
            if np.any(result.failed_mask):
                local_failed = np.flatnonzero(result.failed_mask)
                failed_rows = active_rows[local_failed]
                row_failed[failed_rows] = True
                component_failure_counts[k] += int(local_failed.size)
                for local_idx, row in zip(local_failed, failed_rows):
                    residuals = result.residuals if result.residuals is not None else np.full(result.values.shape, np.nan)
                    radii = result.bracket_radii if result.bracket_radii is not None else np.full(result.values.shape, np.nan)
                    reasons = result.failure_reasons if result.failure_reasons is not None else np.full(result.values.shape, "unknown", dtype=object)
                    failures.append(
                        InverseFailure(
                            row=int(row),
                            component=int(k),
                            reason=str(reasons[local_idx] or "unknown"),
                            residual=float(residuals[local_idx]) if np.isfinite(residuals[local_idx]) else float("nan"),
                            bracket_radius=float(radii[local_idx]) if np.isfinite(radii[local_idx]) else float("nan"),
                            evaluations=(
                                int(result.evaluation_counts[local_idx])
                                if result.evaluation_counts is not None
                                else int(result.stats.evaluation_count)
                            ),
                        )
                    )
                x_std[failed_rows, k:] = np.nan
        values = invert_standardization(x_std, params)
        if np.any(row_failed):
            values[row_failed] = np.nan
        return InverseResult(
            values=values,
            diagnostics=InverseDiagnostics(
                failed_mask=row_failed,
                failures=tuple(failures),
                component_failure_counts=component_failure_counts,
                max_residual=float(max_residual),
                evaluation_counts=evaluation_counts,
            ),
        )

    def sample(
        self,
        n_samples: int,
        random_state=None,
        max_iter: int = 100,
        tol: float = 1e-8,
        inverse_options: InverseOptions | None = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray | InverseResult:
        """Draw unconditional target-space samples from the learned map.

        Parameters
        ----------
        n_samples : int
            Number of independent reference draws to invert.
        random_state : int, numpy.random.Generator, or None, optional
            Random source passed to ``numpy.random.default_rng``.
        max_iter : int, default=100
            Maximum inverse iterations per component and row.
        tol : float, default=1e-8
            Absolute scalar inverse residual tolerance.
        inverse_options : InverseOptions, optional
            Additional inverse resource limits.
        return_diagnostics : bool, default=False
            Return ``InverseResult`` instead of only samples.

        Returns
        -------
        numpy.ndarray or InverseResult
            Raw target samples, optionally with inversion diagnostics.
        """
        rng = np.random.default_rng(random_state)
        z = rng.standard_normal((n_samples, self.input_dim))
        return self.inverse(
            z,
            max_iter=max_iter,
            tol=tol,
            inverse_options=inverse_options,
            return_diagnostics=return_diagnostics,
        )

    def conditional_sample(
        self,
        n_samples: int,
        fixed_indices,
        fixed_values,
        random_state=None,
        max_iter: int = 100,
        tol: float = 1e-8,
        inverse_options: InverseOptions | None = None,
        return_diagnostics: bool = False,
    ) -> np.ndarray | InverseResult:
        """Draw exact prefix-conditioned samples by clamp-and-invert.

        Parameters
        ----------
        n_samples : int
            Number of conditional draws.
        fixed_indices : iterable of int
            Leading coordinate prefix ``[0, ..., m - 1]`` to condition on.
        fixed_values : array-like, shape (m,) or (n_samples, m)
            Raw target values for the fixed prefix.
        random_state : int, numpy.random.Generator, or None, optional
            Random source for unfixed reference coordinates.
        max_iter : int, default=100
            Maximum inverse iterations per component and row.
        tol : float, default=1e-8
            Absolute scalar inverse residual tolerance.
        inverse_options : InverseOptions, optional
            Additional inverse resource limits.
        return_diagnostics : bool, default=False
            Return ``InverseResult`` instead of only samples.

        Returns
        -------
        numpy.ndarray or InverseResult
            Raw conditional target samples, optionally with diagnostics.

        Raises
        ------
        ValueError
            If ``fixed_indices`` are not a leading coordinate prefix.

        Notes
        -----
        Arbitrary-coordinate conditionals require a map with a different
        variable ordering or a general conditional inference method.
        """
        indices = validate_leading_fixed_indices(fixed_indices, self.input_dim)
        rng = np.random.default_rng(random_state)
        reference = np.zeros((n_samples, self.input_dim), dtype=float)
        free_indices = [idx for idx in range(self.input_dim) if idx not in indices]
        if free_indices:
            reference[:, free_indices] = rng.standard_normal((n_samples, len(free_indices)))
        return self.inverse(
            reference,
            max_iter=max_iter,
            tol=tol,
            fixed_indices=indices,
            fixed_values=fixed_values,
            inverse_options=inverse_options,
            return_diagnostics=return_diagnostics,
        )

    def pullback_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the learned target log-density on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Learned target log density ``log eta(S(x)) + log |det dS/dx|``.
        """
        z = self.evaluate(x)
        return self.reference.evaluate_logpdf(z) + self.log_det(x)

    def pullback_pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the learned target density on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Learned target density.
        """
        return np.exp(self.pullback_logpdf(x))

    def conditional_pullback_logpdf(self, x: np.ndarray, fixed_indices) -> np.ndarray:
        """Evaluate a learned conditional log-density for a fixed prefix.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates, including fixed and free entries.
        fixed_indices : iterable of int
            Leading coordinate prefix ``[0, ..., m - 1]`` conditioned on.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Conditional log density of the remaining coordinates.

        Raises
        ------
        ValueError
            If indices are not a leading prefix.
        """
        fixed = validate_leading_fixed_indices(fixed_indices, self.input_dim)
        samples = ensure_2d(x, expected_dim=self.input_dim)
        free = [idx for idx in range(self.input_dim) if idx not in fixed]
        if not free:
            return np.zeros(samples.shape[0], dtype=float)
        z_free = self.evaluate(samples)[:, free]
        reference_logpdf = -0.5 * np.sum(z_free * z_free, axis=1) - 0.5 * len(free) * np.log(2.0 * np.pi)
        return reference_logpdf + self.component_log_det(samples, free)

    def conditional_pullback_pdf(self, x: np.ndarray, fixed_indices) -> np.ndarray:
        """Evaluate a learned conditional density for a fixed prefix.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates, including fixed and free entries.
        fixed_indices : iterable of int
            Leading coordinate prefix conditioned on.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Conditional density of the remaining coordinates.
        """
        return np.exp(self.conditional_pullback_logpdf(x, fixed_indices))

    def pullback_score(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the score of the learned target density on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Raw-coordinate gradient of the learned target log density.
        """
        params = self.standardization_params
        x_scaled = apply_standardization(ensure_2d(x, expected_dim=self.input_dim), params)
        score_std = self.pullback_score_standardized(x_scaled)
        return score_std / params.std[None, :]

    def pullback_score_standardized(self, x_std: np.ndarray) -> np.ndarray:
        """Evaluate the learned target score on standardized inputs.

        Parameters
        ----------
        x_std : array-like, shape (n_samples, input_dim)
            Target coordinates standardized with the fitted map state.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Standardized-coordinate gradient of the learned log density.
        """
        standardized = ensure_2d(x_std, expected_dim=self.input_dim)
        z = self.evaluate_standardized(standardized)
        reference_score = self.reference.evaluate_score(z)
        score = np.zeros_like(standardized)
        for axis in range(self.input_dim):
            total = np.zeros(standardized.shape[0], dtype=float)
            for k in range(axis, self.input_dim):
                component = self.components[k]
                inputs = standardized[:, : k + 1]
                partial = component.evaluate_partial_derivative(inputs, axis)
                total += reference_score[:, k] * partial
                deriv_xk = component.evaluate_derivative_xk(inputs)
                mixed = component.evaluate_mixed_derivative_xk(inputs, axis)
                total += mixed / deriv_xk
            score[:, axis] = total
        return score

    def pushforward_logpdf(self, z: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a target's pushforward log-density on reference inputs.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        target : callable
            Function accepting raw target coordinates and returning one target
            log-density value per row.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Pushforward log density in reference coordinates.
        """
        x = self.inverse(z)
        return np.asarray(target(x), dtype=float) - self.log_det(x)

    def pushforward_pdf(self, z: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a target's pushforward density on reference inputs.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        target : callable
            Function accepting raw target coordinates and returning target
            log-density values per row.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Pushforward density in reference coordinates.
        """
        return np.exp(self.pushforward_logpdf(z, target))

    @staticmethod
    def _stack_inputs(x_prev: np.ndarray, x_curr: np.ndarray) -> np.ndarray:
        """Assemble one component's already-solved and active coordinates."""
        if x_prev.size == 0:
            return x_curr.reshape(-1, 1)
        return np.concatenate([x_prev, x_curr.reshape(-1, 1)], axis=1)
