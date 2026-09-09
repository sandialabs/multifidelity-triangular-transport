"""Implement refactored peer non-hierarchical multifidelity maps."""

from __future__ import annotations

from copy import deepcopy
from typing import Callable

import numpy as np

from .._describe import format_nhmf_map_lines
from ..single_fidelity import OptimizationParams, Reference, TriangularMap
from ..single_fidelity.inversion import (
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
from ..single_fidelity.params import StandardizationParams
from ..single_fidelity.utils import (
    apply_standardization,
    ensure_2d,
    invert_standardization,
    standardize_dataset,
    validate_fixed_indices,
    validate_leading_fixed_indices,
)

from .components import (
    CorrectedParentComponent,
    NonHierarchicalComponent,
    ParentComponentAdapter,
    build_component_expansion_pair,
)
from .coefficients import NHMFMapCoefficients
from .params import NonHierarchicalMapParams


class NonHierarchicalTriangularMap:
    """Represent a peer NHMF transport with low-fidelity parent maps.

    Parameters
    ----------
    train_data : list of array-like
        Datasets ordered ``[X_0, ..., X_M]``. ``X_0`` is high fidelity and has
        shape ``(n_0, input_dim)``; remaining datasets are peer parents.
    nhmf_params : NonHierarchicalMapParams
        Parent-map configurations and coupled shift, scale, and trainable
        parent-correction settings.

    Notes
    -----
    Parent maps are pretrained and incorporated through the monotonicity-
    preserving shift and scale terms from Section 4.2. When
    ``use_corrections=True``, the coupled map also learns correction terms
    applied to those pretrained low-fidelity parent maps.
    """

    def __init__(
        self,
        train_data,
        nhmf_params: NonHierarchicalMapParams,
    ) -> None:
        """Build an NHMF map from high-fidelity data and peer parents.

        Parameters
        ----------
        train_data : list of array-like
            ``[X_0, ..., X_M]`` with high fidelity first.
        nhmf_params : NonHierarchicalMapParams
            Configuration matching the parent datasets.
        """
        datasets = [ensure_2d(dataset) for dataset in train_data]
        input_dim = datasets[0].shape[1]
        if any(dataset.shape[1] != input_dim for dataset in datasets[1:]):
            raise ValueError("All train_data datasets must have the same coordinate dimension.")
        if len(datasets) != 1 + len(nhmf_params.low_fidelity_map_params):
            raise ValueError("train_data must contain one high-fidelity dataset plus one dataset per low-fidelity map.")

        self.input_dim = input_dim
        self.train_data = datasets
        self.params = nhmf_params
        for map_params in self.params.low_fidelity_map_params:
            map_params.bind(input_dim)
        self.reference = Reference.standard_normal(input_dim)
        self.standardized_train_data, self.hf_standardization = self._standardize_high_fidelity_data(self.train_data[0])
        self.low_fidelity_maps: list[TriangularMap] = []
        self.low_fidelity_standardizations: list[StandardizationParams] = []
        self.components: list[NonHierarchicalComponent] = []
        self._last_use_corrections: bool | None = None
        self._pretrained_low_fidelity_component_coeffs: list[list[np.ndarray]] = []

    def __str__(self) -> str:
        """Return a human-readable summary of the NHMF configuration."""
        return "\n".join(format_nhmf_map_lines(self))

    __repr__ = __str__

    @property
    def coefficients(self) -> NHMFMapCoefficients:
        """Expose structured coefficients for the coupled NHMF map."""
        return NHMFMapCoefficients(components=[component.coefficients for component in self.components])

    @staticmethod
    def _standardize_high_fidelity_data(data: np.ndarray) -> tuple[np.ndarray, StandardizationParams]:
        """Standardize the high-fidelity training dataset."""
        return standardize_dataset(data)

    def _build_low_fidelity_maps(self) -> list[TriangularMap]:
        """Instantiate the standalone low-fidelity parent maps."""
        maps: list[TriangularMap] = []
        for dataset, map_params in zip(self.train_data[1:], self.params.low_fidelity_map_params):
            params = deepcopy(map_params)
            maps.append(
                TriangularMap(
                    dataset,
                    params,
                    reference=Reference.standard_normal(self.input_dim),
                )
            )
        return maps

    def _pretrain_low_fidelity_maps(self, verbose: bool = False) -> None:
        """Train all parent maps before the coupled NHMF stage."""
        self.low_fidelity_maps = self._build_low_fidelity_maps()
        for map_idx, low_map in enumerate(self.low_fidelity_maps, start=1):
            low_map.train(optimization=deepcopy(low_map.params.optimization), verbose=verbose)
        self.low_fidelity_standardizations = [
            StandardizationParams(
                mean=low_map.standardization_params.mean.copy(),
                std=low_map.standardization_params.std.copy(),
            )
            for low_map in self.low_fidelity_maps
        ]
        self._pretrained_low_fidelity_component_coeffs = [
            [component.coeffs.copy() for component in low_map.components]
            for low_map in self.low_fidelity_maps
        ]

    def _build_component(self, component_dim: int, use_corrections: bool) -> NonHierarchicalComponent:
        """Build one coupled NHMF component for a target dimension."""
        shift_pair = build_component_expansion_pair(
            input_dim=component_dim,
            total_order=self.params.shift_order,
            quadrature_rule=self.params.quadrature_rule,
            sigma=self.params.sigma,
            rectifier_epsilon=self.params.rectifier_epsilon,
        )
        scale_pairs = [
            build_component_expansion_pair(
                input_dim=component_dim,
                total_order=self.params.scale_order,
                quadrature_rule=self.params.quadrature_rule,
                sigma=self.params.sigma,
                rectifier_epsilon=self.params.rectifier_epsilon,
            )
            for _ in self.low_fidelity_maps
        ]
        scale_constant = 1.0 / len(self.low_fidelity_maps)
        for scale_pair in scale_pairs:
            scale_pair.initialize_constant(scale_constant)
        parent_terms = []
        low_fidelity_component_data = []
        for parent_map in self.low_fidelity_maps:
            adapter = ParentComponentAdapter(parent_map, component_dim - 1)
            low_fidelity_component_data.append(parent_map.standardized_train_data[:, :component_dim].copy())
            if use_corrections:
                correction = build_component_expansion_pair(
                    input_dim=component_dim,
                    total_order=self.params.correction_order,
                    quadrature_rule=self.params.quadrature_rule,
                    sigma=self.params.sigma,
                    rectifier_epsilon=self.params.rectifier_epsilon,
                )
                parent_terms.append(CorrectedParentComponent(adapter, correction))
            else:
                parent_terms.append(adapter)
        return NonHierarchicalComponent(
            input_dim=component_dim,
            shift_pair=shift_pair,
            scale_pairs=scale_pairs,
            parent_terms=parent_terms,
            low_fidelity_component_data=low_fidelity_component_data,
            optimization=deepcopy(self.params.optimization),
            hf_weight=self.params.hf_weight,
            regularize_parent_terms=self.params.regularize_parent_terms,
        )

    def train(
        self,
        use_corrections: bool = True,
        optimization: OptimizationParams | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the NHMF map with or without corrected parent terms.

        Parameters
        ----------
        use_corrections : bool, default=True
            Allow trainable correction terms applied to the pretrained
            low-fidelity parent maps. Set to ``False`` to use only the
            parent-informed shift and scale terms.
        optimization : OptimizationParams, optional
            Override the settings stored in ``nhmf_params``.
        verbose : bool, default=False
            Print parent and component training progress.
        """
        self._last_use_corrections = bool(use_corrections)
        self._pretrain_low_fidelity_maps(verbose=verbose)
        self.components = []
        component_optimization = deepcopy(optimization or self.params.optimization)
        for component_dim in range(1, self.input_dim + 1):
            if verbose:
                print(f"Training NHMF component {component_dim}/{self.input_dim}.")
            component = self._build_component(component_dim, use_corrections=use_corrections)
            component.optimization = deepcopy(component_optimization)
            x_hf_standardized = self.standardized_train_data[:, :component_dim]
            component.train(x_hf_standardized, reg_cst=component_optimization.reg_cst, verbose=verbose)
            self.components.append(component)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Map high-fidelity target samples to the reference.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Reference-space coordinates.
        """
        samples = ensure_2d(x, expected_dim=self.input_dim)
        x_scaled = apply_standardization(samples, self.hf_standardization)
        output = np.zeros_like(x_scaled)
        for component_dim, component in enumerate(self.components, start=1):
            output[:, component_dim - 1] = component.evaluate(x_scaled[:, :component_dim])
        return output

    def log_det(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the high-fidelity log determinant on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            ``log |det dS/dx|`` in raw target coordinates.
        """
        samples = ensure_2d(x, expected_dim=self.input_dim)
        x_scaled = apply_standardization(samples, self.hf_standardization)
        log_det = np.zeros(x_scaled.shape[0], dtype=float)
        for component_dim, component in enumerate(self.components, start=1):
            log_det += np.log(component.evaluate_derivative_xk(x_scaled[:, :component_dim]))
        log_det -= np.sum(np.log(self.hf_standardization.std))
        return log_det

    def component_log_det(self, x: np.ndarray, component_indices) -> np.ndarray:
        """Evaluate selected diagonal log-determinant terms on raw HF inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.
        component_indices : iterable of int
            Zero-based triangular component indices whose terms are summed.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Sum of selected raw-coordinate diagonal log-Jacobian terms.
        """
        indices = validate_fixed_indices(component_indices, self.input_dim)
        samples = ensure_2d(x, expected_dim=self.input_dim)
        if not indices:
            return np.zeros(samples.shape[0], dtype=float)
        x_scaled = apply_standardization(samples, self.hf_standardization)
        log_det = np.zeros(x_scaled.shape[0], dtype=float)
        for idx in indices:
            component = self.components[idx]
            log_det += np.log(component.evaluate_derivative_xk(x_scaled[:, : idx + 1]))
        log_det -= np.sum(np.log(self.hf_standardization.std[indices]))
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
        """Invert the NHMF transport from reference to high-fidelity space.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        max_iter : int, default=100
            Maximum inverse iterations per component and row.
        tol : float, default=1e-8
            Absolute scalar inverse residual tolerance.
        fixed_indices : iterable of int, optional
            Leading high-fidelity coordinate prefix to clamp.
        fixed_values : array-like, shape (m,) or (n_samples, m), optional
            Raw values of fixed high-fidelity coordinates.
        max_batch_rows : int, optional
            Maximum rows solved in one inverse batch.
        inverse_options : InverseOptions, optional
            Additional inverse resource limits.
        return_diagnostics : bool, default=False
            Return ``InverseResult`` instead of only values.

        Returns
        -------
        numpy.ndarray or InverseResult
            Raw high-fidelity coordinates, optionally with diagnostics.
        """
        z = ensure_2d(z, expected_dim=self.input_dim)
        options = inverse_options or InverseOptions()
        if max_batch_rows is None or z.shape[0] <= validate_max_batch_rows(max_batch_rows):
            result = self._inverse_rows_result(z, max_iter, tol, fixed_indices, fixed_values, options)
            return unwrap_inverse_result(result, return_diagnostics)
        batch_rows = validate_max_batch_rows(max_batch_rows)
        batches: list[InverseResult] = []
        for start in range(0, z.shape[0], batch_rows):
            stop = min(start + batch_rows, z.shape[0])
            fixed_for_batch = (
                batch_fixed_values(fixed_values, start, stop)
            )
            batches.append(self._inverse_rows_result(z[start:stop], max_iter, tol, fixed_indices, fixed_for_batch, options))
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
                fixed_std[:, idx] = (fixed_values[:, col] - self.hf_standardization.mean[idx]) / self.hf_standardization.std[idx]

        x_std = np.zeros_like(z)
        row_failed = np.zeros(z.shape[0], dtype=bool)
        failures: list[InverseFailure] = []
        component_failure_counts = np.zeros(self.input_dim, dtype=int)
        evaluation_counts = np.zeros(z.shape[0], dtype=int)
        max_residual = 0.0
        for component_idx, component in enumerate(self.components):
            if fixed_mask[component_idx]:
                x_std[:, component_idx] = fixed_std[:, component_idx]
                continue
            active_rows = np.flatnonzero(~row_failed)
            if active_rows.size == 0:
                x_std[:, component_idx] = np.nan
                continue
            target = z[active_rows, component_idx]
            x_curr = target.copy()
            x_prev = x_std[active_rows, :component_idx]
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
            x_std[active_rows, component_idx] = x_curr
            if result.evaluation_counts is not None:
                evaluation_counts[active_rows] += np.asarray(result.evaluation_counts, dtype=int)
            if result.stats.max_residual > max_residual:
                max_residual = float(result.stats.max_residual)
            if np.any(result.failed_mask):
                local_failed = np.flatnonzero(result.failed_mask)
                failed_rows = active_rows[local_failed]
                row_failed[failed_rows] = True
                component_failure_counts[component_idx] += int(local_failed.size)
                residuals = result.residuals if result.residuals is not None else np.full(result.values.shape, np.nan)
                radii = result.bracket_radii if result.bracket_radii is not None else np.full(result.values.shape, np.nan)
                reasons = result.failure_reasons if result.failure_reasons is not None else np.full(result.values.shape, "unknown", dtype=object)
                for local_idx, row in zip(local_failed, failed_rows):
                    failures.append(
                        InverseFailure(
                            row=int(row),
                            component=int(component_idx),
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
                x_std[failed_rows, component_idx:] = np.nan
        values = invert_standardization(x_std, self.hf_standardization)
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
        """Draw unconditional high-fidelity samples from the NHMF transport.

        Parameters
        ----------
        n_samples : int
            Number of reference draws to invert.
        random_state : int, numpy.random.Generator, or None, optional
            Random source for standard-normal reference draws.
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
            Raw high-fidelity samples, optionally with diagnostics.
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
        """Draw exact prefix-conditioned high-fidelity samples.

        Parameters
        ----------
        n_samples : int
            Number of conditional draws.
        fixed_indices : iterable of int
            Leading high-fidelity coordinate prefix ``[0, ..., m - 1]``.
        fixed_values : array-like, shape (m,) or (n_samples, m)
            Raw high-fidelity values for the fixed prefix.
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
            Raw conditional samples, optionally with diagnostics.

        Raises
        ------
        ValueError
            If the conditioning indices are not a leading prefix.
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
        """Evaluate the learned high-fidelity log-density on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Learned high-fidelity target log density.
        """
        z = self.evaluate(x)
        return self.reference.evaluate_logpdf(z) + self.log_det(x)

    def pullback_pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the learned high-fidelity density on raw inputs.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Learned high-fidelity target density.
        """
        return np.exp(self.pullback_logpdf(x))

    def conditional_pullback_logpdf(self, x: np.ndarray, fixed_indices) -> np.ndarray:
        """Evaluate a conditional high-fidelity log-density for a fixed prefix.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.
        fixed_indices : iterable of int
            Leading high-fidelity coordinate prefix conditioned on.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Conditional log density of remaining coordinates.
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
        """Evaluate a conditional high-fidelity density for a fixed prefix.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.
        fixed_indices : iterable of int
            Leading high-fidelity coordinate prefix conditioned on.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Conditional density of remaining coordinates.
        """
        return np.exp(self.conditional_pullback_logpdf(x, fixed_indices))

    def pushforward_logpdf(self, z: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a target's pushforward log-density through the NHMF map.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        target : callable
            Function accepting raw high-fidelity coordinates and returning one
            target log-density value per row.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Pushforward log density in reference coordinates.
        """
        x = self.inverse(z)
        return np.asarray(target(x), dtype=float) - self.log_det(x)

    def pushforward_pdf(self, z: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a target's pushforward density through the NHMF map.

        Parameters
        ----------
        z : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.
        target : callable
            Function accepting raw high-fidelity coordinates and returning
            target log-density values per row.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Pushforward density in reference coordinates.
        """
        return np.exp(self.pushforward_logpdf(z, target))

    @staticmethod
    def _stack_inputs(x_prev: np.ndarray, x_curr: np.ndarray) -> np.ndarray:
        """Assemble already-solved and active coordinates for inversion."""
        if x_prev.size == 0:
            return x_curr.reshape(-1, 1)
        return np.concatenate([x_prev, x_curr.reshape(-1, 1)], axis=1)
