"""Implement refactored hierarchical multifidelity triangular transports."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Callable

import numpy as np

from .._describe import format_hierarchical_map_lines
from ..single_fidelity import MapParams, OptimizationParams, Reference, StandardizationParams, TriangularMap
from ..single_fidelity.inversion import (
    InverseDiagnostics,
    InverseFailure,
    InverseOptions,
    InverseResult,
    batch_fixed_values,
    combine_inverse_results,
    unwrap_inverse_result,
    validate_max_batch_rows,
)
from ..single_fidelity.utils import (
    apply_standardization,
    ensure_2d,
    invert_standardization,
    standardize_dataset,
    validate_fixed_indices,
    validate_leading_fixed_indices,
)

from .params import HierarchicalMapParams
from .coefficients import HierarchicalMapCoefficients


@dataclass
class HierarchicalStageState:
    """Store the configuration and trained state for one fidelity stage."""

    fidelity_idx: int
    map_params: MapParams
    train_data: np.ndarray
    stage_train_data: np.ndarray | None = None
    reference: Reference | None = None
    map: TriangularMap | None = None


class HierarchicalTriangularMap:
    """Compose stagewise triangular maps across a fidelity hierarchy.

    Parameters
    ----------
    train_data : list of array-like
        Datasets ordered ``[X_0, ..., X_M]`` with high fidelity first. Each
        has shape ``(n_samples_i, input_dim)``.
    hierarchical_params : HierarchicalMapParams
        One single-fidelity configuration per dataset.

    Notes
    -----
    :meth:`train` selects the changing-reference or fixed-reference
    formulation. The trained composition maps ``pi_0`` to ``eta``.
    """

    TRAINING_METHODS = {"changing_reference", "fixed_reference"}

    def __init__(
        self,
        train_data,
        hierarchical_params: HierarchicalMapParams,
    ) -> None:
        """Build a hierarchy from high-to-low ordered datasets.

        Parameters
        ----------
        train_data : list of array-like
            Datasets ordered ``[X_0, ..., X_M]``; each is two-dimensional.
        hierarchical_params : HierarchicalMapParams
            Stage configurations matching ``train_data``.
        """
        datasets = [ensure_2d(dataset) for dataset in train_data]
        input_dim = datasets[0].shape[1]
        if any(dataset.shape[1] != input_dim for dataset in datasets[1:]):
            raise ValueError("All train_data datasets must have the same coordinate dimension.")
        if len(datasets) != len(hierarchical_params.fidelity_map_params):
            raise ValueError("train_data and fidelity_map_params must have the same length.")

        self.input_dim = input_dim
        self.train_data = datasets
        self.params = hierarchical_params
        for map_params in self.params.fidelity_map_params:
            map_params.bind(input_dim)
        self.num_fidelities = len(datasets)
        standardized_payloads = [standardize_dataset(dataset) for dataset in datasets]
        self.standardized_train_data = [payload[0] for payload in standardized_payloads]
        self.fidelity_standardizations = [payload[1] for payload in standardized_payloads]
        self._identity_standardization = StandardizationParams(
            mean=np.zeros(self.input_dim, dtype=float),
            std=np.ones(self.input_dim, dtype=float),
        )
        self._last_training_method: str | None = None
        self._stages = [
            HierarchicalStageState(
                fidelity_idx=fidelity_idx,
                map_params=map_params,
                train_data=dataset,
            )
            for fidelity_idx, (dataset, map_params) in enumerate(
                zip(self.train_data, self.params.fidelity_map_params)
            )
        ]

    def __str__(self) -> str:
        """Return a human-readable summary of the hierarchical map."""
        return "\n".join(format_hierarchical_map_lines(self))

    __repr__ = __str__

    @property
    def stages(self) -> list[HierarchicalStageState]:
        """Return the configured stage states in fidelity order."""
        return self._stages

    @property
    def maps(self) -> list[TriangularMap]:
        """Return trained stage maps in fidelity order."""
        return [stage.map for stage in self._stages if stage.map is not None]

    @property
    def coefficients(self) -> HierarchicalMapCoefficients:
        """Expose structured coefficients for each trained stage map."""
        return HierarchicalMapCoefficients(stages=[stage.map.coefficients for stage in self._stages if stage.map is not None])

    @property
    def evaluation_order(self) -> list[int]:
        """Return the stage order used by the currently trained hierarchy."""
        method = self._require_trained_method()
        if method == "changing_reference":
            return list(range(self.num_fidelities))
        return list(range(self.num_fidelities - 1, -1, -1))

    def _require_trained_method(self) -> str:
        """Return the trained method or raise if the hierarchy is untrained."""
        if self._last_training_method is None:
            raise ValueError("The hierarchical map must be trained before evaluation, inversion, or density queries.")
        return self._last_training_method

    def _stage_indices_low_to_high(self) -> range:
        """Iterate from the lowest fidelity stage to the highest."""
        return range(self.num_fidelities - 1, -1, -1)

    def _reset_stage_training_state(self) -> None:
        """Clear all stage references, derived data, and trained maps."""
        for stage in self._stages:
            stage.stage_train_data = None
            stage.reference = None
            stage.map = None

    def _validate_training_method(self, method: str | None) -> str:
        """Validate the requested hierarchical training method."""
        if method is None:
            raise ValueError("train() requires method='changing_reference' or method='fixed_reference'.")
        if method not in self.TRAINING_METHODS:
            raise ValueError(
                f"Unsupported training method '{method}'. Expected one of {sorted(self.TRAINING_METHODS)}."
            )
        return method

    def _stage(self, fidelity_idx: int) -> HierarchicalStageState:
        """Return one configured stage by fidelity index."""
        return self._stages[fidelity_idx]

    def _require_stage_map(self, fidelity_idx: int) -> TriangularMap:
        """Return the trained map for one stage or raise if it is missing."""
        stage_map = self._stage(fidelity_idx).map
        if stage_map is None:
            raise ValueError(f"Fidelity stage {fidelity_idx} is not trained.")
        return stage_map

    def _standard_reference(self) -> Reference:
        """Return the standard normal reference used by fixed-reference stages."""
        return Reference.standard_normal(self.input_dim)

    def _standardized_fidelity_data(self, fidelity_idx: int) -> np.ndarray:
        """Return independently standardized data for one fidelity stage."""
        return self.standardized_train_data[fidelity_idx]

    def _standardize_high_fidelity_input(self, x: np.ndarray) -> np.ndarray:
        """Map raw high-fidelity coordinates into the HMF composition chart."""
        return apply_standardization(ensure_2d(x, expected_dim=self.input_dim), self.fidelity_standardizations[0])

    def _invert_high_fidelity_standardization(self, x_std: np.ndarray) -> np.ndarray:
        """Map high-fidelity standardized coordinates back to raw coordinates."""
        return invert_standardization(ensure_2d(x_std, expected_dim=self.input_dim), self.fidelity_standardizations[0])

    def _changing_reference_for_stage(self, fidelity_idx: int) -> Reference:
        """Construct the map-induced reference for one changing-reference stage."""
        if fidelity_idx == self.num_fidelities - 1:
            return self._standard_reference()
        lower_map = self._require_stage_map(fidelity_idx + 1)
        return Reference(
            logpdf_fn=lower_map.pullback_logpdf,
            score_fn=lower_map.pullback_score,
        )

    def _create_stage_map(self, fidelity_idx: int, train_data: np.ndarray, reference: Reference) -> TriangularMap:
        """Instantiate one stage map with a chosen reference density."""
        map_params = deepcopy(self._stage(fidelity_idx).map_params)
        return TriangularMap(
            train_data,
            map_params,
            reference=reference,
            standardization=self._identity_standardization,
        )

    def _fixed_reference_stage_data(self, fidelity_idx: int) -> np.ndarray:
        """Build transformed stage data for fixed-reference training."""
        stage_data = self._standardized_fidelity_data(fidelity_idx).copy()
        for lower_idx in range(self.num_fidelities - 1, fidelity_idx, -1):
            stage_data = self._require_stage_map(lower_idx).evaluate(stage_data)
        return stage_data

    def _train_changing_reference(
        self,
        optimization: OptimizationParams | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the hierarchy with map-induced references from low to high fidelity."""
        for stage_number, fidelity_idx in enumerate(self._stage_indices_low_to_high(), start=1):
            stage = self._stage(fidelity_idx)
            reference = self._changing_reference_for_stage(fidelity_idx)
            stage_data = self._standardized_fidelity_data(fidelity_idx)
            current_map = self._create_stage_map(fidelity_idx, train_data=stage_data, reference=reference)

            current_optimization = deepcopy(optimization or current_map.params.optimization)
            current_map.train_aao(optimization=current_optimization, verbose=False)

            stage.reference = reference
            stage.stage_train_data = stage_data.copy()
            stage.map = current_map

            if verbose:
                print(
                    f"Completed stage {stage_number}/{self.num_fidelities} "
                    f"for method 'changing_reference' (fidelity index {fidelity_idx})."
                )

    def _train_fixed_reference(
        self,
        optimization: OptimizationParams | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the hierarchy with Gaussian references and transformed stage data."""
        for stage_number, fidelity_idx in enumerate(self._stage_indices_low_to_high(), start=1):
            stage = self._stage(fidelity_idx)
            stage_data = self._fixed_reference_stage_data(fidelity_idx)
            reference = self._standard_reference()
            current_map = self._create_stage_map(fidelity_idx, train_data=stage_data, reference=reference)

            current_optimization = deepcopy(optimization or current_map.params.optimization)
            current_map.train(optimization=current_optimization, verbose=False)

            stage.reference = reference
            stage.stage_train_data = stage_data.copy()
            stage.map = current_map

            if verbose:
                print(
                    f"Completed stage {stage_number}/{self.num_fidelities} "
                    f"for method 'fixed_reference' (fidelity index {fidelity_idx})."
                )

    def train(
        self,
        method: str | None = None,
        optimization: OptimizationParams | None = None,
        verbose: bool = False,
    ) -> None:
        """Train the hierarchy stage-by-stage.

        Parameters
        ----------
        method : {"changing_reference", "fixed_reference"}, optional
            Hierarchical formulation from manuscript Section 4.1.
        optimization : OptimizationParams, optional
            Optimizer settings applied to each stage.
        verbose : bool, default=False
            Print stage completion messages.

        Notes
        -----
        Training data are provided in high-to-low fidelity order, but training
        always proceeds from the lowest fidelity stage upward.
        """
        method = self._validate_training_method(method)
        self._reset_stage_training_state()
        self._last_training_method = method

        if method == "changing_reference":
            self._train_changing_reference(optimization=optimization, verbose=verbose)
        else:
            self._train_fixed_reference(optimization=optimization, verbose=verbose)

    def _compose_forward(
        self,
        x: np.ndarray,
        include_log_det: bool = False,
        return_states: bool = False,
    ):
        """Compose trained stage maps forward through the hierarchy."""
        self._require_trained_method()
        current = self._standardize_high_fidelity_input(x)
        states = [current.copy()] if return_states else None
        total_log_det = (
            np.full(current.shape[0], -np.sum(np.log(self.fidelity_standardizations[0].std)), dtype=float)
            if include_log_det
            else None
        )

        for fidelity_idx in self.evaluation_order:
            current_map = self._require_stage_map(fidelity_idx)
            if include_log_det:
                total_log_det += current_map.log_det(current)
            current = current_map.evaluate(current)
            if states is not None:
                states.append(current.copy())

        if include_log_det and states is not None:
            return current, total_log_det, states
        if include_log_det:
            return current, total_log_det
        if states is not None:
            return current, states
        return current

    def partial_evaluations(self, x: np.ndarray) -> list[np.ndarray]:
        """Return states after high-fidelity pre-standardization and each stage.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        list of numpy.ndarray
            Standardized high-fidelity input followed by the state after each
            stage in evaluation order.
        """
        _, states = self._compose_forward(x, return_states=True)
        return states

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Map high-fidelity target samples through the hierarchy to ``eta``.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Reference-space coordinates.
        """
        return self._compose_forward(x)

    def component_log_det(self, x: np.ndarray, component_indices) -> np.ndarray:
        """Evaluate selected composed diagonal log-determinant terms.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.
        component_indices : iterable of int
            Zero-based diagonal component indices to sum.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Raw-coordinate sum of selected composed diagonal log-Jacobian
            terms.
        """
        self._require_trained_method()
        indices = validate_fixed_indices(component_indices, self.input_dim)
        samples = ensure_2d(x, expected_dim=self.input_dim)
        if not indices:
            return np.zeros(samples.shape[0], dtype=float)
        current = self._standardize_high_fidelity_input(samples)
        total = np.full(
            samples.shape[0],
            -np.sum(np.log(self.fidelity_standardizations[0].std[indices])),
            dtype=float,
        )
        for fidelity_idx in self.evaluation_order:
            current_map = self._require_stage_map(fidelity_idx)
            total += current_map.component_log_det(current, indices)
            current = current_map.evaluate(current)
        return total

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
        """Invert the full hierarchy from reference space to high fidelity.

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
            Maximum rows solved in one batch.
        inverse_options : InverseOptions, optional
            Additional inverse resource limits.
        return_diagnostics : bool, default=False
            Return ``InverseResult`` instead of only values.

        Returns
        -------
        numpy.ndarray or InverseResult
            Raw high-fidelity coordinates, optionally with diagnostics.

        Raises
        ------
        ValueError
            If shapes or leading-prefix conditioning are invalid.
        RuntimeError
            If inversion fails and diagnostics were not requested.
        """
        self._require_trained_method()
        output = ensure_2d(z, expected_dim=self.input_dim)
        options = inverse_options or InverseOptions()
        if max_batch_rows is None or output.shape[0] <= validate_max_batch_rows(max_batch_rows):
            result = self._inverse_rows_result(output, max_iter, tol, fixed_indices, fixed_values, options)
            return unwrap_inverse_result(result, return_diagnostics)
        batch_rows = validate_max_batch_rows(max_batch_rows)
        batches: list[InverseResult] = []
        for start in range(0, output.shape[0], batch_rows):
            stop = min(start + batch_rows, output.shape[0])
            fixed_for_batch = (
                batch_fixed_values(fixed_values, start, stop)
            )
            batches.append(self._inverse_rows_result(output[start:stop], max_iter, tol, fixed_indices, fixed_for_batch, options))
        result = self._combine_inverse_results(batches)
        return unwrap_inverse_result(result, return_diagnostics)

    def _combine_inverse_results(self, batches: list[InverseResult]) -> InverseResult:
        """Combine row-batched inverse results."""
        return combine_inverse_results(batches)

    def _inverse_rows_result(
        self,
        output: np.ndarray,
        max_iter: int,
        tol: float,
        fixed_indices,
        fixed_values,
        inverse_options: InverseOptions,
    ) -> InverseResult:
        """Invert one row batch through the complete hierarchy."""
        output = np.asarray(output, dtype=float).copy()
        if fixed_indices is not None and len(fixed_indices) > 0:
            return self._inverse_conditioned_stagewise_result(
                output,
                max_iter=max_iter,
                tol=tol,
                fixed_indices=fixed_indices,
                fixed_values=fixed_values,
                inverse_options=inverse_options,
            )
        target_space_map_idx = self.evaluation_order[0]
        failures: list[InverseFailure] = []
        failed_mask = np.zeros(output.shape[0], dtype=bool)
        component_counts = np.zeros(self.input_dim, dtype=int)
        evaluation_counts = np.zeros(output.shape[0], dtype=int)
        max_residual = 0.0
        for fidelity_idx in reversed(self.evaluation_order):
            current_map = self._require_stage_map(fidelity_idx)
            active_rows = np.flatnonzero(~failed_mask)
            if active_rows.size == 0:
                break
            kwargs = {"max_iter": max_iter, "tol": tol,
                      "inverse_options": inverse_options,
                      "return_diagnostics": True}
            if fidelity_idx == target_space_map_idx:
                kwargs["fixed_indices"] = fixed_indices
                kwargs["fixed_values"] = fixed_values
            stage_result = current_map.inverse(output[active_rows], **kwargs)
            output[active_rows] = stage_result.values
            stage_failed = np.asarray(stage_result.diagnostics.failed_mask, dtype=bool)
            failed_mask[active_rows[stage_failed]] = True
            component_counts += stage_result.diagnostics.component_failure_counts
            if stage_result.diagnostics.evaluation_counts is not None:
                evaluation_counts[active_rows] += np.asarray(stage_result.diagnostics.evaluation_counts, dtype=int)
            if stage_result.diagnostics.max_residual > max_residual:
                max_residual = float(stage_result.diagnostics.max_residual)
            failures.extend(
                InverseFailure(
                    row=int(active_rows[failure.row]),
                    component=failure.component,
                    reason=failure.reason,
                    residual=failure.residual,
                    bracket_radius=failure.bracket_radius,
                    evaluations=failure.evaluations,
                    stage=fidelity_idx,
                )
                for failure in stage_result.diagnostics.failures
            )
        values = self._invert_high_fidelity_standardization(output)
        if np.any(failed_mask):
            values[failed_mask] = np.nan
        return InverseResult(
            values=values,
            diagnostics=InverseDiagnostics(
                failed_mask=failed_mask,
                failures=tuple(failures),
                component_failure_counts=component_counts,
                max_residual=float(max_residual),
                evaluation_counts=evaluation_counts,
            ),
        )

    def _conditioned_stage_fixed_values(
        self,
        fixed_indices,
        fixed_values,
        n_rows: int,
    ) -> tuple[list[int], dict[int, np.ndarray]]:
        """Propagate a fixed leading target prefix into every stage chart."""
        indices = validate_leading_fixed_indices(fixed_indices, self.input_dim)
        if fixed_values is None:
            raise ValueError("fixed_values must be provided when fixed_indices are specified.")
        values = np.asarray(fixed_values, dtype=float)
        if values.ndim == 1:
            values = np.broadcast_to(values, (n_rows, values.shape[0]))
        if values.shape != (n_rows, len(indices)):
            raise ValueError("fixed_values must have shape (n_samples, len(fixed_indices)) or be 1D.")

        # A triangular map's leading outputs depend only on its leading inputs,
        # so zeros are a valid completion for the free trailing coordinates.
        completed = np.zeros((n_rows, self.input_dim), dtype=float)
        completed[:, indices] = values
        current = self._standardize_high_fidelity_input(completed)
        stage_values: dict[int, np.ndarray] = {}
        for fidelity_idx in self.evaluation_order:
            stage_values[fidelity_idx] = current[:, indices].copy()
            current = self._require_stage_map(fidelity_idx).evaluate(current)
        return indices, stage_values

    def _inverse_conditioned_stagewise_result(
        self,
        output: np.ndarray,
        max_iter: int,
        tol: float,
        fixed_indices,
        fixed_values,
        inverse_options: InverseOptions,
    ) -> InverseResult:
        """Invert each hierarchy stage using propagated leading fixed values."""
        output = ensure_2d(output, expected_dim=self.input_dim).copy()
        indices, stage_fixed_values = self._conditioned_stage_fixed_values(
            fixed_indices,
            fixed_values,
            output.shape[0],
        )
        row_failed = np.zeros(output.shape[0], dtype=bool)
        failures: list[InverseFailure] = []
        component_counts = np.zeros(self.input_dim, dtype=int)
        evaluation_counts = np.zeros(output.shape[0], dtype=int)
        max_residual = 0.0

        for fidelity_idx in reversed(self.evaluation_order):
            current_map = self._require_stage_map(fidelity_idx)
            active_rows = np.flatnonzero(~row_failed)
            if active_rows.size == 0:
                break
            stage_result = current_map.inverse(
                output[active_rows],
                max_iter=max_iter,
                tol=tol,
                fixed_indices=indices,
                fixed_values=stage_fixed_values[fidelity_idx][active_rows],
                inverse_options=inverse_options,
                return_diagnostics=True,
            )
            output[active_rows] = stage_result.values
            component_counts += stage_result.diagnostics.component_failure_counts
            if stage_result.diagnostics.evaluation_counts is not None:
                evaluation_counts[active_rows] += np.asarray(stage_result.diagnostics.evaluation_counts, dtype=int)
            if stage_result.diagnostics.max_residual > max_residual:
                max_residual = float(stage_result.diagnostics.max_residual)
            for failure in stage_result.diagnostics.failures:
                failures.append(
                    InverseFailure(
                        row=int(active_rows[int(failure.row)]),
                        component=int(failure.component),
                        reason=str(failure.reason),
                        residual=float(failure.residual),
                        bracket_radius=float(failure.bracket_radius),
                        evaluations=int(failure.evaluations),
                        stage=int(fidelity_idx),
                    )
                )
            stage_failed = np.asarray(stage_result.diagnostics.failed_mask, dtype=bool)
            if np.any(stage_failed):
                failed_rows = active_rows[stage_failed]
                row_failed[failed_rows] = True
                output[failed_rows] = np.nan

        values = self._invert_high_fidelity_standardization(output)
        if np.any(row_failed):
            values[row_failed] = np.nan
        return InverseResult(
            values=values,
            diagnostics=InverseDiagnostics(
                failed_mask=row_failed,
                failures=tuple(failures),
                component_failure_counts=component_counts,
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
        """Draw unconditional high-fidelity samples from the hierarchy.

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
        self._require_trained_method()
        rng = np.random.default_rng(random_state)
        reference = rng.standard_normal((n_samples, self.input_dim))
        return self.inverse(
            reference,
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
        """Draw exact prefix-conditioned samples under the hierarchy.

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
            Raw conditional high-fidelity samples, optionally with diagnostics.

        Raises
        ------
        ValueError
            If the conditioning indices are not a leading prefix.
        """
        self._require_trained_method()
        fixed_indices = validate_leading_fixed_indices(fixed_indices, self.input_dim)
        if len(fixed_indices) == 0:
            return self.sample(
                n_samples,
                random_state=random_state,
                max_iter=max_iter,
                tol=tol,
                inverse_options=inverse_options,
                return_diagnostics=return_diagnostics,
            )
        rng = np.random.default_rng(random_state)
        reference = np.zeros((n_samples, self.input_dim), dtype=float)
        free_indices = [idx for idx in range(self.input_dim) if idx not in fixed_indices]
        if free_indices:
            reference[:, free_indices] = rng.standard_normal((n_samples, len(free_indices)))
        return self.inverse(
            reference,
            max_iter=max_iter,
            tol=tol,
            fixed_indices=fixed_indices,
            fixed_values=fixed_values,
            inverse_options=inverse_options,
            return_diagnostics=return_diagnostics,
        )

    def log_det(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the composed log determinant in target coordinates.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            ``log |det dS/dx|`` for the complete hierarchy.
        """
        _, total_log_det = self._compose_forward(x, include_log_det=True)
        return total_log_det

    def pullback_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the learned high-fidelity target log-density.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Raw high-fidelity target coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Learned high-fidelity target log density.
        """
        reference = self._standard_reference()
        z, log_det = self._compose_forward(x, include_log_det=True)
        return reference.evaluate_logpdf(z) + log_det

    def pullback_pdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the learned high-fidelity target density.

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
            Raw high-fidelity coordinates.
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
            Raw high-fidelity coordinates.
        fixed_indices : iterable of int
            Leading high-fidelity coordinate prefix conditioned on.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Conditional density of remaining coordinates.
        """
        return np.exp(self.conditional_pullback_logpdf(x, fixed_indices))

    def pushforward_logpdf(self, z: np.ndarray, target: Callable[[np.ndarray], np.ndarray]) -> np.ndarray:
        """Evaluate a target's pushforward log-density through the hierarchy.

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
        """Evaluate a target's pushforward density through the hierarchy.

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
