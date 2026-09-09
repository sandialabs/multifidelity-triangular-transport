"""Implement one coupled NHMF component and its cache-aware objectives."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize

from ..single_fidelity import SoftPlusRectifier
from ..single_fidelity.params import OptimizationParams
from ..single_fidelity.utils import ensure_2d, summarize_minimize_result

from .coefficients import NHMFComponentCoefficients, NHMFGradientBlocks
from .expansion_pair import (
    ComponentExpansionPair,
    ExpansionPairCache,
    build_component_expansion_pair,
)
from .parent_terms import CorrectedParentComponent, ParentComponentAdapter, ParentTerm, ParentTermCache


@dataclass
class NHMFComponentCache:
    """Store all precomputed data needed by one NHMF component."""

    x_hf_standardized: np.ndarray
    upper_bounds: np.ndarray
    quad_hf_standardized: np.ndarray
    shift_cache: ExpansionPairCache
    scale_caches: list[ExpansionPairCache]
    parent_caches: list[ParentTermCache]
    low_fidelity_caches: list[object | None]


@dataclass
class NHMFPrecomputedTerms:
    """Store coefficient-dependent contractions built from static NHMF caches."""

    sample_non_total: np.ndarray
    sample_pre_total: np.ndarray
    quad_pre_total: np.ndarray
    scale_non_values: list[np.ndarray]
    parent_non_values: list[np.ndarray]
    scale_pre_values: list[np.ndarray]
    parent_pre_values: list[np.ndarray]
    scale_quad_pre_values: list[np.ndarray]
    parent_quad_pre_values: list[np.ndarray]


class NonHierarchicalComponent:
    r"""Implement one coupled NHMF component and its joint objective.

    The high-fidelity component uses

    .. math::

       h_k = s_k + \sum_j a_{j,k}p_{j,k}, \qquad
       T_k = b_k + \int_0^{x_k} \operatorname{softplus}(h_k)\,dt.

    ``shift_pair`` supplies :math:`b_k` and :math:`s_k`, each
    ``scale_pair`` supplies :math:`a_{j,k}`, and a parent term supplies
    :math:`p_{j,k}`.  Cache objects contain basis evaluations only; direct
    and cached paths then use the same objective algebra.
    """

    def __init__(
        self,
        input_dim: int,
        shift_pair: ComponentExpansionPair,
        scale_pairs: list[ComponentExpansionPair],
        parent_terms: list[ParentTerm],
        low_fidelity_component_data: list[np.ndarray],
        optimization: OptimizationParams,
        hf_weight: float = 1.0,
        regularize_parent_terms: bool = True,
    ) -> None:
        """Initialize one NHMF component from shift, scale, and parent terms."""
        self.input_dim = input_dim
        self.shift_pair = shift_pair
        self.scale_pairs = scale_pairs
        self.parent_terms = parent_terms
        self.low_fidelity_component_data = [
            ensure_2d(data, expected_dim=input_dim) for data in low_fidelity_component_data
        ]
        self.optimization = optimization
        self.hf_weight = float(hf_weight)
        self.regularize_parent_terms = bool(regularize_parent_terms)
        self.rectifier = SoftPlusRectifier(shift_pair.rectifier_epsilon)
        self.rectifier_epsilon = shift_pair.rectifier_epsilon
        self._training_cache: NHMFComponentCache | None = None
        self._low_fidelity_weights = np.array(
            [data.shape[0] for data in self.low_fidelity_component_data],
            dtype=float,
        )
        self._low_fidelity_weights /= np.sum(self._low_fidelity_weights)
        self._shift_size = self.shift_pair.coeffs.size
        self._scale_sizes = [scale_pair.coeffs.size for scale_pair in self.scale_pairs]
        self._parent_sizes = [parent_term.coeffs.size for parent_term in self.parent_terms]
        self.scale_constant_optimization_result: dict[str, object] | None = None

    def _regularized_coeffs(self, coeffs: np.ndarray) -> np.ndarray:
        """Return the coefficient vector with excluded regularization blocks zeroed."""
        blocks = self._split_flat_gradient(np.asarray(coeffs, dtype=float).reshape(-1))
        if not self.regularize_parent_terms:
            blocks.parents = [np.zeros_like(parent_block) for parent_block in blocks.parents]
        return blocks.flatten()

    @property
    def coefficients(self) -> NHMFComponentCoefficients:
        """Expose the NHMF coefficients in structured form."""
        return NHMFComponentCoefficients(
            shift=self.shift_pair.coefficients,
            scales=[scale_pair.coefficients for scale_pair in self.scale_pairs],
            parents=[parent_term.coefficients for parent_term in self.parent_terms],
        )

    @property
    def coeffs(self) -> np.ndarray:
        """Return a flat optimizer view of the NHMF component coefficients."""
        return self.coefficients.flatten()

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        """Assign flattened coefficients to all NHMF parameter blocks."""
        self.coefficients.set_from_flat(coeffs)

    def low_fidelity_weights(self) -> np.ndarray:
        """Return low-fidelity objective weights based on sample counts."""
        return self._low_fidelity_weights.copy()

    @classmethod
    def _constant_coeff_positions(cls, scale_pair: ComponentExpansionPair) -> tuple[int, int]:
        """Return flat coefficient positions for one scale pair's constants."""
        return scale_pair.constant_coeff_positions()

    def _scale_constants(self) -> np.ndarray:
        """Return current nonmonotone constant values for the scale pairs."""
        return np.array([scale_pair.constant_value() for scale_pair in self.scale_pairs], dtype=float)

    def _set_scale_constants(self, constants: np.ndarray) -> None:
        """Set each scale pair to one constant in both expansion blocks."""
        values = np.asarray(constants, dtype=float).reshape(-1)
        if values.size != len(self.scale_pairs):
            raise ValueError("Scale constant vector length must match the number of parent scale pairs.")
        for value, scale_pair in zip(values, self.scale_pairs):
            scale_pair.initialize_constant(float(value))

    @staticmethod
    def _summarize_minimize_result(result, optimization: OptimizationParams) -> dict[str, object]:
        """Return a lightweight, serializable summary of a scipy minimize result."""
        summary = summarize_minimize_result(result, optimization)
        summary["reg_cst"] = 0.0
        return summary

    def _quad_inputs(self, x_hf_standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Build quadrature points in high-fidelity standardized coordinates."""
        return self.shift_pair._quad_inputs(ensure_2d(x_hf_standardized, expected_dim=self.input_dim))

    def precompute(self, x_hf_standardized: np.ndarray) -> NHMFComponentCache:
        """Precompute all HF-side sample and quadrature basis tables."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        quad_hf_standardized, upper_bounds = self._quad_inputs(samples)
        shift_cache = self.shift_pair.precompute(samples, quad_inputs=quad_hf_standardized)
        scale_caches = [
            scale_pair.precompute(samples, quad_inputs=quad_hf_standardized)
            for scale_pair in self.scale_pairs
        ]
        low_fidelity_caches = [
            parent_term.precompute_parent_data(lf_data)
            for parent_term, lf_data in zip(self.parent_terms, self.low_fidelity_component_data)
        ]
        parent_caches = [
            parent_term.precompute_from_hf(
                x_hf_standardized=samples,
                quad_hf_standardized=quad_hf_standardized,
                low_fidelity_cache=lf_cache,
            )
            for parent_term, lf_cache in zip(self.parent_terms, low_fidelity_caches)
        ]
        return NHMFComponentCache(
            x_hf_standardized=samples,
            upper_bounds=upper_bounds,
            quad_hf_standardized=quad_hf_standardized,
            shift_cache=shift_cache,
            scale_caches=scale_caches,
            parent_caches=parent_caches,
            low_fidelity_caches=low_fidelity_caches,
        )

    def evaluate_nonmonotone(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the NHMF nonmonotone contribution."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        total = self.shift_pair.evaluate_nonmonotone(samples)
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            total += (
                scale_pair.evaluate_nonmonotone(samples)
                * parent_term.non_value_from_hf(samples)
            )
        return total

    def _pre_monotone_h(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Assemble the pre-monotone argument before rectification."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        total = self.shift_pair.evaluate_pre_monotone(samples)
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            total += (
                scale_pair.evaluate_pre_monotone(samples)
                * parent_term.pre_value_from_hf(samples)
            )
        return total

    def _pre_monotone_h_quad(self, x_hf_standardized: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Assemble the quadrature pre-monotone argument without precomputation."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        quad_inputs, upper_bounds = self._quad_inputs(samples)
        q = self.shift_pair.quadrature_rule.num_points
        n = samples.shape[0]
        h_quad = self.shift_pair.evaluate_pre_monotone(quad_inputs.reshape(-1, self.input_dim)).reshape(q, n)
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            scale_pre_quad = scale_pair.evaluate_pre_monotone(quad_inputs.reshape(-1, self.input_dim)).reshape(q, n)
            parent_pre_quad = parent_term.pre_value_from_hf(
                quad_inputs.reshape(-1, self.input_dim)
            ).reshape(q, n)
            h_quad += scale_pre_quad * parent_pre_quad
        return h_quad, upper_bounds

    def _precomputed_terms(self, cache: NHMFComponentCache) -> NHMFPrecomputedTerms:
        """Build all coefficient-dependent NHMF value contractions from one cache."""
        sample_non_total = cache.shift_cache.sample_non_grad @ self.shift_pair.coeffs
        sample_pre_total = cache.shift_cache.sample_pre_grad @ self.shift_pair.coeffs
        quad_pre_total = np.einsum(
            "qnm,m->qn",
            cache.shift_cache.quad_pre_grad,
            self.shift_pair.coeffs,
        )

        scale_non_values: list[np.ndarray] = []
        parent_non_values: list[np.ndarray] = []
        scale_pre_values: list[np.ndarray] = []
        parent_pre_values: list[np.ndarray] = []
        scale_quad_pre_values: list[np.ndarray] = []
        parent_quad_pre_values: list[np.ndarray] = []

        for scale_pair, scale_cache, parent_term, parent_cache in zip(
            self.scale_pairs,
            cache.scale_caches,
            self.parent_terms,
            cache.parent_caches,
        ):
            scale_non = scale_cache.sample_non_grad @ scale_pair.coeffs
            parent_eval = parent_term.evaluate_from_cache(parent_cache)
            parent_non = parent_eval.non_value
            scale_pre = scale_cache.sample_pre_grad @ scale_pair.coeffs
            parent_pre = parent_eval.pre_value
            scale_quad_pre = np.einsum(
                "qnm,m->qn",
                scale_cache.quad_pre_grad,
                scale_pair.coeffs,
            )
            parent_quad_pre = parent_eval.quad_pre_value

            scale_non_values.append(scale_non)
            parent_non_values.append(parent_non)
            scale_pre_values.append(scale_pre)
            parent_pre_values.append(parent_pre)
            scale_quad_pre_values.append(scale_quad_pre)
            parent_quad_pre_values.append(parent_quad_pre)

            sample_non_total += scale_non * parent_non
            sample_pre_total += scale_pre * parent_pre
            quad_pre_total += scale_quad_pre * parent_quad_pre

        return NHMFPrecomputedTerms(
            sample_non_total=sample_non_total,
            sample_pre_total=sample_pre_total,
            quad_pre_total=quad_pre_total,
            scale_non_values=scale_non_values,
            parent_non_values=parent_non_values,
            scale_pre_values=scale_pre_values,
            parent_pre_values=parent_pre_values,
            scale_quad_pre_values=scale_quad_pre_values,
            parent_quad_pre_values=parent_quad_pre_values,
        )

    def evaluate_derivative_xk(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the derivative of the NHMF component with respect to ``x_k``."""
        return self.rectifier.evaluate(self._pre_monotone_h(x_hf_standardized)) + self.rectifier_epsilon

    def evaluate(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the full NHMF component on HF-standardized inputs."""
        h_quad, upper_bounds = self._pre_monotone_h_quad(x_hf_standardized)
        rectified = self.rectifier.evaluate(h_quad) + self.rectifier_epsilon
        monotone = 0.5 * upper_bounds * np.einsum(
            "q,qn->n",
            self.shift_pair.quadrature_rule.weights,
            rectified,
        )
        return self.evaluate_nonmonotone(x_hf_standardized) + monotone

    def _group_nonmonotone_coeff_gradients(self, x_hf_standardized: np.ndarray) -> list[np.ndarray]:
        """Group nonmonotone coefficient gradients by parameter block."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        groups = [self.shift_pair.evaluate_nonmonotone_coeff_gradient(samples)]
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            groups.append(
                scale_pair.evaluate_nonmonotone_coeff_gradient(samples)
                * parent_term.non_value_from_hf(samples)[:, None]
            )
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            groups.append(
                parent_term.non_coeff_grad_from_hf(samples)
                * scale_pair.evaluate_nonmonotone(samples)[:, None]
            )
        return groups

    def _group_pre_monotone_coeff_gradients(self, x_hf_standardized: np.ndarray) -> list[np.ndarray]:
        """Group pre-monotone coefficient gradients by parameter block."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        groups = [self.shift_pair.evaluate_pre_monotone_coeff_gradient(samples)]
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            groups.append(
                scale_pair.evaluate_pre_monotone_coeff_gradient(samples)
                * parent_term.pre_value_from_hf(samples)[:, None]
            )
        for scale_pair, parent_term in zip(self.scale_pairs, self.parent_terms):
            groups.append(
                parent_term.pre_coeff_grad_from_hf(samples)
                * scale_pair.evaluate_pre_monotone(samples)[:, None]
            )
        return groups

    def evaluate_coeff_gradient(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate NHMF value gradients with respect to all coefficients."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        non_groups = self._group_nonmonotone_coeff_gradients(samples)
        quad_inputs, upper_bounds = self._quad_inputs(samples)
        q = self.shift_pair.quadrature_rule.num_points
        n = samples.shape[0]
        shift_pre_quad = self.shift_pair.evaluate_pre_monotone(quad_inputs.reshape(-1, self.input_dim)).reshape(q, n)
        scale_pre_quad_list = [
            scale_pair.evaluate_pre_monotone(quad_inputs.reshape(-1, self.input_dim)).reshape(q, n)
            for scale_pair in self.scale_pairs
        ]
        parent_pre_quad_list = [
            parent_term.pre_value_from_hf(quad_inputs.reshape(-1, self.input_dim)).reshape(q, n)
            for parent_term in self.parent_terms
        ]
        h_quad = shift_pre_quad.copy()
        for scale_pre_quad, parent_pre_quad in zip(scale_pre_quad_list, parent_pre_quad_list):
            h_quad += scale_pre_quad * parent_pre_quad
        rectifier_prime = self.rectifier.evaluate_derivative(h_quad)

        pre_groups: list[np.ndarray] = []
        shift_pre_grad_quad = self.shift_pair.evaluate_pre_monotone_coeff_gradient(
            quad_inputs.reshape(-1, self.input_dim)
        ).reshape(q, n, -1)
        pre_groups.append(
            0.5
            * upper_bounds[:, None]
            * np.einsum(
                "q,qn,qnm->nm",
                self.shift_pair.quadrature_rule.weights,
                rectifier_prime,
                shift_pre_grad_quad,
            )
        )
        for scale_pair, scale_pre_quad, parent_pre_quad in zip(
            self.scale_pairs,
            scale_pre_quad_list,
            parent_pre_quad_list,
        ):
            scale_pre_grad_quad = scale_pair.evaluate_pre_monotone_coeff_gradient(
                quad_inputs.reshape(-1, self.input_dim)
            ).reshape(q, n, -1)
            pre_groups.append(
                0.5
                * upper_bounds[:, None]
                * np.einsum(
                    "q,qn,qnm->nm",
                    self.shift_pair.quadrature_rule.weights,
                    rectifier_prime * parent_pre_quad,
                    scale_pre_grad_quad,
                )
            )
        for parent_term, scale_pre_quad in zip(self.parent_terms, scale_pre_quad_list):
            parent_pre_grad_quad = parent_term.pre_coeff_grad_from_hf(
                quad_inputs.reshape(-1, self.input_dim)
            ).reshape(q, n, -1)
            pre_groups.append(
                0.5
                * upper_bounds[:, None]
                * np.einsum(
                    "q,qn,qnm->nm",
                    self.shift_pair.quadrature_rule.weights,
                    rectifier_prime * scale_pre_quad,
                    parent_pre_grad_quad,
                )
            )
        return np.concatenate(
            [non_group + pre_group for non_group, pre_group in zip(non_groups, pre_groups)],
            axis=1,
        )

    def evaluate_derivative_xk_coeff_gradient(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate gradients of the NHMF derivative with respect to coefficients."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        h_eval = self._pre_monotone_h(samples)
        rectifier_prime = self.rectifier.evaluate_derivative(h_eval)
        pre_groups = self._group_pre_monotone_coeff_gradients(samples)
        zero_non_groups = [np.zeros_like(group) for group in self._group_nonmonotone_coeff_gradients(samples)]
        return np.concatenate(
            [
                zero_non + rectifier_prime[:, None] * pre_group
                for zero_non, pre_group in zip(zero_non_groups, pre_groups)
            ],
            axis=1,
        )

    def _high_fidelity_precomp_terms(
        self,
        cache: NHMFComponentCache,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Evaluate cached values, derivatives, and coefficient gradients together."""
        terms = self._precomputed_terms(cache)
        quad_rectifier = self.rectifier.evaluate(terms.quad_pre_total) + self.rectifier_epsilon
        quad_rectifier_prime = self.rectifier.evaluate_derivative(terms.quad_pre_total)
        sample_rectifier_prime = self.rectifier.evaluate_derivative(terms.sample_pre_total)

        values = terms.sample_non_total + 0.5 * cache.upper_bounds * np.einsum(
            "q,qn->n",
            self.shift_pair.quadrature_rule.weights,
            quad_rectifier,
        )
        derivative = self.rectifier.evaluate(terms.sample_pre_total) + self.rectifier_epsilon

        non_grad_blocks = [cache.shift_cache.sample_non_grad]
        non_grad_blocks.extend(
            scale_cache.sample_non_grad * parent_non[:, None]
            for scale_cache, parent_non in zip(cache.scale_caches, terms.parent_non_values)
        )
        non_grad_blocks.extend(
            parent_cache.trainable_sample_non_grad * scale_non[:, None]
            for parent_cache, scale_non in zip(cache.parent_caches, terms.scale_non_values)
        )

        sample_pre_grad_blocks = [cache.shift_cache.sample_pre_grad]
        sample_pre_grad_blocks.extend(
            scale_cache.sample_pre_grad * parent_pre[:, None]
            for scale_cache, parent_pre in zip(cache.scale_caches, terms.parent_pre_values)
        )
        sample_pre_grad_blocks.extend(
            parent_cache.trainable_sample_pre_grad * scale_pre[:, None]
            for parent_cache, scale_pre in zip(cache.parent_caches, terms.scale_pre_values)
        )

        quad_pre_grad_blocks = [cache.shift_cache.quad_pre_grad]
        quad_pre_grad_blocks.extend(
            scale_cache.quad_pre_grad * parent_quad[:, :, None]
            for scale_cache, parent_quad in zip(cache.scale_caches, terms.parent_quad_pre_values)
        )
        quad_pre_grad_blocks.extend(
            parent_cache.trainable_quad_pre_grad * scale_quad[:, :, None]
            for parent_cache, scale_quad in zip(cache.parent_caches, terms.scale_quad_pre_values)
        )

        coeff_grad_blocks = []
        deriv_coeff_grad_blocks = []
        for non_grad_block, sample_pre_grad_block, quad_pre_grad_block in zip(
            non_grad_blocks,
            sample_pre_grad_blocks,
            quad_pre_grad_blocks,
        ):
            monotone_grad = 0.5 * cache.upper_bounds[:, None] * np.einsum(
                "q,qn,qnm->nm",
                self.shift_pair.quadrature_rule.weights,
                quad_rectifier_prime,
                quad_pre_grad_block,
            )
            coeff_grad_blocks.append(non_grad_block + monotone_grad)
            deriv_coeff_grad_blocks.append(sample_rectifier_prime[:, None] * sample_pre_grad_block)

        return (
            values,
            derivative,
            np.concatenate(coeff_grad_blocks, axis=1),
            np.concatenate(deriv_coeff_grad_blocks, axis=1),
        )

    def _high_fidelity_objective_value(
        self,
        x_hf_standardized: np.ndarray,
        cache: NHMFComponentCache | None = None,
    ) -> float:
        """Evaluate the coupled high-fidelity contribution to the objective."""
        if cache is None:
            values = self.evaluate(x_hf_standardized)
            derivative = self.evaluate_derivative_xk(x_hf_standardized)
        else:
            values, derivative, _, _ = self._high_fidelity_precomp_terms(cache)
        return float(self.hf_weight * np.mean(0.5 * values * values - np.log(derivative)))

    def _low_fidelity_objective_value(self, cache: NHMFComponentCache | None = None) -> float:
        """Evaluate the weighted standalone parent objectives."""
        total = 0.0
        for weight, parent_term, lf_data, parent_cache in zip(
            self._low_fidelity_weights,
            self.parent_terms,
            self.low_fidelity_component_data,
            cache.low_fidelity_caches if cache is not None else [None] * len(self.parent_terms),
        ):
            total += weight * parent_term.objective_on_parent_data(
                parent_term.coeffs,
                lf_data,
                cache=parent_cache,
            )
        return float(total)

    def objective(self, coeffs: np.ndarray, x_hf_standardized: np.ndarray, reg_cst: float | None = None) -> float:
        """Evaluate the coupled NHMF objective without precomputation."""
        reg = self.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        regularized_coeffs = self._regularized_coeffs(coeffs)
        return (
            self._high_fidelity_objective_value(x_hf_standardized)
            + self._low_fidelity_objective_value()
            + reg * np.dot(regularized_coeffs, regularized_coeffs)
        )

    def objective_precomp(
        self,
        coeffs: np.ndarray,
        x_hf_standardized: np.ndarray,
        reg_cst: float | None = None,
        cache: NHMFComponentCache | None = None,
    ) -> float:
        """Evaluate the coupled NHMF objective with cached bases."""
        reg = self.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        local_cache = self.precompute(x_hf_standardized) if cache is None else cache
        regularized_coeffs = self._regularized_coeffs(coeffs)
        return (
            self._high_fidelity_objective_value(x_hf_standardized, cache=local_cache)
            + self._low_fidelity_objective_value(cache=local_cache)
            + reg * np.dot(regularized_coeffs, regularized_coeffs)
        )

    def _split_flat_gradient(self, flat: np.ndarray) -> NHMFGradientBlocks:
        """Split a flat gradient vector into structured NHMF blocks."""
        start = 0
        shift = flat[start : start + self._shift_size].copy()
        start += self._shift_size
        scales = []
        for size in self._scale_sizes:
            scales.append(flat[start : start + size].copy())
            start += size
        parents = []
        for size in self._parent_sizes:
            parents.append(flat[start : start + size].copy())
            start += size
        if start != flat.size:
            raise ValueError("Gradient length does not match NHMF coefficient structure.")
        return NHMFGradientBlocks(shift=shift, scales=scales, parents=parents)

    def _high_fidelity_gradient(
        self,
        x_hf_standardized: np.ndarray,
        cache: NHMFComponentCache | None = None,
    ) -> NHMFGradientBlocks:
        """Evaluate the high-fidelity gradient in structured block form."""
        if cache is None:
            values = self.evaluate(x_hf_standardized)
            derivative = self.evaluate_derivative_xk(x_hf_standardized)
            coeff_grad = self.evaluate_coeff_gradient(x_hf_standardized)
            deriv_coeff_grad = self.evaluate_derivative_xk_coeff_gradient(x_hf_standardized)
        else:
            values, derivative, coeff_grad, deriv_coeff_grad = self._high_fidelity_precomp_terms(cache)
        hf_gradient = self.hf_weight * np.mean(
            values[:, None] * coeff_grad - (1.0 / derivative)[:, None] * deriv_coeff_grad,
            axis=0,
        )
        return self._split_flat_gradient(hf_gradient)

    def _low_fidelity_gradient(self, cache: NHMFComponentCache | None = None) -> NHMFGradientBlocks:
        """Evaluate the standalone parent gradients in structured block form."""
        blocks = NHMFGradientBlocks.zeros_like(self.coefficients)
        for idx, (weight, parent_term, lf_data, parent_cache) in enumerate(
            zip(
                self._low_fidelity_weights,
                self.parent_terms,
                self.low_fidelity_component_data,
                cache.low_fidelity_caches if cache is not None else [None] * len(self.parent_terms),
            )
        ):
            blocks.parents[idx] += weight * parent_term.gradient_on_parent_data(
                parent_term.coeffs,
                lf_data,
                cache=parent_cache,
            )
        return blocks

    def gradient(self, coeffs: np.ndarray, x_hf_standardized: np.ndarray, reg_cst: float | None = None) -> np.ndarray:
        """Evaluate the coupled NHMF gradient without precomputation."""
        reg = self.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        blocks = self._high_fidelity_gradient(x_hf_standardized)
        blocks.add_scaled(self._low_fidelity_gradient())
        return blocks.flatten() + 2.0 * reg * self._regularized_coeffs(coeffs)

    def gradient_precomp(
        self,
        coeffs: np.ndarray,
        x_hf_standardized: np.ndarray,
        reg_cst: float | None = None,
        cache: NHMFComponentCache | None = None,
    ) -> np.ndarray:
        """Evaluate the coupled NHMF gradient with cached bases."""
        reg = self.optimization.reg_cst if reg_cst is None else reg_cst
        self.set_coeffs(coeffs)
        local_cache = self.precompute(x_hf_standardized) if cache is None else cache
        blocks = self._high_fidelity_gradient(x_hf_standardized, cache=local_cache)
        blocks.add_scaled(self._low_fidelity_gradient(cache=local_cache))
        return blocks.flatten() + 2.0 * reg * self._regularized_coeffs(coeffs)

    def scale_constant_objective(
        self,
        constants: np.ndarray,
        x_hf_standardized: np.ndarray,
        cache: NHMFComponentCache | None = None,
    ) -> float:
        """Evaluate the HF-only objective for constant parent-scale weights."""
        self._set_scale_constants(constants)
        return self._high_fidelity_objective_value(x_hf_standardized, cache=cache)

    def scale_constant_gradient(
        self,
        constants: np.ndarray,
        x_hf_standardized: np.ndarray,
        cache: NHMFComponentCache | None = None,
    ) -> np.ndarray:
        """Evaluate the HF-only gradient with respect to scale constants."""
        self._set_scale_constants(constants)
        blocks = self._high_fidelity_gradient(x_hf_standardized, cache=cache)
        gradient = np.zeros(len(self.scale_pairs), dtype=float)
        for idx, (scale_pair, scale_block) in enumerate(zip(self.scale_pairs, blocks.scales)):
            non_idx, pre_idx = self._constant_coeff_positions(scale_pair)
            non_weight, pre_weight = scale_pair.constant_gradient_weights()
            gradient[idx] = non_weight * scale_block[non_idx] + pre_weight * scale_block[pre_idx]
        return gradient

    def initialize_scale_constants(
        self,
        x_hf_standardized: np.ndarray,
        optimization: OptimizationParams | None = None,
        cache: NHMFComponentCache | None = None,
    ) -> None:
        """Optimize unconstrained parent-scale constants before full coupling."""
        scale_optimization = optimization or self.optimization
        local_cache = self.precompute(x_hf_standardized) if cache is None else cache
        x0 = self._scale_constants()
        initial_fun = self.scale_constant_objective(x0, x_hf_standardized, cache=local_cache)
        result = minimize(
            self.scale_constant_objective,
            x0,
            args=(x_hf_standardized, local_cache),
            jac=self.scale_constant_gradient,
            method=scale_optimization.optimizer,
            options={"gtol": scale_optimization.gtol, "maxiter": scale_optimization.maxiter},
        )
        self._set_scale_constants(result.x)
        diagnostics = self._summarize_minimize_result(result, scale_optimization)
        diagnostics["initial_fun"] = float(initial_fun)
        diagnostics["constants"] = [float(value) for value in np.asarray(result.x, dtype=float).reshape(-1)]
        self.scale_constant_optimization_result = diagnostics

    def train(self, x_hf_standardized: np.ndarray, reg_cst: float | None = None, verbose: bool = False) -> None:
        """Optimize the NHMF component coefficients."""
        reg = self.optimization.reg_cst if reg_cst is None else reg_cst
        local_cache = self.precompute(x_hf_standardized)
        self.initialize_scale_constants(
            x_hf_standardized,
            optimization=self.optimization,
            cache=local_cache,
        )
        objective = self.objective_precomp
        gradient = self.gradient_precomp
        args = (x_hf_standardized, reg, local_cache)
        result = minimize(
            objective,
            self.coeffs,
            args=args,
            jac=gradient,
            method=self.optimization.optimizer,
            options={"gtol": self.optimization.gtol, "maxiter": self.optimization.maxiter},
        )
        self.set_coeffs(result.x)
        self._training_cache = local_cache
        if verbose:
            print(f"Completed NHMF training for component {self.input_dim}.")


__all__ = [
    "ComponentExpansionPair",
    "CorrectedParentComponent",
    "ExpansionPairCache",
    "NHMFComponentCache",
    "NHMFComponentCoefficients",
    "NHMFGradientBlocks",
    "NonHierarchicalComponent",
    "ParentComponentAdapter",
    "ParentTerm",
    "ParentTermCache",
    "build_component_expansion_pair",
]
