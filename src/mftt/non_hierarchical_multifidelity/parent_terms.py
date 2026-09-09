"""Implement NHMF parent-term adapters and corrected parent terms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from ..single_fidelity import SoftPlusRectifier, TriangularMap
from ..single_fidelity.components import ComponentCache, ComponentCoefficients, ExpansionCoefficients, MapComponent
from ..single_fidelity.utils import ensure_2d

from .coefficients import FlattenableCoefficients
from .expansion_pair import ComponentExpansionPair


@dataclass
class ParentTermCache:
    """Store cached parent sample and quadrature evaluations for one input space."""

    sample_inputs: np.ndarray
    quad_inputs: np.ndarray
    base_sample_non_basis: np.ndarray
    base_sample_pre_basis: np.ndarray
    base_quad_pre_basis: np.ndarray
    trainable_sample_non_grad: np.ndarray
    trainable_sample_pre_grad: np.ndarray
    trainable_quad_pre_grad: np.ndarray
    low_fidelity_cache: object | None = None
    correction_sample_non_basis: np.ndarray | None = None
    correction_sample_pre_basis: np.ndarray | None = None
    correction_quad_pre_basis: np.ndarray | None = None


@dataclass
class ParentTermEvaluation:
    """Values and coefficient Jacobians of one parent term.

    Grouping these six arrays makes the NHMF equations visible and keeps the
    direct and cached paths from reimplementing the same parent contractions.
    """

    non_value: np.ndarray
    pre_value: np.ndarray
    quad_pre_value: np.ndarray
    non_coeff_grad: np.ndarray
    pre_coeff_grad: np.ndarray
    quad_pre_coeff_grad: np.ndarray


class ParentTerm(Protocol):
    """Small interface consumed by the coupled NHMF algebra."""

    @property
    def coeffs(self) -> np.ndarray:
        """Return a flat view of the trainable parent block."""

    @property
    def coefficients(self) -> FlattenableCoefficients:
        """Expose structured trainable coefficients."""

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        """Assign trainable coefficients."""

    def precompute_from_hf(
        self,
        x_hf_standardized: np.ndarray,
        quad_hf_standardized: np.ndarray,
        low_fidelity_cache: object | None = None,
    ) -> ParentTermCache:
        """Cache parent evaluations induced by HF-standardized coupled inputs."""

    def precompute_parent_data(self, x_parent_standardized: np.ndarray) -> object | None:
        """Cache standalone low-fidelity parent training data."""

    def evaluate_from_cache(self, cache: ParentTermCache) -> ParentTermEvaluation:
        """Evaluate values and coefficient Jacobians from a cache."""

    def objective_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> float:
        """Evaluate the standalone parent objective on parent-standardized data."""

    def gradient_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> np.ndarray:
        """Evaluate the standalone parent gradient on parent-standardized data."""


class ParentComponentAdapter:
    """Adapt a pretrained low-fidelity component for the coupled NHMF stage."""

    def __init__(
        self,
        transport_map: TriangularMap,
        component_index: int,
    ) -> None:
        """Bind one parent component trained in its own local standardization."""
        self.transport_map = transport_map
        self.component_index = component_index
        self.component_dim = component_index + 1
        self.component: MapComponent = transport_map.components[component_index]

    @property
    def coeffs(self) -> np.ndarray:
        """Return a copy of the underlying parent coefficients."""
        return self.coefficients.flatten().copy()

    @property
    def coefficients(self) -> ComponentCoefficients:
        """Expose the underlying parent coefficients in structured form."""
        return self.component.coefficients

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        """Update the underlying parent coefficients."""
        self.component.set_coeffs(coeffs)

    def precompute_from_hf(
        self,
        x_hf_standardized: np.ndarray,
        quad_hf_standardized: np.ndarray,
        low_fidelity_cache: object | None = None,
    ) -> ParentTermCache:
        """Cache parent basis evaluations on HF-standardized coupled inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.component_dim)
        quad_samples = np.asarray(quad_hf_standardized, dtype=float)
        base_sample_non_basis = self.component.evaluate_nonmonotone_coeff_gradient(samples)
        base_sample_pre_basis = self.component.evaluate_pre_monotone_coeff_gradient(samples)
        base_quad_pre_basis = self.component.evaluate_pre_monotone_coeff_gradient(
            quad_samples.reshape(-1, self.component_dim)
        ).reshape(quad_samples.shape[0], quad_samples.shape[1], -1)
        return ParentTermCache(
            sample_inputs=samples,
            quad_inputs=quad_samples,
            base_sample_non_basis=base_sample_non_basis,
            base_sample_pre_basis=base_sample_pre_basis,
            base_quad_pre_basis=base_quad_pre_basis,
            trainable_sample_non_grad=base_sample_non_basis,
            trainable_sample_pre_grad=base_sample_pre_basis,
            trainable_quad_pre_grad=base_quad_pre_basis,
            low_fidelity_cache=low_fidelity_cache,
        )

    def evaluate_from_cache(self, cache: ParentTermCache) -> ParentTermEvaluation:
        """Evaluate the parent term and all Jacobian blocks from cache."""
        return ParentTermEvaluation(
            non_value=self.non_value_from_cache(cache),
            pre_value=self.pre_value_from_cache(cache),
            quad_pre_value=self.pre_quad_value_from_cache(cache),
            non_coeff_grad=self.non_coeff_grad_from_cache(cache),
            pre_coeff_grad=self.pre_coeff_grad_from_cache(cache),
            quad_pre_coeff_grad=self.pre_quad_coeff_grad_from_cache(cache),
        )

    def precompute_parent_data(self, x_parent_standardized: np.ndarray) -> ComponentCache:
        """Precompute parent training data on parent-local standardized coordinates."""
        return self.component.precompute(ensure_2d(x_parent_standardized, expected_dim=self.component_dim))

    def non_value_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the parent nonmonotone value on HF-standardized coupled inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.component_dim)
        return self.component.evaluate_nonmonotone(samples)

    def pre_value_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the parent pre-monotone value on HF-standardized coupled inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.component_dim)
        return self.component.evaluate_pre_monotone(samples)

    def non_coeff_grad_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate parent nonmonotone coefficient gradients on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.component_dim)
        return self.component.evaluate_nonmonotone_coeff_gradient(samples)

    def pre_coeff_grad_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate parent pre-monotone coefficient gradients on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.component_dim)
        return self.component.evaluate_pre_monotone_coeff_gradient(samples)

    def non_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached parent nonmonotone values."""
        return cache.base_sample_non_basis @ self.component.coeffs

    def pre_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached parent pre-monotone values."""
        return cache.base_sample_pre_basis @ self.component.coeffs

    def pre_quad_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached parent quadrature pre-monotone values."""
        return np.einsum("qnm,m->qn", cache.base_quad_pre_basis, self.component.coeffs)

    def non_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached parent nonmonotone coefficient gradients."""
        return cache.trainable_sample_non_grad

    def pre_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached parent pre-monotone coefficient gradients."""
        return cache.trainable_sample_pre_grad

    def pre_quad_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached quadrature pre-monotone coefficient gradients."""
        return cache.trainable_quad_pre_grad

    def objective_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> float:
        """Evaluate the parent objective on parent-standardized data."""
        return self.component.objective(coeffs, x_parent_standardized, cache=cache)

    def gradient_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> np.ndarray:
        """Evaluate the parent gradient on parent-standardized data."""
        return self.component.gradient(coeffs, x_parent_standardized, cache=cache)


class CorrectedParentComponent:
    """Augment a parent component with trainable correction expansions."""

    def __init__(self, base_parent: ParentComponentAdapter, correction: ComponentExpansionPair) -> None:
        """Bind a base parent adapter and its correction term."""
        self.base_parent = base_parent
        self.correction = correction
        self.rectifier = SoftPlusRectifier(correction.rectifier_epsilon)
        self.rectifier_epsilon = correction.rectifier_epsilon
        self.input_dim = correction.input_dim

    @property
    def coeffs(self) -> np.ndarray:
        """Return a copy of the correction coefficients."""
        return self.coefficients.flatten().copy()

    @property
    def coefficients(self) -> ExpansionCoefficients:
        """Expose the correction coefficients in structured form."""
        return self.correction.coefficients

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        """Update the correction coefficients."""
        self.correction.set_coeffs(coeffs)

    def precompute_from_hf(
        self,
        x_hf_standardized: np.ndarray,
        quad_hf_standardized: np.ndarray,
        low_fidelity_cache: object | None = None,
    ) -> ParentTermCache:
        """Precompute corrected-parent bases on HF-standardized coupled inputs."""
        base_cache = self.base_parent.precompute_from_hf(
            x_hf_standardized=x_hf_standardized,
            quad_hf_standardized=quad_hf_standardized,
            low_fidelity_cache=low_fidelity_cache,
        )
        base_cache.correction_sample_non_basis = self.correction.evaluate_nonmonotone_coeff_gradient(
            base_cache.sample_inputs
        )
        base_cache.correction_sample_pre_basis = self.correction.evaluate_pre_monotone_coeff_gradient(
            base_cache.sample_inputs
        )
        base_cache.correction_quad_pre_basis = self.correction.evaluate_pre_monotone_coeff_gradient(
            base_cache.quad_inputs.reshape(-1, self.input_dim)
        ).reshape(base_cache.quad_inputs.shape[0], base_cache.quad_inputs.shape[1], -1)
        base_cache.trainable_sample_non_grad = base_cache.correction_sample_non_basis
        base_cache.trainable_sample_pre_grad = base_cache.correction_sample_pre_basis
        base_cache.trainable_quad_pre_grad = base_cache.correction_quad_pre_basis
        return base_cache

    def evaluate_from_cache(self, cache: ParentTermCache) -> ParentTermEvaluation:
        """Evaluate the corrected parent and Jacobian blocks from cache."""
        return ParentTermEvaluation(
            non_value=self.non_value_from_cache(cache),
            pre_value=self.pre_value_from_cache(cache),
            quad_pre_value=self.pre_quad_value_from_cache(cache),
            non_coeff_grad=self.non_coeff_grad_from_cache(cache),
            pre_coeff_grad=self.pre_coeff_grad_from_cache(cache),
            quad_pre_coeff_grad=self.pre_quad_coeff_grad_from_cache(cache),
        )

    def precompute_parent_data(self, x_parent_standardized: np.ndarray) -> ParentTermCache:
        """Precompute corrected-parent bases directly on parent-standardized data."""
        samples = ensure_2d(x_parent_standardized, expected_dim=self.input_dim)
        quad_parent_standardized, _ = self.correction._quad_inputs(samples)
        base_sample_non_basis = self.base_parent.component.evaluate_nonmonotone_coeff_gradient(samples)
        base_sample_pre_basis = self.base_parent.component.evaluate_pre_monotone_coeff_gradient(samples)
        base_quad_pre_basis = self.base_parent.component.evaluate_pre_monotone_coeff_gradient(
            quad_parent_standardized.reshape(-1, self.input_dim)
        ).reshape(quad_parent_standardized.shape[0], quad_parent_standardized.shape[1], -1)
        correction_sample_non_basis = self.correction.evaluate_nonmonotone_coeff_gradient(samples)
        correction_sample_pre_basis = self.correction.evaluate_pre_monotone_coeff_gradient(samples)
        correction_quad_pre_basis = self.correction.evaluate_pre_monotone_coeff_gradient(
            quad_parent_standardized.reshape(-1, self.input_dim)
        ).reshape(quad_parent_standardized.shape[0], quad_parent_standardized.shape[1], -1)
        return ParentTermCache(
            sample_inputs=samples,
            quad_inputs=quad_parent_standardized,
            base_sample_non_basis=base_sample_non_basis,
            base_sample_pre_basis=base_sample_pre_basis,
            base_quad_pre_basis=base_quad_pre_basis,
            trainable_sample_non_grad=correction_sample_non_basis,
            trainable_sample_pre_grad=correction_sample_pre_basis,
            trainable_quad_pre_grad=correction_quad_pre_basis,
            correction_sample_non_basis=correction_sample_non_basis,
            correction_sample_pre_basis=correction_sample_pre_basis,
            correction_quad_pre_basis=correction_quad_pre_basis,
        )

    def non_value_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate corrected nonmonotone parent values on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        return self.base_parent.non_value_from_hf(samples) + self.correction.evaluate_nonmonotone(samples)

    def pre_value_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate corrected pre-monotone parent values on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        return self.base_parent.pre_value_from_hf(samples) + self.correction.evaluate_pre_monotone(samples)

    def non_coeff_grad_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate corrected nonmonotone coefficient gradients on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        return self.correction.evaluate_nonmonotone_coeff_gradient(samples)

    def pre_coeff_grad_from_hf(self, x_hf_standardized: np.ndarray) -> np.ndarray:
        """Evaluate corrected pre-monotone coefficient gradients on HF-standardized inputs."""
        samples = ensure_2d(x_hf_standardized, expected_dim=self.input_dim)
        return self.correction.evaluate_pre_monotone_coeff_gradient(samples)

    def non_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate corrected nonmonotone values from cached bases."""
        if cache.correction_sample_non_basis is None:
            raise ValueError("Correction cache is missing nonmonotone basis evaluations.")
        return (
            self.base_parent.non_value_from_cache(cache)
            + cache.correction_sample_non_basis @ self.correction.coeffs
        )

    def pre_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate corrected pre-monotone values from cached bases."""
        if cache.correction_sample_pre_basis is None:
            raise ValueError("Correction cache is missing pre-monotone basis evaluations.")
        return (
            self.base_parent.pre_value_from_cache(cache)
            + cache.correction_sample_pre_basis @ self.correction.coeffs
        )

    def pre_quad_value_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate corrected quadrature pre-monotone values from cache."""
        if cache.correction_quad_pre_basis is None:
            raise ValueError("Correction cache is missing quadrature pre-monotone basis evaluations.")
        return (
            self.base_parent.pre_quad_value_from_cache(cache)
            + np.einsum("qnm,m->qn", cache.correction_quad_pre_basis, self.correction.coeffs)
        )

    def non_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached correction gradients for nonmonotone terms."""
        return cache.trainable_sample_non_grad

    def pre_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached correction gradients for pre-monotone terms."""
        return cache.trainable_sample_pre_grad

    def pre_quad_coeff_grad_from_cache(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate cached quadrature pre-monotone coefficient gradients."""
        return cache.trainable_quad_pre_grad

    def objective_on_parent_standardized(self, coeffs: np.ndarray, x_parent_standardized: np.ndarray) -> float:
        """Evaluate the corrected-parent objective without precomputation."""
        self.set_coeffs(coeffs)
        cache = self.precompute_parent_data(x_parent_standardized)
        values = self.evaluate_on_parent_standardized_precomp(cache)
        derivative = self.derivative_on_parent_standardized_precomp(cache)
        return float(np.mean(0.5 * values * values - np.log(derivative)))

    def gradient_on_parent_standardized(self, coeffs: np.ndarray, x_parent_standardized: np.ndarray) -> np.ndarray:
        """Evaluate the corrected-parent gradient without precomputation."""
        self.set_coeffs(coeffs)
        cache = self.precompute_parent_data(x_parent_standardized)
        values = self.evaluate_on_parent_standardized_precomp(cache)
        derivative = self.derivative_on_parent_standardized_precomp(cache)
        coeff_grad = self.coeff_gradient_on_parent_standardized_precomp(cache)
        deriv_coeff_grad = self.deriv_coeff_gradient_on_parent_standardized_precomp(cache)
        return np.mean(values[:, None] * coeff_grad - (1.0 / derivative)[:, None] * deriv_coeff_grad, axis=0)

    def coeff_gradient_on_parent_standardized_precomp(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate corrected-parent coefficient gradients from cache."""
        non_grad = self.non_coeff_grad_from_cache(cache)
        quad_corr_eval = np.einsum("qnm,m->qn", cache.correction_quad_pre_basis, self.correction.coeffs)
        base_quad_eval = self.base_parent.pre_quad_value_from_cache(cache)
        rectifier_prime = self.rectifier.evaluate_derivative(base_quad_eval + quad_corr_eval)
        mono_grad = 0.5 * cache.sample_inputs[:, -1][:, None] * np.einsum(
            "q,qn,qnm->nm",
            self.correction.quadrature_rule.weights,
            rectifier_prime,
            cache.correction_quad_pre_basis,
        )
        return non_grad + mono_grad

    def deriv_coeff_gradient_on_parent_standardized_precomp(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate corrected-parent derivative gradients from cache."""
        rectifier_prime = self.rectifier.evaluate_derivative(self.pre_value_from_cache(cache))
        return rectifier_prime[:, None] * cache.correction_sample_pre_basis

    def objective_on_parent_standardized_precomp(self, coeffs: np.ndarray, cache: ParentTermCache) -> float:
        """Evaluate the corrected-parent objective from cached bases."""
        self.set_coeffs(coeffs)
        values = self.evaluate_on_parent_standardized_precomp(cache)
        derivative = self.derivative_on_parent_standardized_precomp(cache)
        return float(np.mean(0.5 * values * values - np.log(derivative)))

    def derivative_on_parent_standardized_precomp(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate the corrected-parent derivative from cache."""
        return self.rectifier.evaluate(self.pre_value_from_cache(cache)) + self.rectifier_epsilon

    def gradient_on_parent_standardized_precomp(self, coeffs: np.ndarray, cache: ParentTermCache) -> np.ndarray:
        """Evaluate the corrected-parent gradient from cached bases."""
        self.set_coeffs(coeffs)
        values = self.evaluate_on_parent_standardized_precomp(cache)
        derivative = self.derivative_on_parent_standardized_precomp(cache)
        coeff_grad = self.coeff_gradient_on_parent_standardized_precomp(cache)
        deriv_coeff_grad = self.deriv_coeff_gradient_on_parent_standardized_precomp(cache)
        return np.mean(values[:, None] * coeff_grad - (1.0 / derivative)[:, None] * deriv_coeff_grad, axis=0)

    def evaluate_on_parent_standardized_precomp(self, cache: ParentTermCache) -> np.ndarray:
        """Evaluate the corrected parent map from cached bases."""
        rectified = self.rectifier.evaluate(self.pre_quad_value_from_cache(cache)) + self.rectifier_epsilon
        monotone = 0.5 * cache.sample_inputs[:, -1] * np.einsum(
            "q,qn->n",
            self.correction.quadrature_rule.weights,
            rectified,
        )
        return self.non_value_from_cache(cache) + monotone

    def objective_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> float:
        """Evaluate the corrected-parent objective on parent-standardized data."""
        if cache is None:
            return self.objective_on_parent_standardized(coeffs, x_parent_standardized)
        return self.objective_on_parent_standardized_precomp(coeffs, cache)

    def gradient_on_parent_data(
        self,
        coeffs: np.ndarray,
        x_parent_standardized: np.ndarray,
        cache: object | None = None,
    ) -> np.ndarray:
        """Evaluate the corrected-parent gradient on parent-standardized data."""
        if cache is None:
            return self.gradient_on_parent_standardized(coeffs, x_parent_standardized)
        return self.gradient_on_parent_standardized_precomp(coeffs, cache)
