"""Implement shared-parameterized single-fidelity map components."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import minimize
from scipy.special import expit

from .basis import evaluate_multiindex_derivative_matrix, evaluate_multiindex_matrix
from .coefficients import ComponentCoefficients, ExpansionCoefficients
from .params import BasisSpec, MapComponentParams, OptimizationParams
from .utils import summarize_minimize_result


class SoftPlusRectifier:
    """Provide the monotone rectifier used in triangular components."""

    def __init__(self, epsilon: float = 1e-4) -> None:
        self.epsilon = float(epsilon)

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        return np.logaddexp(0.0, np.asarray(x, dtype=float))

    def inverse(self, x: np.ndarray) -> np.ndarray:
        return np.log(np.expm1(np.asarray(x, dtype=float)))

    def evaluate_derivative(self, x: np.ndarray) -> np.ndarray:
        return expit(np.asarray(x, dtype=float))


@dataclass
class ComponentCache:
    """Store basis evaluations for one component and dataset."""

    x: np.ndarray
    non_basis_eval: np.ndarray
    pre_basis_eval: np.ndarray
    quad_pre_basis: np.ndarray
    upper_bounds: np.ndarray


class HermiteExpansion:
    """Represent one Hermite basis and its shared coefficient vector."""

    def __init__(self, basis_spec: BasisSpec) -> None:
        self.basis_spec = basis_spec
        self.multi_indices = list(basis_spec.multi_indices)
        self.normalization = np.asarray(basis_spec.normalization, dtype=float)
        self.coeffs = np.zeros(len(self.multi_indices), dtype=float)

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        coeffs = np.asarray(coeffs, dtype=float).reshape(-1)
        if coeffs.size != len(self.multi_indices):
            raise ValueError("Coefficient vector length does not match expansion shape.")
        self.coeffs = coeffs.copy()

    def evaluate_basis(self, x: np.ndarray) -> np.ndarray:
        return evaluate_multiindex_matrix(
            np.asarray(x, dtype=float), self.multi_indices, self.normalization, self.basis_spec.sigma
        )

    def evaluate_basis_derivative(self, x: np.ndarray, axis: int) -> np.ndarray:
        return evaluate_multiindex_derivative_matrix(
            np.asarray(x, dtype=float), self.multi_indices, self.normalization, self.basis_spec.sigma, axis
        )

    def initialize_zero(self) -> None:
        self.coeffs[:] = 0.0


class MapComponent:
    r"""Implement one projected-shared lower-triangular component.

    For component :math:`k`,

    .. math::

       T_k(x_{1:k}) = a_k(x_{1:k-1}) +
       \int_0^{x_k} \operatorname{softplus}(h_k(x_{1:k-1},t))\,dt
       + \varepsilon x_k.

    ``nonmonotone_mask`` stores the coefficients of :math:`a_k`; the
    complementary ``pre_monotone_mask`` stores :math:`h_k`.  Quadrature is
    needed only for :math:`T_k`; its diagonal derivative is the direct
    rectified value ``softplus(h_k) + epsilon``.
    """

    def __init__(self, params: MapComponentParams) -> None:
        self.params = params
        self.input_dim = params.input_dim
        self.rectifier = SoftPlusRectifier(epsilon=params.rectifier_epsilon)
        self.expansion = HermiteExpansion(params.basis_spec)
        self.nonmonotone_mask = np.array(
            [multi_index[-1] == 0 for multi_index in self.expansion.multi_indices], dtype=bool
        )
        self.pre_monotone_mask = ~self.nonmonotone_mask
        if params.initialization != "identity":
            raise ValueError("Only identity initialization is supported.")
        self._initialize_identity()
        self.last_optimization_result: dict[str, object] | None = None

    @property
    def coeffs(self) -> np.ndarray:
        return self.coefficients.flatten()

    @property
    def coefficients(self) -> ComponentCoefficients:
        return ComponentCoefficients(expansion=ExpansionCoefficients(shared=self.expansion.coeffs))

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        self.coefficients.set_from_flat(coeffs)

    def _initialize_identity(self) -> None:
        self.expansion.initialize_zero()
        linear_seed = (0,) * (self.input_dim - 1) + (1,)
        idx = self.expansion.multi_indices.index(linear_seed)
        constant = self.rectifier.inverse(np.array([1.0 - self.rectifier.epsilon]))[0]
        derivative_basis = self.expansion.evaluate_basis_derivative(
            np.zeros((1, self.input_dim), dtype=float), self.input_dim - 1
        )[0, idx]
        if derivative_basis == 0.0:
            raise ValueError("Shared linear seed has zero derivative under the configured basis.")
        self.expansion.coeffs[idx] = constant / derivative_basis

    def _masked_basis(
        self, x: np.ndarray, mask: np.ndarray, *, derivative_axis: int | None = None
    ) -> np.ndarray:
        if derivative_axis is None:
            basis = self.expansion.evaluate_basis(np.asarray(x, dtype=float))
        else:
            basis = self.expansion.evaluate_basis_derivative(np.asarray(x, dtype=float), derivative_axis)
        basis[:, ~mask] = 0.0
        return basis

    def _non_basis(self, x: np.ndarray) -> np.ndarray:
        return self._masked_basis(x, self.nonmonotone_mask)

    def _pre_basis(self, x: np.ndarray) -> np.ndarray:
        return self._masked_basis(x, self.pre_monotone_mask, derivative_axis=self.input_dim - 1)

    def precompute(self, x: np.ndarray) -> ComponentCache:
        samples = np.asarray(x, dtype=float)
        non_basis_eval = self._non_basis(samples)
        pre_basis_eval = self._pre_basis(samples)
        upper_bounds = samples[:, -1]
        quad_points = 0.5 * upper_bounds[None, :] * (self.params.quadrature_rule.points[:, None] + 1.0)
        quad_inputs = np.broadcast_to(
            samples[None, :, :],
            (self.params.quadrature_rule.num_points, samples.shape[0], self.input_dim),
        ).copy()
        quad_inputs[:, :, -1] = quad_points
        quad_pre_basis = self._pre_basis(quad_inputs.reshape(-1, self.input_dim))
        quad_pre_basis = quad_pre_basis.reshape(self.params.quadrature_rule.num_points, samples.shape[0], -1)
        return ComponentCache(samples, non_basis_eval, pre_basis_eval, quad_pre_basis, upper_bounds)

    def evaluate_nonmonotone(self, x: np.ndarray) -> np.ndarray:
        return self._non_basis(x) @ self.expansion.coeffs

    def evaluate_pre_monotone(self, x: np.ndarray) -> np.ndarray:
        return self._pre_basis(x) @ self.expansion.coeffs

    def evaluate_nonmonotone_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._non_basis(x)

    def evaluate_pre_monotone_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._pre_basis(x)

    def _sample_pre_monotone_eval_from_cache(self, cache: ComponentCache) -> np.ndarray:
        return cache.pre_basis_eval @ self.expansion.coeffs

    def _derivative_eval_from_cache(self, cache: ComponentCache) -> np.ndarray:
        return self.rectifier.evaluate(self._sample_pre_monotone_eval_from_cache(cache)) + self.rectifier.epsilon

    def _quad_pre_monotone_eval_from_cache(self, cache: ComponentCache) -> np.ndarray:
        return np.einsum("qnm,m->qn", cache.quad_pre_basis, self.expansion.coeffs)

    def _monotone_from_quad_eval(self, cache: ComponentCache, quad_eval: np.ndarray) -> np.ndarray:
        rectified = self.rectifier.evaluate(quad_eval) + self.rectifier.epsilon
        return 0.5 * cache.upper_bounds * np.einsum("q,qn->n", self.params.quadrature_rule.weights, rectified)

    def evaluate(self, x: np.ndarray, cache: ComponentCache | None = None) -> np.ndarray:
        local_cache = self.precompute(x) if cache is None else cache
        nonmonotone = local_cache.non_basis_eval @ self.expansion.coeffs
        monotone = self._monotone_from_quad_eval(
            local_cache, self._quad_pre_monotone_eval_from_cache(local_cache)
        )
        return nonmonotone + monotone

    def evaluate_derivative_xk(self, x: np.ndarray, cache: ComponentCache | None = None) -> np.ndarray:
        if cache is None:
            return self.rectifier.evaluate(self.evaluate_pre_monotone(x)) + self.rectifier.epsilon
        return self._derivative_eval_from_cache(cache)

    def evaluate_partial_derivative(
        self, x: np.ndarray, axis: int, cache: ComponentCache | None = None
    ) -> np.ndarray:
        if axis < 0 or axis >= self.input_dim:
            raise ValueError(f"axis must be in [0, {self.input_dim}), received {axis}.")
        local_cache = self.precompute(x) if cache is None else cache
        if axis == self.input_dim - 1:
            return self.evaluate_derivative_xk(local_cache.x, cache=local_cache)
        samples = local_cache.x
        step = 1e-5 * np.maximum(1.0, np.abs(samples[:, axis]))
        plus, minus = samples.copy(), samples.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        return (self.evaluate(plus) - self.evaluate(minus)) / (2.0 * step)

    def evaluate_mixed_derivative_xk(
        self, x: np.ndarray, axis: int, cache: ComponentCache | None = None
    ) -> np.ndarray:
        if axis < 0 or axis >= self.input_dim:
            raise ValueError(f"axis must be in [0, {self.input_dim}), received {axis}.")
        samples = np.asarray(x, dtype=float)
        step = 1e-5 * np.maximum(1.0, np.abs(samples[:, axis]))
        plus, minus = samples.copy(), samples.copy()
        plus[:, axis] += step
        minus[:, axis] -= step
        return (self.evaluate_derivative_xk(plus) - self.evaluate_derivative_xk(minus)) / (2.0 * step)

    def evaluate_coeff_gradient(self, x: np.ndarray, cache: ComponentCache | None = None) -> np.ndarray:
        local_cache = self.precompute(x) if cache is None else cache
        quad_eval = self._quad_pre_monotone_eval_from_cache(local_cache)
        rectifier_prime = self.rectifier.evaluate_derivative(quad_eval)
        mono_grad = 0.5 * local_cache.upper_bounds[:, None] * np.einsum(
            "q,qn,qnm->nm",
            self.params.quadrature_rule.weights,
            rectifier_prime,
            local_cache.quad_pre_basis,
        )
        return local_cache.non_basis_eval + mono_grad

    def evaluate_derivative_xk_coeff_gradient(
        self, x: np.ndarray, cache: ComponentCache | None = None
    ) -> np.ndarray:
        local_cache = self.precompute(x) if cache is None else cache
        rectifier_prime = self.rectifier.evaluate_derivative(
            self._sample_pre_monotone_eval_from_cache(local_cache)
        )
        return rectifier_prime[:, None] * local_cache.pre_basis_eval

    def objective(
        self, coeffs: np.ndarray, x: np.ndarray, reg_cst: float = 0.0, cache: ComponentCache | None = None
    ) -> float:
        self.set_coeffs(coeffs)
        local_cache = self.precompute(x) if cache is None else cache
        values = self.evaluate(local_cache.x, cache=local_cache)
        derivative = self.evaluate_derivative_xk(local_cache.x, cache=local_cache)
        return float(np.mean(0.5 * values * values - np.log(derivative)) + reg_cst * np.dot(coeffs, coeffs))

    def gradient(
        self, coeffs: np.ndarray, x: np.ndarray, reg_cst: float = 0.0, cache: ComponentCache | None = None
    ) -> np.ndarray:
        self.set_coeffs(coeffs)
        local_cache = self.precompute(x) if cache is None else cache
        values = self.evaluate(local_cache.x, cache=local_cache)
        derivative = self.evaluate_derivative_xk(local_cache.x, cache=local_cache)
        coeff_grad = self.evaluate_coeff_gradient(local_cache.x, cache=local_cache)
        deriv_coeff_grad = self.evaluate_derivative_xk_coeff_gradient(local_cache.x, cache=local_cache)
        total = np.mean(values[:, None] * coeff_grad - deriv_coeff_grad / derivative[:, None], axis=0)
        return total + 2.0 * reg_cst * coeffs

    def train(self, x: np.ndarray, optimization: OptimizationParams, cache: ComponentCache | None = None) -> None:
        local_cache = self.precompute(x) if cache is None else cache
        result = minimize(
            self.objective,
            self.coeffs,
            args=(local_cache.x, optimization.reg_cst, local_cache),
            jac=self.gradient,
            method=optimization.optimizer,
            options={"gtol": optimization.gtol, "maxiter": optimization.maxiter},
        )
        self.set_coeffs(result.x)
        self.last_optimization_result = summarize_minimize_result(result, optimization)
