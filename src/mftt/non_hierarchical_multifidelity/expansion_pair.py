"""Implement projected-shared NHMF expansion blocks."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..single_fidelity import BasisSpec, SoftPlusRectifier, generate_multi_indices
from ..single_fidelity.components import ExpansionCoefficients, HermiteExpansion
from ..single_fidelity.params import QuadratureRule
from ..single_fidelity.utils import ensure_2d


@dataclass
class ExpansionPairCache:
    """Store cached projected basis evaluations for one expansion."""

    sample_non_basis: np.ndarray
    sample_pre_basis: np.ndarray
    quad_pre_basis: np.ndarray
    sample_non_grad: np.ndarray
    sample_pre_grad: np.ndarray
    quad_pre_grad: np.ndarray


@dataclass
class ComponentExpansionPair:
    """Store one projected-shared Hermite expansion for an NHMF function block."""

    input_dim: int
    basis_spec: BasisSpec
    quadrature_rule: QuadratureRule
    rectifier_epsilon: float = 1e-4

    def __post_init__(self) -> None:
        self.expansion = HermiteExpansion(self.basis_spec)
        self.rectifier = SoftPlusRectifier(self.rectifier_epsilon)
        self.expansion.initialize_zero()
        self.nonmonotone_mask = np.array(
            [multi_index[-1] == 0 for multi_index in self.expansion.multi_indices], dtype=bool
        )
        self.pre_monotone_mask = ~self.nonmonotone_mask

    @property
    def coeffs(self) -> np.ndarray:
        return self.coefficients.flatten()

    @property
    def coefficients(self) -> ExpansionCoefficients:
        return ExpansionCoefficients(shared=self.expansion.coeffs)

    def set_coeffs(self, coeffs: np.ndarray) -> None:
        self.coefficients.set_from_flat(coeffs)

    def _masked_basis(
        self, x: np.ndarray, mask: np.ndarray, *, derivative_axis: int | None = None
    ) -> np.ndarray:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        if derivative_axis is None:
            basis = self.expansion.evaluate_basis(samples)
        else:
            basis = self.expansion.evaluate_basis_derivative(samples, derivative_axis)
        basis[:, ~mask] = 0.0
        return basis

    def _non_basis(self, x: np.ndarray) -> np.ndarray:
        return self._masked_basis(x, self.nonmonotone_mask)

    def _pre_basis(self, x: np.ndarray) -> np.ndarray:
        return self._masked_basis(x, self.pre_monotone_mask, derivative_axis=self.input_dim - 1)

    def initialize_constant(self, constant: float) -> None:
        self.expansion.initialize_zero()
        self._set_nonmonotone_constant(float(constant))
        self._set_pre_monotone_constant(float(constant))

    def _set_nonmonotone_constant(self, constant: float) -> None:
        self.expansion.coeffs[self._constant_index()] = constant

    def _set_pre_monotone_constant(self, constant: float) -> None:
        idx = self._linear_seed_index()
        derivative_basis = self._linear_seed_derivative()
        if derivative_basis == 0.0:
            raise ValueError("Shared linear seed has zero derivative under the configured basis.")
        self.expansion.coeffs[idx] = constant / derivative_basis

    def _linear_seed_derivative(self) -> float:
        idx = self._linear_seed_index()
        return float(
            self.expansion.evaluate_basis_derivative(
                np.zeros((1, self.input_dim), dtype=float), self.input_dim - 1
            )[0, idx]
        )

    def _constant_index(self) -> int:
        for idx, multi_index in enumerate(self.expansion.multi_indices):
            if all(order == 0 for order in multi_index):
                return idx
        raise ValueError("Constant initialization requires a constant basis term.")

    def _linear_seed_index(self) -> int:
        linear_seed = (0,) * (self.input_dim - 1) + (1,)
        try:
            return self.expansion.multi_indices.index(linear_seed)
        except ValueError as exc:
            raise ValueError("Shared constant initialization requires a final-coordinate linear seed.") from exc

    def constant_coeff_positions(self) -> tuple[int, int]:
        return self._constant_index(), self._linear_seed_index()

    def constant_value(self) -> float:
        return float(self.expansion.coeffs[self._constant_index()])

    def constant_gradient_weights(self) -> tuple[float, float]:
        return 1.0, 1.0 / self._linear_seed_derivative()

    def _quad_inputs(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        upper_bounds = samples[:, -1]
        quad_points = 0.5 * upper_bounds[None, :] * (self.quadrature_rule.points[:, None] + 1.0)
        quad_inputs = np.broadcast_to(
            samples[None, :, :],
            (self.quadrature_rule.num_points, samples.shape[0], self.input_dim),
        ).copy()
        quad_inputs[:, :, -1] = quad_points
        return quad_inputs, upper_bounds

    def evaluate_nonmonotone(self, x: np.ndarray) -> np.ndarray:
        return self._non_basis(x) @ self.expansion.coeffs

    def evaluate_pre_monotone(self, x: np.ndarray) -> np.ndarray:
        return self._pre_basis(x) @ self.expansion.coeffs

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        quad_inputs, upper_bounds = self._quad_inputs(samples)
        quad_basis = self._pre_basis(quad_inputs.reshape(-1, self.input_dim)).reshape(
            self.quadrature_rule.num_points, samples.shape[0], -1
        )
        quad_eval = np.einsum("qnm,m->qn", quad_basis, self.expansion.coeffs)
        rectified = self.rectifier.evaluate(quad_eval) + self.rectifier_epsilon
        monotone = 0.5 * upper_bounds * np.einsum("q,qn->n", self.quadrature_rule.weights, rectified)
        return self.evaluate_nonmonotone(samples) + monotone

    def evaluate_derivative_xk(self, x: np.ndarray) -> np.ndarray:
        return self.rectifier.evaluate(self.evaluate_pre_monotone(x)) + self.rectifier_epsilon

    def evaluate_nonmonotone_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._non_basis(x)

    def evaluate_pre_monotone_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        return self._pre_basis(x)

    def evaluate_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        non_grad = self._non_basis(samples)
        quad_inputs, upper_bounds = self._quad_inputs(samples)
        quad_basis = self._pre_basis(quad_inputs.reshape(-1, self.input_dim)).reshape(
            self.quadrature_rule.num_points, samples.shape[0], -1
        )
        quad_eval = np.einsum("qnm,m->qn", quad_basis, self.expansion.coeffs)
        rectifier_prime = self.rectifier.evaluate_derivative(quad_eval)
        mono_grad = 0.5 * upper_bounds[:, None] * np.einsum(
            "q,qn,qnm->nm", self.quadrature_rule.weights, rectifier_prime, quad_basis
        )
        return non_grad + mono_grad

    def evaluate_derivative_xk_coeff_gradient(self, x: np.ndarray) -> np.ndarray:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        rectifier_prime = self.rectifier.evaluate_derivative(self.evaluate_pre_monotone(samples))
        return rectifier_prime[:, None] * self._pre_basis(samples)

    def precompute(self, x: np.ndarray, quad_inputs: np.ndarray | None = None) -> ExpansionPairCache:
        samples = ensure_2d(x, expected_dim=self.input_dim)
        if quad_inputs is None:
            quad_inputs, _ = self._quad_inputs(samples)
        quad_samples = np.asarray(quad_inputs, dtype=float).reshape(-1, self.input_dim)
        sample_non_basis = self._non_basis(samples)
        sample_pre_basis = self._pre_basis(samples)
        quad_pre_basis = self._pre_basis(quad_samples).reshape(
            self.quadrature_rule.num_points, samples.shape[0], -1
        )
        return ExpansionPairCache(
            sample_non_basis=sample_non_basis,
            sample_pre_basis=sample_pre_basis,
            quad_pre_basis=quad_pre_basis,
            sample_non_grad=sample_non_basis,
            sample_pre_grad=sample_pre_basis,
            quad_pre_grad=quad_pre_basis,
        )


def build_component_expansion_pair(
    input_dim: int,
    total_order: int,
    quadrature_rule: QuadratureRule,
    sigma: float,
    rectifier_epsilon: float,
) -> ComponentExpansionPair:
    """Build one projected-shared NHMF expansion pair from a total order."""
    multi_indices = generate_multi_indices(input_dim, total_order)
    if total_order == 0:
        multi_indices.append((0,) * (input_dim - 1) + (1,))
    return ComponentExpansionPair(
        input_dim=input_dim,
        basis_spec=BasisSpec(
            input_dim=input_dim,
            multi_indices=multi_indices,
            total_order=total_order,
            sigma=sigma,
        ),
        quadrature_rule=quadrature_rule,
        rectifier_epsilon=rectifier_epsilon,
    )
