"""Define parameter containers for single-fidelity transport maps."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np

from .basis import compute_normalization, generate_multi_indices


@dataclass
class StandardizationParams:
    """Store per-coordinate mean and standard deviation values.

    Parameters
    ----------
    mean : array-like, shape (input_dim,)
        Coordinate means in raw target coordinates.
    std : array-like, shape (input_dim,)
        Strictly positive coordinate standard deviations. Standardization is
        ``(x - mean) / std``; these values are fitted map state.
    """
    mean: np.ndarray
    std: np.ndarray

    def __post_init__(self) -> None:
        """Validate and normalize standardization arrays."""
        self.mean = np.asarray(self.mean, dtype=float).reshape(-1)
        self.std = np.asarray(self.std, dtype=float).reshape(-1)
        if self.mean.shape != self.std.shape:
            raise ValueError("mean and std must have the same shape.")
        if np.any(self.std <= 0.0):
            raise ValueError("All standard deviations must be positive.")


@dataclass
class BasisSpec:
    """Describe a Hermite basis through an order or explicit multi-indices.

    Parameters
    ----------
    input_dim : int
        Number of coordinates visible to this triangular component.
    total_order : int, optional
        Maximum total Hermite order. Required when ``multi_indices`` is not
        supplied.
    multi_indices : list of tuple of int, optional
        Explicit basis multi-indices, each of length ``input_dim``.
    sigma : float, default=30.0
        Hermite normalization scale.
    """
    input_dim: int
    total_order: int | None = None
    multi_indices: list[tuple[int, ...]] | None = None
    sigma: float = 30.0
    normalization: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        """Populate implied multi-indices and normalization factors."""
        if self.multi_indices is None:
            if self.total_order is None:
                raise ValueError("Either total_order or multi_indices must be provided.")
            self.multi_indices = generate_multi_indices(self.input_dim, self.total_order)
        else:
            self.multi_indices = [tuple(multi_index) for multi_index in self.multi_indices]
            for multi_index in self.multi_indices:
                if len(multi_index) != self.input_dim:
                    raise ValueError("Each multi-index must match BasisSpec.input_dim.")
            if self.total_order is None:
                self.total_order = max(sum(multi_index) for multi_index in self.multi_indices)
        self.normalization = compute_normalization(self.multi_indices, sigma=self.sigma)


@dataclass
class QuadratureRule:
    """Store quadrature nodes and weights used in monotone integrals.

    Parameters
    ----------
    points : array-like, shape (num_points,)
        Quadrature nodes on the reference interval.
    weights : array-like, shape (num_points,)
        Corresponding quadrature weights.
    num_points : int
        Number of nodes and weights.
    """
    points: np.ndarray
    weights: np.ndarray
    num_points: int

    @classmethod
    def legendre(cls, num_points: int = 10) -> "QuadratureRule":
        """Construct a Gauss-Legendre quadrature rule.

        Parameters
        ----------
        num_points : int, default=10
            Number of Gauss-Legendre nodes.

        Returns
        -------
        QuadratureRule
            Rule with nodes and weights on ``[-1, 1]``.
        """
        points, weights = np.polynomial.legendre.leggauss(num_points)
        return cls(points=np.asarray(points, dtype=float), weights=np.asarray(weights, dtype=float), num_points=num_points)


@dataclass
class OptimizationParams:
    """Collect optimizer settings shared across training routines.

    Parameters
    ----------
    optimizer : {"BFGS", "L-BFGS-B"}, default="BFGS"
        SciPy optimizer used for component objectives.
    reg_cst : float, default=0.0
        Nonnegative quadratic coefficient regularization constant.
    gtol : float, default=1e-6
        Gradient-norm convergence tolerance.
    maxiter : int, default=1000
        Maximum optimizer iterations.
    """
    optimizer: Literal["BFGS", "L-BFGS-B"] = "BFGS"
    reg_cst: float = 0.0
    gtol: float = 1e-6
    maxiter: int = 1000

@dataclass
class MapComponentParams:
    """Collect basis and quadrature settings for one triangular component.

    Parameters
    ----------
    input_dim : int
        Number of coordinates visible to component ``k``.
    basis_spec : BasisSpec
        Hermite basis; it must include the final-coordinate linear seed.
    quadrature_rule : QuadratureRule, optional
        Rule used by the monotone integral.
    rectifier_epsilon : float, default=1e-4
        Positive monotonicity-rectifier floor.
    initialization : str, default="identity"
        Initial coefficient construction. ``"identity"`` preserves the
        final-coordinate linear seed.
    """
    input_dim: int
    basis_spec: BasisSpec
    quadrature_rule: QuadratureRule = field(default_factory=QuadratureRule.legendre)
    rectifier_epsilon: float = 1e-4
    initialization: str = "identity"

    def __post_init__(self) -> None:
        """Validate the shared basis and identity seed."""
        if self.basis_spec.input_dim != self.input_dim:
            raise ValueError("basis_spec.input_dim must match component input_dim.")
        linear_seed = (0,) * (self.input_dim - 1) + (1,)
        if linear_seed not in self.basis_spec.multi_indices:
            raise ValueError("Shared component bases must include a final-coordinate linear seed.")


@dataclass
class MapParams:
    """Collect configuration for a full single-fidelity triangular map.

    Parameters
    ----------
    total_order : int, optional
        Total Hermite order used to generate every component basis. Supply
        this or ``component_params``.
    component_params : list of MapComponentParams, optional
        Explicit settings for components ``1`` through ``input_dim``. Supply
        this or ``total_order``.
    optimization : OptimizationParams, optional
        Optimizer settings shared by component training. A default instance is
        created when omitted.
    """
    total_order: int | None = None
    component_params: list[MapComponentParams] | None = None
    optimization: OptimizationParams | None = None

    def __post_init__(self) -> None:
        """Record the user-facing basis construction before binding to data."""
        # Record how the map was specified before ``bind`` expands a
        # total-order request into explicit component bases.  The resulting
        # bases alone cannot distinguish those two user-facing constructions.
        self._basis_construction = "explicit_multi_indices" if self.component_params is not None else "total_order"
        if self.optimization is None:
            self.optimization = OptimizationParams()

    def bind(self, input_dim: int) -> "MapParams":
        """Bind this configuration to a data dimension and build components.

        Parameters
        ----------
        input_dim : int
            Positive training-data coordinate dimension.

        Returns
        -------
        MapParams
            This configuration after validation and basis expansion.

        Raises
        ------
        ValueError
            If the dimension conflicts with existing settings or the basis
            specification is incomplete.
        """
        input_dim = int(input_dim)
        if input_dim < 1:
            raise ValueError("input_dim must be positive.")
        if self.component_params is None:
            if self.total_order is None:
                raise ValueError("total_order is required when component_params are not provided.")
            self.component_params = [
                _component_params_from_total_order(
                    k,
                    self.total_order,
                )
                for k in range(1, input_dim + 1)
            ]
        if len(self.component_params) != input_dim:
            raise ValueError("component_params length must match input_dim.")
        for expected_dim, component_params in enumerate(self.component_params, start=1):
            if component_params.input_dim != expected_dim:
                raise ValueError("Each component_params.input_dim must match its component dimension.")
        if self.total_order is None:
            self.total_order = max(
                component_params.basis_spec.total_order
                for component_params in self.component_params
            )
        return self


def _component_params_from_total_order(
    input_dim: int,
    total_order: int,
    sigma: float = 30.0,
    rectifier_epsilon: float = 1e-4,
    quadrature_rule: QuadratureRule | None = None,
    ) -> MapComponentParams:
    """Build one shared component's parameters from a total-order specification."""
    rule = quadrature_rule or QuadratureRule.legendre()
    shared_multi_indices = generate_multi_indices(input_dim, total_order)
    if total_order == 0:
        shared_multi_indices.append((0,) * (input_dim - 1) + (1,))
    return MapComponentParams(
        input_dim=input_dim,
        basis_spec=BasisSpec(
            input_dim=input_dim,
            multi_indices=shared_multi_indices,
            total_order=total_order,
            sigma=sigma,
        ),
        quadrature_rule=rule,
        rectifier_epsilon=rectifier_epsilon,
    )


def component_params_from_total_order(
    input_dim: int,
    total_order: int,
    sigma: float = 30.0,
    rectifier_epsilon: float = 1e-4,
    quadrature_rule: QuadratureRule | None = None,
) -> list[MapComponentParams]:
    """Build default component parameters from a shared total order."""
    return [
        _component_params_from_total_order(
            k,
            total_order,
            sigma=sigma,
            rectifier_epsilon=rectifier_epsilon,
            quadrature_rule=quadrature_rule,
        )
        for k in range(1, input_dim + 1)
    ]


def component_params_from_multi_index_sets(
    multi_index_sets: Sequence[Sequence[tuple[int, ...]]],
    sigma: float = 30.0,
    rectifier_epsilon: float = 1e-4,
    quadrature_rule: QuadratureRule | None = None,
) -> list[MapComponentParams]:
    """Build component parameters from explicit shared basis sets."""
    rule = quadrature_rule or QuadratureRule.legendre()
    component_params: list[MapComponentParams] = []
    for k, multi_indices in enumerate(
        multi_index_sets,
        start=1,
    ):
        component_params.append(
            MapComponentParams(
                input_dim=k,
                basis_spec=BasisSpec(input_dim=k, multi_indices=list(multi_indices), sigma=sigma),
                quadrature_rule=rule,
                rectifier_epsilon=rectifier_epsilon,
            )
        )
    return component_params
