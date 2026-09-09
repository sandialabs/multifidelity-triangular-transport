"""Define parameter containers for peer non-hierarchical multifidelity maps."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..single_fidelity import OptimizationParams, QuadratureRule, MapParams


@dataclass
class NonHierarchicalMapParams:
    """Collect basis orders and optimizer settings for the NHMF construction.

    Parameters
    ----------
    low_fidelity_map_params : list of MapParams
        One parent-map configuration for each dataset after high-fidelity
        ``X_0``.
    shift_order, scale_order, correction_order : int, default=0
        Total orders for the peer shift, monotone scale, and trainable
        corrections to pretrained low-fidelity parent maps.
    optimization : OptimizationParams, optional
        Optimizer settings for the coupled high-fidelity fit.
    quadrature_rule : QuadratureRule, optional
        Quadrature used by monotone peer components.
    sigma : float, default=30.0
        Hermite normalization scale for peer expansions.
    rectifier_epsilon : float, default=1e-4
        Positive floor used by the monotonicity rectifier.
    hf_weight : float, default=1.0
        Positive weight on the high-fidelity contribution to the objective.
    regularize_parent_terms : bool, default=True
        Whether parent-derived terms contribute to coefficient regularization.
    """
    low_fidelity_map_params: list[MapParams] = field(default_factory=list)
    shift_order: int = 0
    scale_order: int = 0
    correction_order: int = 0
    optimization: OptimizationParams = field(default_factory=OptimizationParams)
    quadrature_rule: QuadratureRule = field(default_factory=QuadratureRule.legendre)
    sigma: float = 30.0
    rectifier_epsilon: float = 1e-4
    hf_weight: float = 1.0
    regularize_parent_terms: bool = True

    def __post_init__(self) -> None:
        """Validate parent-map dimensions and NHMF order settings."""
        if len(self.low_fidelity_map_params) < 1:
            raise ValueError("At least one low-fidelity MapParams object is required.")
        if self.shift_order < 0 or self.scale_order < 0 or self.correction_order < 0:
            raise ValueError("All NHMF total orders must be non-negative.")
        if self.hf_weight <= 0.0:
            raise ValueError("hf_weight must be positive.")
