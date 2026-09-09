"""Define parameter containers for hierarchical multifidelity maps."""

from __future__ import annotations

from dataclasses import dataclass, field

from ..single_fidelity import MapParams


@dataclass
class HierarchicalMapParams:
    """Collect one :class:`MapParams` configuration for each fidelity stage.

    Parameters
    ----------
    fidelity_map_params : list of MapParams
        One configuration per dataset in high-to-low order
        ``[X_0, ..., X_M]``. Training proceeds from the final entry toward
        high fidelity.
    """
    fidelity_map_params: list[MapParams] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Validate fidelity-map dimensional consistency."""
        if len(self.fidelity_map_params) < 1:
            raise ValueError("At least one fidelity map must be provided.")
