"""Define structured coefficient containers for hierarchical maps."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..single_fidelity.coefficients import MapCoefficients


@dataclass
class HierarchicalMapCoefficients:
    """Expose the stagewise coefficient structure of a hierarchical map."""

    stages: list[MapCoefficients]

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view of all trained stage coefficients."""
        if not self.stages:
            return np.zeros(0, dtype=float)
        return np.concatenate([stage.flatten() for stage in self.stages])

    def shapes(self) -> list[list[dict[str, dict[str, tuple[int, ...]]]]]:
        """Return the nested shape summary for each trained stage."""
        return [stage.shapes() for stage in self.stages]
