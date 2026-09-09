"""Define structured coefficient containers for NHMF maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


class FlattenableCoefficients(Protocol):
    """Describe the minimal interface used by nested coefficient containers."""

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view of the coefficients."""

    def set_from_flat(self, flat: np.ndarray) -> None:
        """Update the coefficients from a flat optimizer vector."""

    def shapes(self):
        """Return a lightweight structural summary of coefficient shapes."""


@dataclass
class NHMFComponentCoefficients:
    """Expose the structured coefficient blocks of one NHMF component."""

    shift: FlattenableCoefficients
    scales: list[FlattenableCoefficients]
    parents: list[FlattenableCoefficients]

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view following the visible block order."""
        groups = [self.shift.flatten()]
        groups.extend(scale.flatten() for scale in self.scales)
        groups.extend(parent.flatten() for parent in self.parents)
        if not groups:
            return np.zeros(0, dtype=float)
        return np.concatenate(groups)

    def set_from_flat(self, flat: np.ndarray) -> None:
        """Restore all coefficient blocks from a flat optimizer vector."""
        coeffs = np.asarray(flat, dtype=float).reshape(-1)
        start = 0
        for block in [self.shift, *self.scales, *self.parents]:
            block_size = block.flatten().size
            stop = start + block_size
            block.set_from_flat(coeffs[start:stop])
            start = stop
        if start != coeffs.size:
            raise ValueError("Coefficient vector length does not match NHMF component structure.")

    def shapes(self) -> dict[str, object]:
        """Return nested array shapes for interactive inspection."""
        return {
            "shift": self.shift.shapes(),
            "scales": [scale.shapes() for scale in self.scales],
            "parents": [parent.shapes() for parent in self.parents],
        }


@dataclass
class NHMFMapCoefficients:
    """Expose the per-component coefficient structure of an NHMF map."""

    components: list[NHMFComponentCoefficients]

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view of all coupled component coefficients."""
        if not self.components:
            return np.zeros(0, dtype=float)
        return np.concatenate([component.flatten() for component in self.components])

    def set_from_flat(self, flat: np.ndarray) -> None:
        """Update all component coefficients from a flat vector."""
        coeffs = np.asarray(flat, dtype=float).reshape(-1)
        start = 0
        for component in self.components:
            stop = start + component.flatten().size
            component.set_from_flat(coeffs[start:stop])
            start = stop
        if start != coeffs.size:
            raise ValueError("Coefficient vector length does not match NHMF map structure.")

    def shapes(self) -> list[dict[str, object]]:
        """Return the nested shape summary for each coupled component."""
        return [component.shapes() for component in self.components]


@dataclass
class NHMFGradientBlocks:
    """Store structured NHMF gradients in the same order as the coefficients."""

    shift: np.ndarray
    scales: list[np.ndarray]
    parents: list[np.ndarray]

    def flatten(self) -> np.ndarray:
        """Flatten the gradient blocks in coefficient order."""
        groups = [self.shift, *self.scales, *self.parents]
        if not groups:
            return np.zeros(0, dtype=float)
        return np.concatenate(groups)

    @classmethod
    def zeros_like(cls, coefficients: NHMFComponentCoefficients) -> "NHMFGradientBlocks":
        """Create zero-valued blocks with the same visible structure."""
        return cls(
            shift=np.zeros_like(coefficients.shift.flatten()),
            scales=[np.zeros_like(scale.flatten()) for scale in coefficients.scales],
            parents=[np.zeros_like(parent.flatten()) for parent in coefficients.parents],
        )

    def add_scaled(self, other: "NHMFGradientBlocks", scale: float = 1.0) -> None:
        """Accumulate another block structure with a scalar weight."""
        self.shift += scale * other.shift
        for target, source in zip(self.scales, other.scales):
            target += scale * source
        for target, source in zip(self.parents, other.parents):
            target += scale * source
