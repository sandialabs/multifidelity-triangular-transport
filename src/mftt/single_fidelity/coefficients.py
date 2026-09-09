"""Define structured coefficient containers for single-fidelity maps."""

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
class ExpansionCoefficients:
    """Expose the shared coefficient array used by one expansion."""

    shared: np.ndarray

    def flatten(self) -> np.ndarray:
        """Return a flat view of the shared coefficient array."""
        return self.shared.reshape(-1)

    def set_from_flat(self, flat: np.ndarray) -> None:
        """Update the shared coefficient array from a flat vector."""
        coeffs = np.asarray(flat, dtype=float).reshape(-1)
        if coeffs.size != self.shared.size:
            raise ValueError("Coefficient vector length does not match expansion structure.")
        self.shared[:] = coeffs

    def shapes(self) -> dict[str, tuple[int, ...]]:
        """Return the shared array shape for interactive inspection."""
        return {"shared": self.shared.shape}


@dataclass
class ComponentCoefficients:
    """Expose the coefficient structure of one triangular component."""

    expansion: ExpansionCoefficients

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view of the component coefficients."""
        return self.expansion.flatten()

    def set_from_flat(self, flat: np.ndarray) -> None:
        """Update the component coefficients from a flat vector."""
        self.expansion.set_from_flat(flat)

    def shapes(self) -> dict[str, dict[str, tuple[int, ...]]]:
        """Return the nested coefficient-array shapes for inspection."""
        return {"expansion": self.expansion.shapes()}


@dataclass
class MapCoefficients:
    """Expose the per-component coefficient structure of a triangular map."""

    components: list[ComponentCoefficients]

    def flatten(self) -> np.ndarray:
        """Return a flat optimizer view of all component coefficients."""
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
            raise ValueError("Coefficient vector length does not match map structure.")

    def shapes(self) -> list[dict[str, dict[str, tuple[int, ...]]]]:
        """Return the nested shape summary for each component."""
        return [component.shapes() for component in self.components]
