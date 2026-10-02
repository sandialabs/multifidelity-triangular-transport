"""Multifidelity triangular transport maps.

The :mod:`mftt` package implements the single-fidelity, hierarchical, and
non-hierarchical constructions from the manuscript `Multifidelity
Formulations for Triangular Transport <https://arxiv.org/abs/2610.00698>`_
(arXiv:2610.00698, 2026).  The names listed in :data:`__all__`
are the stable user-facing API.
"""

__version__ = "0.1.0"

from .diagnostics import *  # noqa: F401,F403
from .single_fidelity import *  # noqa: F401,F403
from .hierarchical_multifidelity import *  # noqa: F401,F403
from .non_hierarchical_multifidelity import *  # noqa: F401,F403
from .single_fidelity.inversion import InverseDiagnostics, InverseFailure, InverseOptions, InverseResult

__all__ = [
    "__version__",
    "TriangularMap",
    "HierarchicalTriangularMap",
    "NonHierarchicalTriangularMap",
    "MapParams",
    "HierarchicalMapParams",
    "NonHierarchicalMapParams",
    "OptimizationParams",
    "Reference",
    "BasisSpec",
    "MapComponentParams",
    "QuadratureRule",
    "StandardizationParams",
    "InverseOptions",
    "InverseFailure",
    "InverseDiagnostics",
    "InverseResult",
    "first_moment_error",
    "second_moment_error",
    "forstner_distance",
    "squared_mmd_gaussian_kernel",
    "multiscale_squared_mmd_gaussian_kernel",
    "gaussian_diagnostics",
    "map_gaussian_diagnostics",
    "median_heuristic_bandwidth",
    "sample_mean",
    "sample_covariance",
]
