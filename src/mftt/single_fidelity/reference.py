"""Define reference-density adapters used by transport maps."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy.stats import multivariate_normal


@dataclass
class Reference:
    """Wrap reference log-density and score evaluations.

    Parameters
    ----------
    logpdf_fn : callable
        Function accepting a batch with shape ``(n_samples, input_dim)`` and
        returning one log-density value per row.
    score_fn : callable, optional
        Function on the same batch returning the gradient of the log density
        with shape ``(n_samples, input_dim)``. It is required for analytic AAO
        gradients.
    """
    logpdf_fn: Callable[[np.ndarray], np.ndarray]
    score_fn: Callable[[np.ndarray], np.ndarray] | None = None

    @classmethod
    def standard_normal(cls, input_dim: int) -> "Reference":
        """Build a standard normal reference.

        Parameters
        ----------
        input_dim : int
            Dimension of the reference coordinates.

        Returns
        -------
        Reference
            Standard normal reference with identity covariance.
        """
        def logpdf(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            return -0.5 * np.sum(x * x, axis=1) - 0.5 * input_dim * np.log(2.0 * np.pi)

        def score(x: np.ndarray) -> np.ndarray:
            return -np.asarray(x, dtype=float)

        return cls(logpdf_fn=logpdf, score_fn=score)

    @classmethod
    def from_scipy(cls, executable) -> "Reference":
        """Build a reference wrapper from a frozen SciPy distribution.

        Parameters
        ----------
        executable : scipy.stats frozen distribution
            Distribution exposing ``logpdf``. A distribution exposing ``mean``
            and ``cov`` also receives an analytic score function.

        Returns
        -------
        Reference
            Adapter using the distribution's log density and, when available,
            its Gaussian score.

        Raises
        ------
        TypeError
            If the unfrozen SciPy ``multivariate_normal`` constructor is used.
        """
        if isinstance(executable, multivariate_normal.__class__):
            raise TypeError("Pass a frozen scipy distribution instance, not the multivariate_normal constructor.")

        if hasattr(executable, "mean") and hasattr(executable, "cov"):
            mean = np.asarray(executable.mean, dtype=float)
            cov = np.asarray(executable.cov, dtype=float)
            cov_inv = np.linalg.inv(cov)

            def score(x: np.ndarray) -> np.ndarray:
                delta = np.asarray(x, dtype=float) - mean
                return -(delta @ cov_inv.T)

            return cls(logpdf_fn=executable.logpdf, score_fn=score)
        return cls(logpdf_fn=executable.logpdf)

    def evaluate_logpdf(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the reference log-density on a batch of reference samples.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples,)
            Log-density values.
        """
        return np.asarray(self.logpdf_fn(np.asarray(x, dtype=float)), dtype=float)

    def evaluate_score(self, x: np.ndarray) -> np.ndarray:
        """Evaluate the reference score on a batch of reference samples.

        Parameters
        ----------
        x : array-like, shape (n_samples, input_dim)
            Reference-space coordinates.

        Returns
        -------
        numpy.ndarray, shape (n_samples, input_dim)
            Gradients of the reference log density.

        Raises
        ------
        ValueError
            If this reference was created without a score function.
        """
        if self.score_fn is None:
            raise ValueError("Reference score function is required for analytic-gradient AAO training.")
        return np.asarray(self.score_fn(np.asarray(x, dtype=float)), dtype=float)
