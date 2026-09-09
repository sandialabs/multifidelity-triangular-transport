"""Checks conditional pullback-density formulas across all map families."""

import numpy as np

from mftt import HierarchicalMapParams, HierarchicalTriangularMap, NonHierarchicalMapParams, NonHierarchicalTriangularMap
from mftt.single_fidelity import MapParams, OptimizationParams, Reference, TriangularMap


def _standard_normal_logpdf(z: np.ndarray) -> np.ndarray:
    samples = np.asarray(z, dtype=float)
    if samples.shape[1] == 0:
        return np.zeros(samples.shape[0], dtype=float)
    return -0.5 * np.sum(samples * samples, axis=1) - 0.5 * samples.shape[1] * np.log(2.0 * np.pi)


def _map_params(input_dim: int, total_order: int = 2, maxiter: int = 8) -> MapParams:
    return MapParams(
        total_order=total_order,
        optimization=OptimizationParams(maxiter=maxiter, gtol=1e-5, reg_cst=1e-3),
    )


def _nhmf_params(input_dim: int, total_order: int = 1, maxiter: int = 5) -> NonHierarchicalMapParams:
    return NonHierarchicalMapParams(
        low_fidelity_map_params=[_map_params(input_dim, total_order, maxiter) for _ in range(2)],
        shift_order=total_order,
        scale_order=total_order,
        correction_order=total_order,
        optimization=OptimizationParams(maxiter=maxiter, gtol=1e-5, reg_cst=1e-3),
    )


def _hmf_params(input_dim: int, total_order: int = 1, maxiter: int = 5) -> HierarchicalMapParams:
    return HierarchicalMapParams(
        fidelity_map_params=[_map_params(input_dim, total_order, maxiter) for _ in range(2)],
    )


def _assert_prefix_rejected(model) -> None:
    calls = [
        lambda: model.conditional_pullback_logpdf(
            np.zeros((1, model.input_dim)), fixed_indices=[1]
        ),
        lambda: model.conditional_sample(
            1, fixed_indices=[1], fixed_values=np.asarray([0.0]), random_state=0
        ),
        lambda: model.inverse(
            np.zeros((1, model.input_dim)),
            fixed_indices=[1],
            fixed_values=np.asarray([0.0]),
        ),
    ]
    for call in calls:
        try:
            call()
        except ValueError as exc:
            assert "leading prefix" in str(exc)
        else:
            raise AssertionError("Expected non-prefix fixed_indices to be rejected.")


def test_single_fidelity_conditional_pullback_logpdf_matches_formula_and_joint_identity():
    rng = np.random.default_rng(510)
    train_data = rng.normal(loc=[0.2, -0.3], scale=[1.4, 0.7], size=(30, 2))
    model = TriangularMap(train_data=train_data, params=_map_params(2, 2, maxiter=10))
    model.train()
    x = rng.normal(size=(5, 2))

    actual = model.conditional_pullback_logpdf(x, fixed_indices=[0])
    z = model.evaluate(x)
    expected = _standard_normal_logpdf(z[:, [1]]) + model.component_log_det(x, [1])
    prefix_marginal = _standard_normal_logpdf(z[:, [0]]) + model.component_log_det(x, [0])

    assert np.allclose(actual, expected)
    assert np.allclose(model.conditional_pullback_pdf(x, [0]), np.exp(actual))
    assert np.allclose(actual, model.pullback_logpdf(x) - prefix_marginal)
    assert np.allclose(model.conditional_pullback_logpdf(x, fixed_indices=[0, 1]), np.zeros(x.shape[0]))
    _assert_prefix_rejected(model)


def test_nhmf_conditional_pullback_logpdf_matches_formula():
    rng = np.random.default_rng(511)
    train_data = [
        rng.normal(loc=[0.3, -0.1], scale=[1.1, 0.9], size=(24, 2)),
        rng.normal(loc=[-0.2, 0.2], scale=[0.8, 1.2], size=(28, 2)),
        rng.normal(loc=[0.1, 0.4], scale=[1.0, 0.7], size=(32, 2)),
    ]
    model = NonHierarchicalTriangularMap(train_data=train_data, nhmf_params=_nhmf_params(2))
    model.train(use_corrections=True)
    x = rng.normal(size=(4, 2))

    actual = model.conditional_pullback_logpdf(x, fixed_indices=[0])
    z = model.evaluate(x)
    expected = _standard_normal_logpdf(z[:, [1]]) + model.component_log_det(x, [1])

    assert np.allclose(actual, expected)
    assert np.allclose(model.conditional_pullback_pdf(x, [0]), np.exp(actual))
    assert np.allclose(model.component_log_det(x, [0, 1]), model.log_det(x))
    _assert_prefix_rejected(model)


def test_hmf_selected_component_log_det_matches_finite_difference_and_conditional_formula():
    rng = np.random.default_rng(512)
    train_data = [
        rng.normal(loc=[0.2, -0.2], scale=[1.2, 0.9], size=(24, 2)),
        rng.normal(loc=[-0.1, 0.3], scale=[0.8, 1.1], size=(28, 2)),
    ]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=_hmf_params(2))
    model.train(method="fixed_reference")
    x = rng.normal(size=(4, 2))

    analytic = np.exp(model.component_log_det(x, [1]))
    step = 1e-5 * np.maximum(1.0, np.abs(x[:, 1]))
    plus = x.copy()
    minus = x.copy()
    plus[:, 1] += step
    minus[:, 1] -= step
    numeric = (model.evaluate(plus)[:, 1] - model.evaluate(minus)[:, 1]) / (2.0 * step)
    actual = model.conditional_pullback_logpdf(x, fixed_indices=[0])
    z = model.evaluate(x)
    expected = _standard_normal_logpdf(z[:, [1]]) + model.component_log_det(x, [1])

    assert np.allclose(analytic, numeric, rtol=1e-4, atol=1e-5)
    assert np.allclose(actual, expected)
    assert np.allclose(model.conditional_pullback_pdf(x, [0]), np.exp(actual))
    assert np.allclose(model.component_log_det(x, [0, 1]), model.log_det(x))
    _assert_prefix_rejected(model)
