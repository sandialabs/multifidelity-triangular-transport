"""Checks analytic single-fidelity gradients and derivative evaluation fast paths."""

import numpy as np

from mftt.single_fidelity import MapComponentParams, MapParams, TriangularMap
from mftt.single_fidelity.params import BasisSpec
from mftt.single_fidelity.components import MapComponent
from tests._helpers import finite_difference_gradient


def test_component_gradient_matches_finite_difference():
    rng = np.random.default_rng(1)
    x = rng.normal(size=(12, 2))
    component = MapComponent(
        MapComponentParams(
            input_dim=2,
            basis_spec=BasisSpec(input_dim=2, total_order=2),
        )
    )
    cache = component.precompute(x)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    analytic = component.gradient(coeffs, x, reg_cst=1e-3, cache=cache)
    numeric = finite_difference_gradient(lambda c: component.objective(c, x, reg_cst=1e-3, cache=cache), coeffs)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_map_aao_gradient_matches_finite_difference():
    rng = np.random.default_rng(2)
    data = rng.normal(size=(20, 2))
    model = TriangularMap(train_data=data, params=MapParams(total_order=2))
    coeffs = model.coeffs + rng.normal(scale=1e-2, size=model.coeffs.shape)
    analytic = model.gradient_aao_precomp(coeffs, reg_cst=1e-3)
    numeric = finite_difference_gradient(lambda c: model.objective_aao_precomp(c, reg_cst=1e-3), coeffs)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_shared_order_zero_initializes_identity_like_map():
    rng = np.random.default_rng(23)
    data = rng.normal(size=(20, 3))
    model = TriangularMap(train_data=data,
        params=MapParams(total_order=0),
    )
    standardized = model.standardized_train_data[:5]
    assert model.total_order == 0
    assert np.allclose(model.evaluate_standardized(standardized), standardized)
    assert all(component.coefficients.shapes()["expansion"]["shared"] == (2,) for component in model.components)


def test_evaluate_derivative_xk_avoids_quadrature_helper(monkeypatch):
    rng = np.random.default_rng(3)
    x = rng.normal(size=(10, 2))
    component = MapComponent(
        MapComponentParams(
            input_dim=2,
            basis_spec=BasisSpec(input_dim=2, total_order=2),
        )
    )
    cache = component.precompute(x)
    called = {"quadrature": False}

    def fail_if_called(local_cache):
        called["quadrature"] = True
        raise AssertionError("Quadrature helper should not be used by evaluate_derivative_xk.")

    monkeypatch.setattr(component, "_quad_pre_monotone_eval_from_cache", fail_if_called)
    derivative = component.evaluate_derivative_xk(x, cache=cache)
    assert derivative.shape == (x.shape[0],)
    assert not called["quadrature"]


def test_uncached_derivative_avoids_full_precompute(monkeypatch):
    rng = np.random.default_rng(31)
    x = rng.normal(size=(10, 2))
    component = MapComponent(
        MapComponentParams(
            input_dim=2,
            basis_spec=BasisSpec(input_dim=2, total_order=2),
        )
    )
    component_from_map = TriangularMap(train_data=rng.normal(size=(20, 2)),
        params=MapParams(total_order=2),
    ).components[1]

    for component in (component, component_from_map):
        cache = component.precompute(x)
        expected = component.evaluate_derivative_xk(x, cache=cache)

        def fail_if_called(_):
            raise AssertionError("Uncached derivatives must not build quadrature caches.")

        monkeypatch.setattr(component, "precompute", fail_if_called)
        actual = component.evaluate_derivative_xk(x)
        assert np.allclose(actual, expected)
