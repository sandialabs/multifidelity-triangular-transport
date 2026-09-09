"""Checks shared-formulation public signatures, identity behavior, and reference scores."""

import inspect

import numpy as np

import mftt as shared
from mftt.single_fidelity import MapParams, Reference, TriangularMap


def test_public_configuration_has_no_parameterization_selector():
    assert "parameterization" not in inspect.signature(shared.MapParams).parameters
    assert "parameterization" not in inspect.signature(shared.NonHierarchicalMapParams).parameters


def test_single_fidelity_public_api_has_one_dataset_and_internal_caching():
    map_signature = inspect.signature(TriangularMap)
    optimization_signature = inspect.signature(shared.OptimizationParams)
    assert list(map_signature.parameters) == ["train_data", "params", "reference", "standardization"]
    assert list(inspect.signature(shared.MapParams).parameters) == [
        "total_order",
        "component_params",
        "optimization",
    ]
    assert list(inspect.signature(shared.HierarchicalTriangularMap).parameters) == [
        "train_data",
        "hierarchical_params",
    ]
    assert list(inspect.signature(shared.NonHierarchicalTriangularMap).parameters) == [
        "train_data",
        "nhmf_params",
    ]
    assert "use_precomputed" not in optimization_signature.parameters
    for method in (
        TriangularMap.evaluate,
        TriangularMap.inverse,
        TriangularMap.train,
        TriangularMap.log_det,
        TriangularMap.pullback_logpdf,
    ):
        assert "data_idx" not in inspect.signature(method).parameters
    assert not any(parameter.kind is inspect.Parameter.VAR_POSITIONAL for parameter in map_signature.parameters.values())
    assert not any(parameter.kind is inspect.Parameter.VAR_KEYWORD for parameter in map_signature.parameters.values())
    assert list(inspect.signature(shared.HierarchicalMapParams).parameters) == [
        "fidelity_map_params",
    ]
    assert list(inspect.signature(shared.NonHierarchicalMapParams).parameters) == [
        "low_fidelity_map_params",
        "shift_order",
        "scale_order",
        "correction_order",
        "optimization",
        "quadrature_rule",
        "sigma",
        "rectifier_epsilon",
        "hf_weight",
        "regularize_parent_terms",
    ]
    assert not hasattr(TriangularMap, "objective_aao")
    assert not hasattr(TriangularMap, "log_det_aao")


def test_dimension_is_inferred_from_training_data():
    rng = np.random.default_rng(710)
    data = rng.normal(size=(24, 3))
    model = TriangularMap(data, MapParams(total_order=1))
    assert model.input_dim == 3
    assert model.evaluate(data[:4]).shape == (4, 3)


def test_identity_map_round_trip_uses_canonical_constructor():
    rng = np.random.default_rng(711)
    data = rng.normal(size=(24, 2))
    probe = rng.normal(size=(7, 2))
    model = TriangularMap(data, MapParams(total_order=1))
    assert np.allclose(model.inverse(model.evaluate(probe)), probe, atol=1e-8)
    assert isinstance(model.train_data, np.ndarray)
    assert isinstance(model.standardized_train_data, np.ndarray)
    assert model.standardization_params.mean.shape == (2,)
    assert model.precompute_training_data() is model.precompute_training_data()


def test_custom_reference_score_is_used_by_aao_gradient_path():
    rng = np.random.default_rng(712)
    data = rng.normal(size=(24, 2))
    reference = Reference(
        logpdf_fn=lambda x: -0.5 * np.sum((x - 0.25) ** 2, axis=1),
        score_fn=lambda x: -(x - 0.25),
    )
    model = TriangularMap(data, MapParams(total_order=1), reference=reference)
    coeffs = model.coeffs.copy()
    assert model.reference is reference
    assert np.all(np.isfinite(model.gradient_aao_precomp(coeffs)))
