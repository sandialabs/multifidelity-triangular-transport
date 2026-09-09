"""Checks non-hierarchical-map training, gradients, initialization, and inversion."""

import numpy as np

from mftt import InverseOptions, InverseResult, NonHierarchicalMapParams, NonHierarchicalTriangularMap
from mftt.non_hierarchical_multifidelity.components import NonHierarchicalComponent
from mftt.single_fidelity import MapParams, OptimizationParams, Reference, TriangularMap
from tests._helpers import finite_difference_gradient


def _make_params(
    input_dim: int,
    total_order: int,
    num_parents: int,
    maxiter: int = 8,
) -> NonHierarchicalMapParams:
    return NonHierarchicalMapParams(
        low_fidelity_map_params=[
            MapParams(
                total_order=total_order,
                optimization=OptimizationParams(maxiter=maxiter, gtol=1e-5, reg_cst=1e-3),
            )
            for _ in range(num_parents)
        ],
        shift_order=total_order,
        scale_order=total_order,
        correction_order=total_order,
        optimization=OptimizationParams(maxiter=maxiter, gtol=1e-5, reg_cst=1e-3),
        hf_weight=1.0,
    )


def _make_data(rng: np.random.Generator) -> list[np.ndarray]:
    return [
        rng.normal(loc=0.3, scale=1.2, size=(18, 2)),
        rng.normal(loc=-0.2, scale=0.9, size=(20, 2)),
        rng.normal(loc=0.1, scale=0.8, size=(24, 2)),
    ]


def _constant_index(expansion) -> int:
    for idx, multi_index in enumerate(expansion.multi_indices):
        if all(order == 0 for order in multi_index):
            return idx
    raise AssertionError("Expected a constant basis term.")


def test_nhmf_constructs_and_generalizes_scale_terms():
    rng = np.random.default_rng(30)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2))
    model.train(use_corrections=False)
    assert len(model.low_fidelity_maps) == 2
    assert len(model.components) == 2
    assert model.components[1].scale_pairs[0].coeffs.size > 2


def test_nhmf_training_enables_corrections_by_default():
    rng = np.random.default_rng(91)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2))
    model.train()
    assert model._last_use_corrections is True
    assert "corrections=True" in str(model)


def test_nhmf_str_reports_parent_structure_before_and_after_training():
    rng = np.random.default_rng(90)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2))
    before = str(model)
    assert "NonHierarchicalTriangularMap" in before
    assert "family=non_hierarchical_multifidelity" in before
    assert "low_fidelity_parents=2" in before
    assert "corrections=untrained" in before

    model.train(use_corrections=True)
    after = str(model)
    assert "corrections=True" in after


def test_nhmf_weighting_matches_relative_sample_counts():
    rng = np.random.default_rng(31)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2))
    model.train(use_corrections=False)
    weights = model.components[1].low_fidelity_weights()
    expected = np.array([20.0, 24.0]) / 44.0
    assert np.allclose(weights, expected)


def test_nhmf_uncorrected_joint_step_updates_parent_coeffs():
    rng = np.random.default_rng(32)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=5))
    model.train(use_corrections=False)
    changed = False
    for low_map, pretrained in zip(model.low_fidelity_maps, model._pretrained_low_fidelity_component_coeffs):
        for component, old_coeffs in zip(low_map.components, pretrained):
            if not np.allclose(component.coeffs, old_coeffs):
                changed = True
    assert changed


def test_nhmf_corrected_mode_keeps_parent_coeffs_frozen():
    rng = np.random.default_rng(33)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=5))
    model.train(use_corrections=True)
    for low_map, pretrained in zip(model.low_fidelity_maps, model._pretrained_low_fidelity_component_coeffs):
        for component, old_coeffs in zip(low_map.components, pretrained):
            assert np.allclose(component.coeffs, old_coeffs)


def test_nhmf_core_workflow_smoke():
    rng = np.random.default_rng(34)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=6))
    model.train(use_corrections=True)
    x = rng.normal(size=(6, 2))
    mapped = model.evaluate(x)
    recovered = model.inverse(mapped)
    recovered_chunked = model.inverse(mapped, max_batch_rows=2)
    conditional = model.conditional_sample(5, fixed_indices=[0], fixed_values=np.array([1.25]), random_state=0)
    assert mapped.shape == x.shape
    assert recovered.shape == x.shape
    assert np.allclose(recovered, x, atol=1e-2)
    assert np.allclose(recovered_chunked, recovered, atol=1e-8)
    assert conditional.shape == (5, 2)
    assert np.allclose(conditional[:, 0], 1.25, atol=1e-5)
    assert model.log_det(x).shape == (6,)


def test_nhmf_conditional_sample_can_return_inverse_diagnostics():
    rng = np.random.default_rng(341)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=4))
    model.train(use_corrections=True)

    result = model.conditional_sample(
        3,
        fixed_indices=[0],
        fixed_values=np.array([0.5]),
        random_state=4,
        return_diagnostics=True,
    )

    assert isinstance(result, InverseResult)
    assert result.values.shape == (3, 2)
    assert result.diagnostics.failed_mask.shape == (3,)


def test_nhmf_conditioned_inverse_handles_mixed_fallback_brackets():
    rng = np.random.default_rng(342)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng),
        nhmf_params=_make_params(2, 1, 2, maxiter=4),
    )
    model.train(use_corrections=True)

    def mixed_evaluate(x):
        return np.where(x[:, 0] < 0.0, x[:, 1], 0.0)

    model.components[1].evaluate = mixed_evaluate
    model.components[1].evaluate_derivative_xk = lambda x: np.zeros(x.shape[0])
    fixed_values = model.hf_standardization.mean[0] + model.hf_standardization.std[0] * np.asarray([[-1.0], [1.0]])
    result = model.inverse(
        np.asarray([[0.0, 1.0], [0.0, 1.0]]),
        fixed_indices=[0],
        fixed_values=fixed_values,
        max_iter=8,
        tol=2.0e-2,
        inverse_options=InverseOptions(max_bracket_expansions=1),
        return_diagnostics=True,
    )

    assert np.all(np.isfinite(result.values[0]))
    assert np.all(np.isnan(result.values[1]))
    assert result.diagnostics.failed_mask.tolist() == [False, True]
    assert result.diagnostics.component_failure_counts.tolist() == [0, 1]
    assert result.diagnostics.failures[0].reason == "bracket_not_found"


def test_nhmf_verbose_training_reports_component_progress(capsys):
    rng = np.random.default_rng(341)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=5))

    model.train(use_corrections=False, verbose=True)
    captured = capsys.readouterr()

    assert "Training NHMF component 1/2." in captured.out
    assert "Training NHMF component 2/2." in captured.out
    assert "Completed NHMF training for component 1." in captured.out
    assert "Completed NHMF training for component 2." in captured.out
    assert "dimension" not in captured.out


def test_nhmf_gradient_matches_finite_difference_without_corrections():
    rng = np.random.default_rng(35)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    analytic = component.gradient(coeffs, x_hf, reg_cst=1e-3)
    numeric = finite_difference_gradient(lambda c: component.objective(c, x_hf, reg_cst=1e-3), coeffs)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_nhmf_gradient_matches_finite_difference_with_corrections():
    rng = np.random.default_rng(36)
    params = _make_params(2, 1, 2, maxiter=3)
    params.shift_order = 1
    params.scale_order = 2
    params.correction_order = 1
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    analytic = component.gradient(coeffs, x_hf, reg_cst=1e-3)
    numeric = finite_difference_gradient(lambda c: component.objective(c, x_hf, reg_cst=1e-3), coeffs)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_shared_nhmf_gradient_matches_finite_difference_without_corrections():
    rng = np.random.default_rng(361)
    params = _make_params(2, 1, 2, maxiter=3)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    analytic = component.gradient(coeffs, x_hf, reg_cst=1e-3)
    numeric = finite_difference_gradient(lambda c: component.objective(c, x_hf, reg_cst=1e-3), coeffs)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_shared_nhmf_precomputed_objective_and_gradient_match_direct_with_corrections():
    rng = np.random.default_rng(362)
    params = _make_params(2, 1, 2, maxiter=3)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)
    assert np.allclose(
        component.objective(coeffs, x_hf, reg_cst=1e-3),
        component.objective_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache),
    )
    assert np.allclose(
        component.gradient(coeffs, x_hf, reg_cst=1e-3),
        component.gradient_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache),
    )


def test_nhmf_precomputed_objective_and_gradient_match_direct_without_corrections():
    rng = np.random.default_rng(37)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)
    assert np.allclose(component.objective(coeffs, x_hf, reg_cst=1e-3), component.objective_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache))
    assert np.allclose(component.gradient(coeffs, x_hf, reg_cst=1e-3), component.gradient_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache))


def test_nhmf_precomputed_objective_and_gradient_match_direct_with_corrections():
    rng = np.random.default_rng(38)
    params = _make_params(2, 2, 2, maxiter=3)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    coeffs = component.coeffs + rng.normal(scale=1e-2, size=component.coeffs.shape)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)
    assert np.allclose(component.objective(coeffs, x_hf, reg_cst=1e-3), component.objective_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache))
    assert np.allclose(component.gradient(coeffs, x_hf, reg_cst=1e-3), component.gradient_precomp(coeffs, x_hf, reg_cst=1e-3, cache=cache))


def test_nhmf_joint_regularization_can_exclude_parent_terms():
    rng = np.random.default_rng(381)
    params = _make_params(2, 2, 2, maxiter=3)
    params.regularize_parent_terms = False
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    x_hf = model.standardized_train_data[:, :2]

    coeffs = np.zeros_like(component.coeffs)
    shift_size = component.coefficients.shift.flatten().size
    scale_sizes = [scale.flatten().size for scale in component.coefficients.scales]
    parent_sizes = [parent.flatten().size for parent in component.coefficients.parents]
    parent_start = shift_size + sum(scale_sizes)
    coeffs[parent_start : parent_start + parent_sizes[0]] = rng.normal(size=parent_sizes[0])

    assert np.allclose(
        component.objective(coeffs, x_hf, reg_cst=1e-2),
        component.objective(coeffs, x_hf, reg_cst=0.0),
    )
    reg_gradient = component.gradient(coeffs, x_hf, reg_cst=1e-2) - component.gradient(coeffs, x_hf, reg_cst=0.0)
    assert np.allclose(reg_gradient[parent_start:], 0.0)


def test_nhmf_precompute_caches_parent_standardized_inputs_and_quadrature():
    rng = np.random.default_rng(39)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)
    assert cache.x_hf_standardized.shape == x_hf.shape
    assert cache.quad_hf_standardized.shape[:2] == (
        component.shift_pair.quadrature_rule.num_points,
        x_hf.shape[0],
    )
    assert len(cache.parent_caches) == len(component.parent_terms)
    for parent_cache in cache.parent_caches:
        assert parent_cache.sample_inputs.shape == x_hf.shape
        assert parent_cache.quad_inputs.shape == cache.quad_hf_standardized.shape
        assert parent_cache.low_fidelity_cache is not None


def test_nhmf_uses_hf_standardization_for_coupled_map_only():
    rng = np.random.default_rng(390)
    train_data = _make_data(rng)
    model = NonHierarchicalTriangularMap(train_data=train_data, nhmf_params=_make_params(2, 1, 2, maxiter=3))

    hf_mean = train_data[0].mean(axis=0)
    hf_std = train_data[0].std(axis=0, ddof=0)
    hf_std = np.where(hf_std == 0.0, 1.0, hf_std)

    assert np.allclose(model.hf_standardization.mean, hf_mean)
    assert np.allclose(model.hf_standardization.std, hf_std)
    assert np.allclose(model.standardized_train_data, (train_data[0] - hf_mean) / hf_std)
    assert model.low_fidelity_standardizations == []


def test_nhmf_pretrained_parent_maps_store_local_standardizations():
    rng = np.random.default_rng(391)
    train_data = _make_data(rng)
    model = NonHierarchicalTriangularMap(train_data=train_data, nhmf_params=_make_params(2, 1, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)

    hf_mean = model.hf_standardization.mean
    hf_std = model.hf_standardization.std
    assert len(model.low_fidelity_standardizations) == len(model.low_fidelity_maps) == 2
    for dataset, low_map, params in zip(train_data[1:], model.low_fidelity_maps, model.low_fidelity_standardizations):
        expected_mean = dataset.mean(axis=0)
        expected_std = dataset.std(axis=0, ddof=0)
        assert np.allclose(params.mean, expected_mean)
        assert np.allclose(params.std, expected_std)
        assert np.allclose(low_map.standardization_params.mean, expected_mean)
        assert np.allclose(low_map.standardization_params.std, expected_std)
        assert not np.allclose(params.mean, hf_mean)
        assert not np.allclose(params.std, hf_std)


def test_nhmf_parent_adapter_matches_direct_parent_component_on_hf_inputs():
    rng = np.random.default_rng(392)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    x_hf = model.standardized_train_data[:, :2]

    for parent_term, low_map in zip(component.parent_terms, model.low_fidelity_maps):
        direct_component = low_map.components[1]
        direct_non = direct_component.evaluate_nonmonotone(x_hf)
        direct_pre = direct_component.evaluate_pre_monotone(x_hf)
        assert np.allclose(parent_term.non_value_from_hf(x_hf), direct_non)
        assert np.allclose(parent_term.pre_value_from_hf(x_hf), direct_pre)


def test_nhmf_training_populates_component_cache_when_precomputed_enabled():
    rng = np.random.default_rng(40)
    params = _make_params(2, 1, 2, maxiter=4)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model.train(use_corrections=True)
    assert all(component._training_cache is not None for component in model.components)


def test_nhmf_scale_constant_initializer_updates_only_projected_constants():
    rng = np.random.default_rng(4011)
    params = _make_params(2, 1, 2, maxiter=20)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)

    component.initialize_scale_constants(x_hf, cache=cache)

    constants = []
    for scale_pair in component.scale_pairs:
        non_idx, pre_idx = scale_pair.constant_coeff_positions()
        coeffs = scale_pair.coeffs
        active = np.zeros_like(coeffs, dtype=bool)
        active[[non_idx, pre_idx]] = True
        constants.append(coeffs[non_idx])
        assert not np.isclose(coeffs[pre_idx], 0.0)
        assert np.allclose(coeffs[~active], 0.0)

    assert len(constants) == 2
    assert np.allclose(component.shift_pair.coeffs, 0.0)
    assert all(np.allclose(parent_term.coeffs, 0.0) for parent_term in component.parent_terms)
    assert component.scale_constant_optimization_result is not None


def test_shared_nhmf_scale_constant_gradient_matches_finite_difference():
    rng = np.random.default_rng(4012)
    params = _make_params(2, 1, 2, maxiter=3)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=params)
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)
    constants = np.array([0.4, 0.6], dtype=float)
    analytic = component.scale_constant_gradient(constants, x_hf, cache=cache)
    numeric = finite_difference_gradient(lambda c: component.scale_constant_objective(c, x_hf, cache=cache), constants)
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_nhmf_scale_constant_initializer_improves_hf_only_objective():
    rng = np.random.default_rng(402)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=20))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    x_hf = model.standardized_train_data[:, :2]
    cache = component.precompute(x_hf)

    before = component._high_fidelity_objective_value(x_hf, cache=cache)
    component.initialize_scale_constants(x_hf, cache=cache)
    after = component._high_fidelity_objective_value(x_hf, cache=cache)

    assert after <= before + 1e-10
    assert np.isclose(component.scale_constant_optimization_result["initial_fun"], before)
    assert np.isclose(component.scale_constant_optimization_result["fun"], after)


def test_nhmf_training_records_scale_constant_initialization_diagnostics():
    rng = np.random.default_rng(403)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 1, 2, maxiter=6))

    model.train(use_corrections=True)

    assert all(component.scale_constant_optimization_result is not None for component in model.components)
    for component in model.components:
        diagnostics = component.scale_constant_optimization_result
        assert diagnostics["reg_cst"] == 0.0
        assert len(diagnostics["constants"]) == 2
        assert diagnostics["fun"] <= diagnostics["initial_fun"] + 1e-10


def test_nhmf_scale_pairs_initialize_to_partition_of_unity_constants():
    rng = np.random.default_rng(41)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=False)
    expected = 0.5
    for scale_pair in component.scale_pairs:
        non_constant = None
        constant_idx, linear_idx = scale_pair.constant_coeff_positions()
        assert scale_pair.expansion.coeffs[constant_idx] == expected
        assert scale_pair.expansion.coeffs[linear_idx] != 0.0
        active = {constant_idx, linear_idx}
        assert all(value == 0.0 for idx, value in enumerate(scale_pair.expansion.coeffs) if idx not in active)


def test_nhmf_shift_and_corrections_remain_zero_initialized():
    rng = np.random.default_rng(42)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    assert np.allclose(component.shift_pair.coeffs, 0.0)
    for parent_term in component.parent_terms:
        assert np.allclose(parent_term.coeffs, 0.0)


def test_nhmf_structured_coefficients_round_trip_and_shapes():
    rng = np.random.default_rng(43)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    blocks = component.coefficients
    assert len(blocks.scales) == 2
    assert len(blocks.parents) == 2
    shape_summary = blocks.shapes()
    assert "shift" in shape_summary
    assert len(shape_summary["scales"]) == 2
    assert len(shape_summary["parents"]) == 2
    updated = component.coeffs + rng.normal(scale=1e-3, size=component.coeffs.shape)
    blocks.set_from_flat(updated)
    assert np.allclose(component.coeffs, updated)


def test_nhmf_coefficients_follow_single_fidelity_style_surface():
    rng = np.random.default_rng(44)
    model = NonHierarchicalTriangularMap(train_data=_make_data(rng), nhmf_params=_make_params(2, 2, 2, maxiter=3))
    model._pretrain_low_fidelity_maps(verbose=False)
    component = model._build_component(component_dim=2, use_corrections=True)
    model.components = [component]
    assert component.coefficients.__class__.__module__.endswith("non_hierarchical_multifidelity.coefficients")
    assert model.coefficients.__class__.__module__.endswith("non_hierarchical_multifidelity.coefficients")
    map_component_coefficients = model.coefficients.components[0]
    component_coefficients = component.coefficients
    assert map_component_coefficients.shapes() == component_coefficients.shapes()
    np.testing.assert_array_equal(map_component_coefficients.flatten(), component_coefficients.flatten())
