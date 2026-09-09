"""Checks hierarchical-map construction, training, coefficients, and conditioned inversion."""

import numpy as np
import pytest

from mftt import (
    HierarchicalMapParams,
    HierarchicalTriangularMap,
    InverseDiagnostics,
    InverseFailure,
    InverseOptions,
    InverseResult,
)
from mftt.single_fidelity import MapParams, OptimizationParams, Reference, TriangularMap
from tests._helpers import finite_difference_gradient


def _make_hmf_params(input_dim: int, total_order: int, num_fidelities: int, maxiter: int = 20) -> HierarchicalMapParams:
    return HierarchicalMapParams(
        fidelity_map_params=[
            MapParams(
                total_order=total_order,
                optimization=OptimizationParams(
                    maxiter=maxiter,
                    gtol=1e-5,
                    reg_cst=1e-3,
                ),
            )
            for _ in range(num_fidelities)
        ],
    )


def _banana_sample(
    n_samples: int,
    rng: np.random.Generator,
    scale_x: float = 1.0,
    bend: float = 0.1,
) -> np.ndarray:
    u = rng.standard_normal(n_samples)
    v = rng.standard_normal(n_samples)
    return np.column_stack([scale_x * u, v + bend * (u**2 - 1.0)])


def _standardized(data: np.ndarray, params) -> np.ndarray:
    return (data - params.mean) / params.std


def test_single_fidelity_pullback_score_matches_finite_difference():
    rng = np.random.default_rng(8)
    data = rng.normal(size=(25, 2))
    model = TriangularMap(train_data=data,
        params=MapParams(total_order=2),
    )
    x = rng.normal(size=(3, 2))
    analytic = model.pullback_score(x)
    numeric = np.vstack(
        [finite_difference_gradient(lambda row: model.pullback_logpdf(row[None, :])[0], sample) for sample in x]
    )
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_hmf_construction_keeps_configured_stages_only():
    params = _make_hmf_params(input_dim=2, total_order=2, num_fidelities=3)
    rng = np.random.default_rng(9)
    train_data = [rng.normal(size=(16, 2)) for _ in range(3)]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=params)
    assert len(model.stages) == 3
    assert len(model.maps) == 0
    assert all(stage.map is None for stage in model.stages)
    assert all(stage.reference is None for stage in model.stages)
    assert len(model.fidelity_standardizations) == 3
    for dataset, standardized, standardization in zip(
        train_data,
        model.standardized_train_data,
        model.fidelity_standardizations,
    ):
        assert np.allclose(standardization.mean, dataset.mean(axis=0))
        assert np.allclose(standardization.std, dataset.std(axis=0))
        assert np.allclose(standardized, _standardized(dataset, standardization))


def test_hmf_composes_shared_single_fidelity_stages():
    rng = np.random.default_rng(901)
    params = HierarchicalMapParams(
        fidelity_map_params=[
            MapParams(
                total_order=1,
                optimization=OptimizationParams(maxiter=3, gtol=1e-5, reg_cst=1e-3),
            )
            for _ in range(2)
        ],
    )
    train_data = [rng.normal(size=(14, 2)), rng.normal(size=(16, 2))]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=params)

    model.train(method="fixed_reference")

    probe = rng.normal(size=(4, 2))
    assert len(model.maps) == 2
    assert all(not hasattr(stage.map.params, "parameterization") for stage in model.stages)
    assert model.evaluate(probe).shape == probe.shape
    assert model.log_det(probe).shape == (probe.shape[0],)


def test_hmf_changing_reference_training_chains_references():
    params = _make_hmf_params(input_dim=2, total_order=2, num_fidelities=3, maxiter=5)
    rng = np.random.default_rng(90)
    train_data = [rng.normal(size=(16, 2)) for _ in range(3)]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=params)
    model.train(method="changing_reference")
    probe = np.array([[0.1, -0.2], [0.4, 0.3]])
    assert len(model.maps) == 3
    assert np.allclose(model.stages[-1].reference.evaluate_score(probe), -probe)
    assert np.allclose(model.stages[1].reference.evaluate_logpdf(probe), model.stages[2].map.pullback_logpdf(probe))
    assert np.allclose(model.stages[0].reference.evaluate_logpdf(probe), model.stages[1].map.pullback_logpdf(probe))


def test_hmf_str_reports_hierarchy_before_and_after_training():
    params = _make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=5)
    rng = np.random.default_rng(91)
    train_data = [rng.normal(size=(12, 2)), rng.normal(size=(14, 2))]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=params)
    before = str(model)
    assert "HierarchicalTriangularMap" in before
    assert "family=hierarchical_multifidelity" in before
    assert "num_fidelities=2" in before
    assert "method=untrained" in before
    assert "samples=[12, 14]" in before

    model.train(method="fixed_reference")
    after = str(model)
    assert "method=fixed_reference" in after


def test_hmf_coefficients_reuse_single_fidelity_structure():
    params = _make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=5)
    rng = np.random.default_rng(92)
    train_data = [rng.normal(size=(12, 2)), rng.normal(size=(14, 2))]
    model = HierarchicalTriangularMap(train_data=train_data, hierarchical_params=params)
    model.train(method="changing_reference")
    coefficients = model.coefficients
    assert len(coefficients.stages) == 2
    assert len(coefficients.stages[0].components) == 2
    expected_stages = [stage.map.coefficients for stage in model.stages if stage.map is not None]
    assert [stage.shapes() for stage in coefficients.stages] == [stage.shapes() for stage in expected_stages]
    for actual, expected in zip(coefficients.stages, expected_stages):
        np.testing.assert_array_equal(actual.flatten(), expected.flatten())


def test_hmf_train_requires_explicit_method():
    rng = np.random.default_rng(12)
    train_data = [rng.normal(size=(12, 2)), rng.normal(size=(14, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2),
    )
    try:
        model.train()
    except ValueError as exc:
        assert "requires method" in str(exc)
    else:
        raise AssertionError("Expected train() without a method to raise.")


def test_hmf_train_rejects_invalid_method():
    rng = np.random.default_rng(13)
    train_data = [rng.normal(size=(12, 2)), rng.normal(size=(14, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2),
    )
    try:
        model.train(method="not_a_method")
    except ValueError as exc:
        assert "Unsupported training method" in str(exc)
    else:
        raise AssertionError("Expected an invalid training method to raise.")


def test_hmf_train_runs_stagewise_aao_precomputed_with_analytic_references():
    rng = np.random.default_rng(11)
    train_data = [
        rng.normal(loc=0.2, scale=1.1, size=(18, 2)),
        rng.normal(loc=-0.1, scale=0.9, size=(20, 2)),
    ]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=8),
    )
    model.train(method="changing_reference")
    probe = rng.normal(size=(5, 2))
    mapped = model.evaluate(probe)
    log_det = model.log_det(probe)
    assert mapped.shape == probe.shape
    assert log_det.shape == (5,)
    assert np.all(np.isfinite(mapped))
    assert np.all(np.isfinite(log_det))


def test_hmf_changing_reference_stage_gradient_matches_finite_difference():
    rng = np.random.default_rng(101)
    train_data = [
        rng.normal(loc=[0.3, -0.2], scale=[1.1, 0.8], size=(18, 2)),
        rng.normal(loc=[-0.4, 0.5], scale=[0.9, 1.2], size=(20, 2)),
    ]
    model = HierarchicalTriangularMap(
        train_data=train_data,
        hierarchical_params=_make_hmf_params(
            input_dim=2,
            total_order=1,
            num_fidelities=2,
            maxiter=4,
        ),
    )
    model.train(method="changing_reference")

    high_stage = model.stages[0]
    lower_stage = model.stages[1]
    assert high_stage.map is not None
    assert lower_stage.map is not None
    assert high_stage.reference is not None

    probe = rng.normal(size=(5, 2))
    assert np.allclose(
        high_stage.reference.evaluate_logpdf(probe),
        lower_stage.map.pullback_logpdf(probe),
    )
    assert np.allclose(
        high_stage.reference.evaluate_score(probe),
        lower_stage.map.pullback_score(probe),
    )

    stage_map = high_stage.map
    coeffs = stage_map.coeffs + rng.normal(scale=1e-2, size=stage_map.coeffs.shape)
    analytic = stage_map.gradient_aao_precomp(coeffs, reg_cst=1e-3)
    numeric = finite_difference_gradient(
        lambda candidate: stage_map.objective_aao_precomp(candidate, reg_cst=1e-3),
        coeffs,
    )
    assert np.allclose(analytic, numeric, atol=1e-5, rtol=1e-4)


def test_hmf_fixed_reference_train_uses_transformed_stage_data_and_gaussian_refs():
    rng = np.random.default_rng(14)
    train_data = [
        rng.normal(loc=[4.0, -3.0], scale=[1.5, 0.4], size=(18, 2)),
        rng.normal(loc=[-2.0, 5.0], scale=[0.6, 1.8], size=(20, 2)),
    ]
    model = HierarchicalTriangularMap(train_data=[data.copy() for data in train_data],
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=8),
    )
    model.train(method="fixed_reference")
    assert model.evaluation_order == [1, 0]
    z_hf = model.standardized_train_data[0]
    z_lf = model.standardized_train_data[1]
    assert np.allclose(model.stages[1].reference.evaluate_score(z_lf[:3]), -z_lf[:3])
    assert np.allclose(model.stages[0].reference.evaluate_score(z_hf[:3]), -z_hf[:3])
    for stage in model.stages:
        assert np.allclose(stage.map.standardization_params.mean, np.zeros(2))
        assert np.allclose(stage.map.standardization_params.std, np.ones(2))
    assert np.allclose(model.stages[1].stage_train_data, z_lf)
    expected_high_stage = model.stages[1].map.evaluate(z_hf)
    assert np.allclose(model.stages[0].stage_train_data, expected_high_stage)
    old_behavior_high_stage = model.stages[1].map.evaluate(train_data[0])
    assert not np.allclose(model.stages[0].stage_train_data, old_behavior_high_stage)


def test_hmf_requires_training_before_evaluation():
    rng = np.random.default_rng(140)
    train_data = [rng.normal(size=(12, 2)), rng.normal(size=(14, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2),
    )
    try:
        model.evaluate(np.zeros((2, 2)))
    except ValueError as exc:
        assert "must be trained" in str(exc)
    else:
        raise AssertionError("Expected evaluate() before training to raise.")


def test_hmf_fixed_reference_core_workflow_smoke():
    rng = np.random.default_rng(15)
    train_data = [
        rng.normal(loc=0.4, scale=1.2, size=(18, 2)),
        rng.normal(loc=-0.2, scale=0.7, size=(22, 2)),
    ]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=10),
    )
    model.train(method="fixed_reference")
    probe = rng.normal(size=(6, 2))
    mapped = model.evaluate(probe)
    recovered = model.inverse(mapped)
    recovered_chunked = model.inverse(mapped, max_batch_rows=2)
    conditional = model.conditional_sample(5, fixed_indices=[0], fixed_values=np.array([0.75]), random_state=2)
    log_det = model.log_det(probe)
    probe_std = _standardized(probe, model.fidelity_standardizations[0])
    manual_after_lower = model.stages[1].map.evaluate(probe_std)
    manual_mapped = model.stages[0].map.evaluate(manual_after_lower)
    manual_log_det = (
        -np.sum(np.log(model.fidelity_standardizations[0].std))
        + model.stages[1].map.log_det(probe_std)
        + model.stages[0].map.log_det(manual_after_lower)
    )
    assert mapped.shape == probe.shape
    assert recovered.shape == probe.shape
    assert conditional.shape == (5, 2)
    assert log_det.shape == (6,)
    assert np.all(np.isfinite(mapped))
    assert np.all(np.isfinite(log_det))
    assert np.allclose(mapped, manual_mapped)
    assert np.allclose(log_det, manual_log_det)
    assert np.allclose(recovered, probe, atol=1e-2)
    assert np.allclose(recovered_chunked, recovered, atol=1e-8)
    assert np.allclose(conditional[:, 0], 0.75, atol=1e-5)


def test_hmf_conditional_sample_can_return_inverse_diagnostics():
    rng = np.random.default_rng(151)
    train_data = [
        rng.normal(loc=0.4, scale=1.2, size=(18, 2)),
        rng.normal(loc=-0.2, scale=0.7, size=(22, 2)),
    ]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=1, num_fidelities=2, maxiter=4),
    )
    model.train(method="fixed_reference")

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


@pytest.mark.parametrize("method", ["fixed_reference", "changing_reference"])
@pytest.mark.parametrize("num_fidelities", [2, 3])
def test_hmf_stagewise_conditioned_inverse_respects_fixed_prefix_and_batching(method, num_fidelities):
    rng = np.random.default_rng(153 + num_fidelities)
    train_data = [
        rng.normal(loc=0.1 * idx, scale=1.0 + 0.1 * idx, size=(24 + 2 * idx, 3))
        for idx in range(num_fidelities)
    ]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(
            input_dim=3,
            total_order=1,
            num_fidelities=num_fidelities,
            maxiter=8,
        ),
    )
    model.train(method=method)
    reference = rng.normal(size=(7, 3))
    fixed_values = np.column_stack(
        [
            np.linspace(-0.4, 0.4, reference.shape[0]),
            np.linspace(0.3, -0.3, reference.shape[0]),
        ]
    )

    stagewise = model.inverse(
        reference,
        fixed_indices=[0, 1],
        fixed_values=fixed_values,
    )
    chunked = model.inverse(
        reference,
        fixed_indices=[0, 1],
        fixed_values=fixed_values,
        max_batch_rows=3,
    )
    assert np.allclose(chunked, stagewise, atol=1.0e-6, rtol=1.0e-6)
    assert np.allclose(stagewise[:, :2], fixed_values, atol=1.0e-10)
    assert np.allclose(model.evaluate(stagewise)[:, 2], reference[:, 2], atol=1.0e-7, rtol=1.0e-7)


def test_hmf_stagewise_conditioned_inverse_propagates_failures_once(monkeypatch):
    rng = np.random.default_rng(158)
    train_data = [rng.normal(size=(20, 2)), rng.normal(size=(22, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=1, num_fidelities=2, maxiter=5),
    )
    model.train(method="fixed_reference")
    reverse_order = list(reversed(model.evaluation_order))
    failing_map = model.stages[reverse_order[0]].map
    downstream_map = model.stages[reverse_order[1]].map
    downstream_batch_sizes = []
    original_downstream_inverse = downstream_map.inverse

    def fail_one_row(output, **kwargs):
        del kwargs
        values = np.asarray(output, dtype=float).copy()
        values[1] = np.nan
        return InverseResult(
            values=values,
            diagnostics=InverseDiagnostics(
                failed_mask=np.asarray([False, True, False]),
                failures=(
                    InverseFailure(
                        row=1,
                        component=1,
                        reason="bracket_not_found",
                        residual=1.0,
                        bracket_radius=2.0,
                        evaluations=9,
                    ),
                ),
                component_failure_counts=np.asarray([0, 1]),
                max_residual=1.0,
            ),
        )

    def record_downstream(output, **kwargs):
        downstream_batch_sizes.append(np.asarray(output).shape[0])
        return original_downstream_inverse(output, **kwargs)

    monkeypatch.setattr(failing_map, "inverse", fail_one_row)
    monkeypatch.setattr(downstream_map, "inverse", record_downstream)
    result = model.inverse(
        np.zeros((3, 2)),
        fixed_indices=[0],
        fixed_values=np.asarray([0.25]),
        inverse_options=InverseOptions(),
        return_diagnostics=True,
    )

    assert result.diagnostics.failed_mask.tolist() == [False, True, False]
    assert np.all(np.isnan(result.values[1]))
    assert downstream_batch_sizes == [2]
    assert len(result.diagnostics.failures) == 1
    assert result.diagnostics.failures[0].row == 1
    assert result.diagnostics.component_failure_counts.tolist() == [0, 1]
    assert result.diagnostics.max_residual == 1.0


def test_hmf_conditioning_validates_prefix_shapes_and_edge_cases():
    rng = np.random.default_rng(159)
    train_data = [rng.normal(size=(18, 2)), rng.normal(size=(20, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=1, num_fidelities=2, maxiter=5),
    )
    model.train(method="fixed_reference")

    empty = model.conditional_sample(4, fixed_indices=[], fixed_values=np.asarray([]), random_state=11)
    unconditional = model.sample(4, random_state=11)
    assert np.allclose(empty, unconditional)

    fixed_values = np.asarray([0.2, -0.3])
    fully_fixed = model.conditional_sample(
        4,
        fixed_indices=[0, 1],
        fixed_values=fixed_values,
        random_state=12,
    )
    assert np.allclose(fully_fixed, np.broadcast_to(fixed_values, (4, 2)), atol=1.0e-10)

    with pytest.raises(ValueError, match="leading prefix"):
        model.conditional_sample(2, fixed_indices=[1], fixed_values=np.asarray([0.1]), random_state=13)
    with pytest.raises(ValueError, match="leading prefix"):
        model.inverse(
            np.zeros((2, 2)),
            fixed_indices=[1],
            fixed_values=np.asarray([0.1]),
        )
    with pytest.raises(ValueError, match="fixed_values must have shape"):
        model.conditional_sample(
            2,
            fixed_indices=[0],
            fixed_values=np.zeros((3, 1)),
            random_state=14,
        )


def test_hmf_both_methods_produce_valid_roundtrip_and_conditionals():
    rng = np.random.default_rng(16)
    train_data = [
        rng.normal(loc=0.1, scale=1.1, size=(18, 2)),
        rng.normal(loc=-0.1, scale=0.9, size=(20, 2)),
    ]
    probe = rng.normal(size=(6, 2))
    for method in ("changing_reference", "fixed_reference"):
        model = HierarchicalTriangularMap(train_data=[data.copy() for data in train_data],
            hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=8),
        )
        model.train(method=method)
        mapped = model.evaluate(probe)
        recovered = model.inverse(mapped)
        conditional = model.conditional_sample(5, fixed_indices=[0], fixed_values=np.array([1.25]), random_state=3)
        assert mapped.shape == probe.shape
        assert np.allclose(recovered, probe, atol=1e-2)
        assert np.allclose(conditional[:, 0], 1.25, atol=1e-5)


def test_hmf_conditional_sampling_tracks_high_fidelity_conditional_mean():
    rng = np.random.default_rng(19)
    banana_params = [
        {"scale_x": 1.05, "bend": 0.65},
        {"scale_x": 0.95, "bend": 0.42},
        {"scale_x": 0.85, "bend": 0.22},
    ]
    train_data = [
        _banana_sample(600, rng, **banana_params[0]),
        _banana_sample(800, rng, **banana_params[1]),
        _banana_sample(1000, rng, **banana_params[2]),
    ]
    fixed_x1 = 1.0
    true_mean = banana_params[0]["bend"] * ((fixed_x1 / banana_params[0]["scale_x"]) ** 2 - 1.0)
    for method in ("changing_reference", "fixed_reference"):
        model = HierarchicalTriangularMap(train_data=[data.copy() for data in train_data],
            hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=3, maxiter=30),
        )
        model.train(method=method)
        conditional = model.conditional_sample(1500, fixed_indices=[0], fixed_values=np.array([fixed_x1]), random_state=7)
        assert np.allclose(conditional[:, 0], fixed_x1, atol=1e-5)
        assert abs(np.mean(conditional[:, 1]) - true_mean) < 0.25


def test_single_fidelity_verbose_training_reports_component_completion(capsys):
    rng = np.random.default_rng(17)
    data = rng.normal(size=(20, 2))
    model = TriangularMap(train_data=data,
        params=MapParams(
            total_order=2,
            optimization=OptimizationParams(maxiter=5, gtol=1e-4, reg_cst=1e-3),
        ),
    )
    model.train(verbose=True)
    captured = capsys.readouterr()
    assert "Completed training component 1/2." in captured.out
    assert "Completed training component 2/2." in captured.out


def test_hmf_verbose_training_reports_stage_completion(capsys):
    rng = np.random.default_rng(18)
    train_data = [rng.normal(size=(16, 2)), rng.normal(size=(18, 2))]
    model = HierarchicalTriangularMap(train_data=train_data,
        hierarchical_params=_make_hmf_params(input_dim=2, total_order=2, num_fidelities=2, maxiter=5),
    )
    model.train(method="fixed_reference", verbose=True)
    captured = capsys.readouterr()
    assert "Completed stage 1/2 for method 'fixed_reference' (fidelity index 1)." in captured.out
    assert "Completed stage 2/2 for method 'fixed_reference' (fidelity index 0)." in captured.out
