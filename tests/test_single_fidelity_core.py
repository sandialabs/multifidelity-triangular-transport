"""Checks core single-fidelity construction, inversion, coefficients, and standardization."""

import numpy as np

from mftt.single_fidelity import (
    InverseOptions,
    InverseResult,
    MapParams,
    MultivariateHermiteFunction,
    StandardizationParams,
    TriangularMap,
    component_params_from_multi_index_sets,
    generate_multi_indices,
)
from mftt.single_fidelity.reference import Reference


def test_generate_multi_indices():
    multi_indices = generate_multi_indices(2, 2)
    assert (0, 0) in multi_indices
    assert (1, 1) in multi_indices
    assert (0, 2) in multi_indices


def test_multivariate_hermite_function_vectorized_output():
    basis = MultivariateHermiteFunction((1, 0))
    x = np.array([[0.0, 1.0], [2.0, -1.0]])
    values = basis.evaluate(x)
    assert values.shape == (2,)


def test_identity_initialized_map_is_near_identity():
    rng = np.random.default_rng(0)
    data = rng.normal(size=(40, 2))
    model = TriangularMap(train_data=data, params=MapParams(total_order=2))
    x = rng.normal(size=(8, 2))
    y = model.evaluate(x)
    x_recovered = model.inverse(y)
    standardized_x = (x - model.standardization_params.mean) / model.standardization_params.std
    expected_logdet = -np.sum(np.log(model.standardization_params.std))
    assert np.allclose(y, standardized_x, atol=5e-3)
    assert np.allclose(model.log_det(x), expected_logdet, atol=5e-3)
    assert np.allclose(x_recovered, x, atol=5e-3)


def test_constructor_reference_is_honored():
    rng = np.random.default_rng(901)
    data = rng.normal(size=(30, 2))
    reference = Reference(
        logpdf_fn=lambda x: -0.5 * np.sum((x - 1.5) ** 2, axis=1),
        score_fn=lambda x: -(x - 1.5),
    )
    model = TriangularMap(data, MapParams(total_order=1), reference=reference)
    assert model.reference is reference
    assert np.allclose(model.pullback_logpdf(data[:3]), reference.evaluate_logpdf(model.evaluate(data[:3])) + model.log_det(data[:3]))


def test_inverse_row_chunking_preserves_order_and_per_row_conditioning():
    rng = np.random.default_rng(101)
    data = rng.normal(size=(40, 2))
    model = TriangularMap(train_data=data,
        params=MapParams(total_order=2),
    )
    reference = rng.normal(size=(8, 2))
    fixed_values = np.linspace(-1.0, 1.0, reference.shape[0]).reshape(-1, 1)
    expected = model.inverse(reference, fixed_indices=[0], fixed_values=fixed_values)

    actual = model.inverse(
        reference,
        fixed_indices=[0],
        fixed_values=fixed_values,
        max_batch_rows=3,
    )

    assert np.allclose(actual, expected, atol=1e-8)
    assert np.allclose(actual[:, 0], fixed_values[:, 0])


def test_inverse_rejects_invalid_row_batch_limits():
    rng = np.random.default_rng(102)
    model = TriangularMap(train_data=rng.normal(size=(20, 2)),
        params=MapParams(total_order=1),
    )
    reference = rng.normal(size=(4, 2))

    for invalid in (0, -1, 1.5):
        try:
            model.inverse(reference, max_batch_rows=invalid)
        except ValueError as exc:
            assert "positive integer" in str(exc)
        else:
            raise AssertionError("Expected an invalid inverse row limit to raise.")


def test_inverse_diagnostics_report_failed_rows_and_fill_nan():
    rng = np.random.default_rng(103)
    model = TriangularMap(train_data=rng.normal(size=(20, 2)),
        params=MapParams(total_order=1),
    )
    model.components[1].evaluate = lambda x: np.zeros(x.shape[0], dtype=float)
    model.components[1].evaluate_derivative_xk = lambda x: np.zeros(x.shape[0], dtype=float)

    result = model.inverse(
        np.asarray([[0.0, 3.0], [0.0, 4.0]], dtype=float),
        max_iter=1,
        inverse_options=InverseOptions(max_bracket_expansions=1),
        return_diagnostics=True,
    )

    assert isinstance(result, InverseResult)
    assert result.values.shape == (2, 2)
    assert np.all(np.isnan(result.values))
    assert result.diagnostics.failed_mask.tolist() == [True, True]
    assert result.diagnostics.component_failure_counts.tolist() == [0, 2]
    assert {failure.reason for failure in result.diagnostics.failures} == {"bracket_not_found"}


def test_conditional_sample_diagnostics_preserve_successful_return_shape():
    rng = np.random.default_rng(104)
    model = TriangularMap(train_data=rng.normal(size=(20, 2)),
        params=MapParams(total_order=1),
    )

    result = model.conditional_sample(
        4,
        fixed_indices=[0],
        fixed_values=np.array([0.25]),
        random_state=0,
        return_diagnostics=True,
    )

    assert isinstance(result, InverseResult)
    assert result.values.shape == (4, 2)
    assert not np.any(result.diagnostics.failed_mask)
    assert np.allclose(result.values[:, 0], 0.25)


def test_custom_multi_index_sets_can_build_map_cleanly():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(30, 2))
    component_params = component_params_from_multi_index_sets(
        multi_index_sets=[
            [(0,), (1,), (2,)],
            [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1)],
        ],
    )
    model = TriangularMap(train_data=data,
        params=MapParams(component_params=component_params),
    )
    x = rng.normal(size=(4, 2))
    y = model.evaluate(x)
    assert y.shape == (4, 2)
    assert model.components[0].params.basis_spec.multi_indices == [(0,), (1,), (2,)]
    assert model.components[1].params.basis_spec.multi_indices == [(0, 0), (0, 1), (1, 0), (0, 2), (1, 1)]


def test_from_multi_index_sets_constructor():
    rng = np.random.default_rng(6)
    data = rng.normal(size=(20, 2))
    model = TriangularMap.from_multi_index_sets(
        train_data=data,
        multi_index_sets=[
            [(0,), (1,)],
            [(0, 0), (0, 1), (2, 0)],
        ],
    )
    x = rng.normal(size=(3, 2))
    y = model.evaluate(x)
    assert y.shape == (3, 2)
    assert model.input_dim == 2
    assert model.total_order == 2
    description = str(model)
    assert "basis=explicit_multi_indices" in description
    assert "component_terms=[2, 3]" in description
    assert "component_max_degrees=[1, 2]" in description
    assert "total_order=" not in description


def test_custom_shared_basis_requires_final_coordinate_linear_seed():
    try:
        component_params_from_multi_index_sets([[(0,)]])
    except ValueError as exc:
        assert "linear seed" in str(exc)
    else:
        raise AssertionError("Expected a shared custom basis without a linear seed to fail.")


def test_triangular_map_str_reports_configuration():
    rng = np.random.default_rng(7)
    data = rng.normal(size=(16, 2))
    model = TriangularMap(train_data=data, params=MapParams(total_order=2))
    description = str(model)
    assert "TriangularMap" in description
    assert "family=single_fidelity" in description
    assert "input_dim=2" in description
    assert "reference=StandardNormal(dim=2)" in description
    assert "components=2" in description
    assert "basis=total_order(2)" in description


def test_single_fidelity_structured_coefficients_round_trip():
    rng = np.random.default_rng(8)
    data = rng.normal(size=(20, 2))
    model = TriangularMap(train_data=data, params=MapParams(total_order=2))
    flat = model.coeffs.copy()
    structured = model.coefficients
    assert structured.shapes()[0]["expansion"]["shared"] == model.components[0].expansion.coeffs.shape
    updated = flat + rng.normal(scale=1e-3, size=flat.shape)
    structured.set_from_flat(updated)
    assert np.allclose(model.coeffs, updated)


def test_triangular_map_honors_fixed_external_standardization():
    rng = np.random.default_rng(81)
    data = rng.normal(loc=[2.0, -1.0], scale=[0.5, 1.5], size=(24, 2))
    fixed_params = StandardizationParams(mean=np.array([10.0, 20.0]), std=np.array([2.0, 4.0]))
    model = TriangularMap(train_data=data,
        params=MapParams(
            total_order=2,
        ),
        standardization=fixed_params,
    )

    assert np.allclose(model.standardization_params.mean, fixed_params.mean)
    assert np.allclose(model.standardization_params.std, fixed_params.std)
    assert np.allclose(model.standardized_train_data, (data - fixed_params.mean) / fixed_params.std)

    x = rng.normal(size=(5, 2))
    y = model.evaluate(x)
    x_recovered = model.inverse(y)
    expected_logdet = -np.sum(np.log(fixed_params.std))
    assert np.allclose(x_recovered, x, atol=5e-3)
    assert np.allclose(model.log_det(x), expected_logdet, atol=5e-3)


def test_triangular_map_rejects_mismatched_external_standardization():
    rng = np.random.default_rng(82)
    data = rng.normal(size=(12, 2))
    try:
        TriangularMap(train_data=data,
            params=MapParams(
                total_order=2,
            ),
            standardization=StandardizationParams(mean=np.array([0.0]), std=np.array([1.0])),
        )
    except ValueError as exc:
        assert "input dimension" in str(exc)
    else:
        raise AssertionError("Expected mismatched external standardization to raise.")
