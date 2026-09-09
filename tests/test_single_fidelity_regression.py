"""Checks single-fidelity optimizer defaults and end-to-end training workflows."""

import numpy as np

from mftt.single_fidelity import MapParams, OptimizationParams, TriangularMap


def test_optimization_params_default_optimizer_is_bfgs():
    params = OptimizationParams()
    assert params.optimizer == "BFGS"


def test_training_and_conditional_sampling_smoke():
    rng = np.random.default_rng(4)
    data = rng.normal(size=(25, 2))
    model = TriangularMap(train_data=data,
        params=MapParams(
            total_order=2,
            optimization=OptimizationParams(maxiter=20, gtol=1e-5, reg_cst=1e-3),
        ),
    )
    model.train()
    samples = model.conditional_sample(5, fixed_indices=[0], fixed_values=np.array([1.25]), random_state=0)
    assert samples.shape == (5, 2)
    assert np.allclose(samples[:, 0], 1.25, atol=1e-5)


def test_l_bfgs_b_single_fidelity_training_smoke():
    rng = np.random.default_rng(5)
    data = rng.normal(size=(28, 2))
    model = TriangularMap(train_data=data,
        params=MapParams(
            total_order=2,
            optimization=OptimizationParams(optimizer="L-BFGS-B", maxiter=20, gtol=1e-5, reg_cst=1e-3),
        ),
    )
    model.train()
    probe = rng.normal(size=(4, 2))
    mapped = model.evaluate(probe)
    diagnostics = model.optimization_diagnostics()

    assert mapped.shape == probe.shape
    assert np.all(np.isfinite(mapped))
    assert all(component is not None for component in diagnostics["components"])
    assert {component["optimizer"] for component in diagnostics["components"]} == {"L-BFGS-B"}
