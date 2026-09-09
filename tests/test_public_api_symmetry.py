"""Checks that every map family exposes the same core API and optimizer options."""

import numpy as np

from mftt import (
    HierarchicalMapParams,
    HierarchicalTriangularMap,
    NonHierarchicalMapParams,
    NonHierarchicalTriangularMap,
)
from mftt.single_fidelity import MapParams, OptimizationParams, Reference, TriangularMap


def _make_map_params(input_dim: int, total_order: int, maxiter: int = 4) -> MapParams:
    return MapParams(
        total_order=total_order,
        optimization=OptimizationParams(maxiter=maxiter, gtol=1e-5, reg_cst=1e-3),
    )


def test_all_map_families_expose_core_transport_api():
    rng = np.random.default_rng(300)
    sf = TriangularMap(train_data=rng.normal(size=(12, 2)), params=_make_map_params(2, 2))
    hmf = HierarchicalTriangularMap(train_data=[rng.normal(size=(12, 2)), rng.normal(size=(14, 2))],
        hierarchical_params=HierarchicalMapParams(fidelity_map_params=[_make_map_params(2, 2), _make_map_params(2, 2)]),
    )
    nhmf = NonHierarchicalTriangularMap(train_data=[rng.normal(size=(12, 2)), rng.normal(size=(14, 2)), rng.normal(size=(16, 2))],
        nhmf_params=NonHierarchicalMapParams(
            low_fidelity_map_params=[_make_map_params(2, 2), _make_map_params(2, 2)],
            shift_order=2,
            scale_order=2,
            correction_order=2,
            optimization=OptimizationParams(maxiter=4, gtol=1e-5, reg_cst=1e-3),
        ),
    )

    for model in (sf, hmf, nhmf):
        for attr in (
            "coefficients",
            "train",
            "evaluate",
            "inverse",
            "sample",
            "conditional_sample",
            "conditional_pullback_logpdf",
            "conditional_pullback_pdf",
            "log_det",
            "pullback_logpdf",
            "pullback_pdf",
            "pushforward_logpdf",
            "pushforward_pdf",
        ):
            assert hasattr(model, attr)


def test_l_bfgs_b_optimizer_option_is_supported_across_map_families():
    rng = np.random.default_rng(301)
    optimizer = OptimizationParams(optimizer="L-BFGS-B", maxiter=6, gtol=1e-5, reg_cst=1e-3)

    sf = TriangularMap(train_data=rng.normal(size=(14, 2)),
        params=MapParams(total_order=2, optimization=optimizer),
    )
    sf.train()
    sf_probe = rng.normal(size=(3, 2))
    sf_mapped = sf.evaluate(sf_probe)
    sf_diag = sf.optimization_diagnostics()
    assert sf_mapped.shape == sf_probe.shape
    assert sf_diag["components"][0]["optimizer"] == "L-BFGS-B"

    hmf = HierarchicalTriangularMap(train_data=[rng.normal(size=(12, 2)), rng.normal(size=(13, 2))],
        hierarchical_params=HierarchicalMapParams(
            fidelity_map_params=[
                MapParams(total_order=1, optimization=optimizer),
                MapParams(total_order=1, optimization=optimizer),
            ],
        ),
    )
    hmf.train(method="fixed_reference")
    hmf_probe = rng.normal(size=(3, 2))
    hmf_mapped = hmf.evaluate(hmf_probe)
    assert hmf_mapped.shape == hmf_probe.shape
    assert hmf.stages[0].map.optimization_diagnostics()["components"][0]["optimizer"] == "L-BFGS-B"

    nhmf = NonHierarchicalTriangularMap(train_data=[rng.normal(size=(12, 2)), rng.normal(size=(13, 2)), rng.normal(size=(14, 2))],
        nhmf_params=NonHierarchicalMapParams(
            low_fidelity_map_params=[
                MapParams(total_order=1, optimization=optimizer),
                MapParams(total_order=1, optimization=optimizer),
            ],
            shift_order=1,
            scale_order=1,
            correction_order=1,
            optimization=optimizer,
        ),
    )
    nhmf.train(use_corrections=False)
    nhmf_probe = rng.normal(size=(3, 2))
    nhmf_mapped = nhmf.evaluate(nhmf_probe)
    assert nhmf_mapped.shape == nhmf_probe.shape
    assert nhmf.low_fidelity_maps[0].optimization_diagnostics()["components"][0]["optimizer"] == "L-BFGS-B"
    assert nhmf.components[0].scale_constant_optimization_result["optimizer"] == "L-BFGS-B"
