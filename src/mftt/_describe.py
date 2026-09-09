"""Format human-readable summaries of configured transport maps."""

from __future__ import annotations

import numpy as np


def summarize_reference(reference, input_dim: int) -> str:
    """Classify a reference object for compact display output."""
    logpdf_fn = getattr(reference, "logpdf_fn", None)
    score_fn = getattr(reference, "score_fn", None)

    bound_self = getattr(logpdf_fn, "__self__", None)
    bound_name = getattr(logpdf_fn, "__name__", "")
    if bound_self is not None and bound_name == "pullback_logpdf":
        return f"MapInducedReference(source={type(bound_self).__name__}, score={'yes' if score_fn is not None else 'no'})"

    probe = np.array([[0.0] * input_dim, [0.25] * input_dim], dtype=float)
    try:
        expected_logpdf = -0.5 * np.sum(probe * probe, axis=1) - 0.5 * input_dim * np.log(2.0 * np.pi)
        observed_logpdf = np.asarray(reference.evaluate_logpdf(probe), dtype=float)
        if score_fn is not None:
            observed_score = np.asarray(reference.evaluate_score(probe), dtype=float)
        else:
            observed_score = None
        if np.allclose(observed_logpdf, expected_logpdf) and (
            observed_score is None or np.allclose(observed_score, -probe)
        ):
            return f"StandardNormal(dim={input_dim})"
    except Exception:
        pass
    return f"CustomReference(score={'yes' if score_fn is not None else 'no'})"


def format_triangular_map_lines(map_obj, title: str = "TriangularMap") -> list[str]:
    """Build one concise line describing a single-fidelity map."""
    if map_obj.params._basis_construction == "total_order":
        basis_summary = f"basis=total_order({map_obj.total_order})"
    else:
        component_terms = [len(component.params.basis_spec.multi_indices) for component in map_obj.components]
        component_max_degrees = [
            max(sum(multi_index) for multi_index in component.params.basis_spec.multi_indices)
            for component in map_obj.components
        ]
        basis_summary = (
            "basis=explicit_multi_indices, "
            f"component_terms={component_terms}, component_max_degrees={component_max_degrees}"
        )
    return [
        (
            f"{title}(family=single_fidelity, input_dim={map_obj.input_dim}, "
            f"{basis_summary}, components={len(map_obj.components)}, "
            f"samples={map_obj.train_data.shape[0]}, "
            f"reference={summarize_reference(map_obj.reference, map_obj.input_dim)})"
        )
    ]


def format_hierarchical_map_lines(map_obj) -> list[str]:
    """Build a concise state summary for a hierarchical map."""
    method = map_obj._last_training_method or "untrained"
    return [
        (
            f"HierarchicalTriangularMap(family=hierarchical_multifidelity, input_dim={map_obj.input_dim}, "
            f"num_fidelities={map_obj.num_fidelities}, method={method}, "
            f"samples={[stage.train_data.shape[0] for stage in map_obj._stages]})"
        )
    ]


def format_nhmf_map_lines(map_obj) -> list[str]:
    """Build a concise state summary for a non-hierarchical map."""
    correction_state = "untrained" if map_obj._last_use_corrections is None else str(map_obj._last_use_corrections)
    return [
        (
            f"NonHierarchicalTriangularMap(family=non_hierarchical_multifidelity, "
            f"input_dim={map_obj.input_dim}, low_fidelity_parents={len(map_obj.params.low_fidelity_map_params)}, "
            f"corrections={correction_state}, samples={[dataset.shape[0] for dataset in map_obj.train_data]})"
        )
    ]
