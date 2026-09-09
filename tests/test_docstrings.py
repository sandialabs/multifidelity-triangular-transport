"""Checks that public APIs document every parameter in NumPy-style docstrings."""

from __future__ import annotations

import inspect
import re

import mftt


def _parameter_entries(obj: object) -> dict[str, list[str]]:
    """Extract described NumPy-style parameter entries from one docstring."""
    lines = (inspect.getdoc(obj) or "").splitlines()
    for index, line in enumerate(lines[:-1]):
        if line.strip() != "Parameters" or set(lines[index + 1].strip()) != {"-"}:
            continue
        entries: dict[str, list[str]] = {}
        current: list[str] = []
        for line_index in range(index + 2, len(lines)):
            line = lines[line_index]
            if (
                line_index + 1 < len(lines)
                and line.strip()
                and set(lines[line_index + 1].strip()) == {"-"}
            ):
                break
            match = re.match(
                r"^\s*([*]{0,2}[A-Za-z_]\w*(?:\s*,\s*[*]{0,2}[A-Za-z_]\w*)*)\s*:",
                line,
            )
            if match:
                current = [name.strip().lstrip("*") for name in match.group(1).split(",")]
                for name in current:
                    entries[name] = []
            elif current and line.strip():
                for name in current:
                    entries[name].append(line.strip())
        return entries
    return {}


def _signature_parameters(obj: object) -> set[str]:
    """Return documented signature parameters, excluding bound receivers."""
    return {
        parameter.name
        for parameter in inspect.signature(obj).parameters.values()
        if parameter.name not in {"self", "cls"}
    }


def _assert_complete_parameters(obj: object, expected: set[str] | None = None) -> None:
    """Assert parameter names and descriptions match the public signature."""
    entries = _parameter_entries(obj)
    expected = _signature_parameters(obj) if expected is None else expected
    assert entries.keys() == expected, f"{obj}: documented parameters do not match signature"
    assert all(descriptions for descriptions in entries.values()), f"{obj}: parameter descriptions are incomplete"


def test_every_exported_callable_has_complete_parameter_documentation() -> None:
    """Keep package-root callable signatures and NumPy fields synchronized."""
    for name in mftt.__all__:
        if name == "__version__":
            continue
        obj = getattr(mftt, name)
        assert inspect.getdoc(obj), f"mftt.{name} lacks a docstring"
        if inspect.isclass(obj):
            _assert_complete_parameters(obj)
        elif callable(obj):
            _assert_complete_parameters(obj)


def test_every_public_map_method_has_complete_parameter_documentation() -> None:
    """Keep all API-rendered map methods synchronized with their signatures."""
    for map_class in (
        mftt.TriangularMap,
        mftt.HierarchicalTriangularMap,
        mftt.NonHierarchicalTriangularMap,
    ):
        for name, method in inspect.getmembers(map_class, predicate=callable):
            if name.startswith("_"):
                continue
            assert inspect.getdoc(method), f"{map_class.__name__}.{name} lacks a docstring"
            _assert_complete_parameters(method)


def test_regression_prone_public_methods_document_every_parameter() -> None:
    """Pin representative formerly incomplete API entries explicitly."""
    methods = (
        mftt.TriangularMap.component_log_det,
        mftt.TriangularMap.conditional_sample,
        mftt.TriangularMap.conditional_pullback_logpdf,
        mftt.TriangularMap.inverse,
        mftt.HierarchicalTriangularMap.component_log_det,
        mftt.HierarchicalTriangularMap.conditional_sample,
        mftt.HierarchicalTriangularMap.inverse,
        mftt.NonHierarchicalTriangularMap.component_log_det,
        mftt.NonHierarchicalTriangularMap.conditional_sample,
        mftt.NonHierarchicalTriangularMap.inverse,
        mftt.Reference.from_scipy,
        mftt.map_gaussian_diagnostics,
    )
    for method in methods:
        _assert_complete_parameters(method)
