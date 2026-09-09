"""Checks documentation, release files, tutorials, and public API references."""

from __future__ import annotations

import re
from pathlib import Path

import mftt


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_public_exports_are_documented() -> None:
    api_text = "\n".join(path.read_text() for path in (PROJECT_ROOT / "docs").glob("api/**/*.md"))
    api_text += (PROJECT_ROOT / "docs" / "api.md").read_text()
    assert mftt.__version__ == "0.1.0"
    for name in mftt.__all__:
        assert hasattr(mftt, name)
        assert f"mftt.{name}" in api_text or f"`{name}`" in api_text or name == "__version__"
        if name != "__version__":
            assert getattr(mftt, name).__doc__


def test_markdown_relative_links_resolve() -> None:
    pattern = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
    markdown_files = [
        PROJECT_ROOT / "README.md",
        *(PROJECT_ROOT / "docs").rglob("*.md"),
        PROJECT_ROOT / "DEVELOPER_GUIDE.md",
    ]
    for document in markdown_files:
        for target in pattern.findall(document.read_text()):
            if target.startswith(("http://", "https://", "#", "mailto:")):
                continue
            target_path = target.split("#", 1)[0]
            assert (document.parent / target_path).exists(), f"broken link in {document}: {target}"


def test_installation_documents_standard_tutorial_ready_installation() -> None:
    installation_text = (PROJECT_ROOT / "docs" / "installation.md").read_text().lower()
    for requirement in ("numpy", "scipy", "jupyter", "matplotlib", "nbconvert"):
        assert requirement in installation_text
    assert "python 3.10" in installation_text
    assert "python3 -m venv" in installation_text
    assert "py -m venv" in installation_text
    assert "python -m pip install ." in installation_text
    assert "there are no separate tutorial or test profiles" in installation_text
    assert (
        "github.com/sandialabs/multifidelity-triangular-transport" in installation_text
    )
    assert "sphinx" not in installation_text

    readme_text = (PROJECT_ROOT / "README.md").read_text()
    assert "docs/_build/html" not in readme_text
    legacy_tutorial_extra = "." + "[tutorials]"
    assert legacy_tutorial_extra not in readme_text

    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text()
    for requirement in (
        '"numpy>=1.21"',
        '"scipy>=1.8"',
        '"jupyter>=1.0"',
        '"matplotlib>=3.6"',
        '"nbconvert>=7.0"',
        '"pytest>=7.0"',
    ):
        assert requirement in pyproject_text
    assert "tutorials =" not in pyproject_text
    assert "test =" not in pyproject_text


def test_public_repository_metadata_is_consistent() -> None:
    repository_url = "https://github.com/sandialabs/multifidelity-triangular-transport"
    pyproject_text = (PROJECT_ROOT / "pyproject.toml").read_text()
    installation_text = (PROJECT_ROOT / "docs" / "installation.md").read_text()

    assert pyproject_text.count(repository_url) == 4
    assert f"git clone {repository_url}.git" in installation_text
    assert not (PROJECT_ROOT / "Gianluca_Instructions.md").exists()


def test_tutorials_do_not_modify_sys_path() -> None:
    legacy_name = "Shared_" + "Only_Library"
    for notebook in (PROJECT_ROOT / "docs" / "tutorials").glob("*.ipynb"):
        text = notebook.read_text()
        assert legacy_name not in text
        assert "sys.path" not in text


def test_sphinx_source_tree_is_complete() -> None:
    assert (PROJECT_ROOT / "docs" / "conf.py").exists()
    assert (PROJECT_ROOT / "docs" / "installation.md").exists()
    assert not (PROJECT_ROOT / "docs" / "quickstart.md").exists()
    assert not (PROJECT_ROOT / "examples").exists()
    assert not (PROJECT_ROOT / "docs" / "tutorials" / "index.md").exists()
    tutorial_names = {
        "single_fidelity_tutorial.ipynb",
        "hierarchical_multifidelity_tutorial.ipynb",
        "non_hierarchical_multifidelity_tutorial.ipynb",
    }
    tutorials_dir = PROJECT_ROOT / "docs" / "tutorials"
    assert {path.name for path in tutorials_dir.glob("*.ipynb")} == tutorial_names
    index_text = (PROJECT_ROOT / "docs" / "index.md").read_text()
    for tutorial_name in tutorial_names:
        assert f"tutorials/{tutorial_name.removesuffix('.ipynb')}" in index_text
    assert "tutorials/index" not in index_text
    assert (PROJECT_ROOT / "docs" / "api" / "maps.md").exists()
    map_pages = PROJECT_ROOT / "docs" / "api" / "maps"
    assert {path.name for path in map_pages.glob("*.md")} == {
        "triangular-map.md",
        "hierarchical-triangular-map.md",
        "non-hierarchical-triangular-map.md",
    }
    assert (PROJECT_ROOT / "docs" / "_static" / "api.css").exists()


def test_developer_notes_are_not_public_sphinx_documents() -> None:
    index_text = (PROJECT_ROOT / "docs" / "index.md").read_text()
    conf_text = (PROJECT_ROOT / "docs" / "conf.py").read_text()
    assert (PROJECT_ROOT / "docs" / "development.md").exists()
    assert (PROJECT_ROOT / "DEVELOPER_GUIDE.md").exists()
    assert "\ndevelopment\n" not in index_text
    assert '"development.md"' in conf_text


def test_license_starts_with_required_ntess_notice() -> None:
    license_text = (PROJECT_ROOT / "LICENSE").read_text()
    required_notice = (
        "Copyright 2026 National Technology & Engineering Solutions of Sandia, LLC\n"
        "(NTESS). Under the terms of Contract DE-NA0003525 with NTESS, the U.S.\n"
        "Government retains certain rights in this software."
    )
    assert license_text.startswith(f"{required_notice}\n\nMIT License\n")
    old_personal_notice = "Copyright (c) 2026 " + "Owen Davis and Gianluca Geraci"
    assert old_personal_notice not in license_text
    assert "html_show_copyright = False" in (PROJECT_ROOT / "docs" / "conf.py").read_text()


def test_autodoc_directives_use_eval_rst() -> None:
    api_sources = list((PROJECT_ROOT / "docs" / "api").rglob("*.md"))
    api_text = "\n".join(path.read_text() for path in api_sources)
    assert "```{autoclass}" not in api_text
    assert "```{autofunction}" not in api_text
    assert api_text.count("```{eval-rst}") == api_text.count(".. autoclass::") + api_text.count(
        ".. autofunction::"
    )


def test_old_package_name_is_absent_from_release_files() -> None:
    legacy_name = "Shared_" + "Only_Library"
    files = list((PROJECT_ROOT / "src").rglob("*.py"))
    files += list((PROJECT_ROOT / "tests").glob("*.py"))
    files += list((PROJECT_ROOT / "docs").glob("*.md"))
    for path in files:
        assert legacy_name not in path.read_text(), path
