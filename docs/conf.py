"""Sphinx configuration for the MFTT documentation."""

from __future__ import annotations

from pathlib import Path

from mftt import __version__


project = "MFTT"
author = "Owen Davis and Gianluca Geraci"
release = __version__
version = __version__

extensions = [
    "myst_nb",
    "sphinx_copybutton",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
]

templates_path: list[str] = []
# Keep the repository developer notes available as Markdown without publishing
# them as part of the user-facing Sphinx site.
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "development.md"]
source_suffix = {".md": "myst-nb", ".ipynb": "myst-nb", ".rst": "restructuredtext"}
master_doc = "index"

html_theme = "furo"
html_title = f"MFTT {version} documentation"
html_show_copyright = False
html_static_path = ["_static"]
html_css_files = ["api.css"]

autosummary_generate = True
autodoc_default_options = {
    "members": True,
    "undoc-members": False,
    "show-inheritance": True,
}
autodoc_typehints = "description"
autodoc_typehints_description_target = "documented_params"
napoleon_numpy_docstring = True
napoleon_google_docstring = False

myst_heading_anchors = 3
myst_enable_extensions = [
    "amsmath",
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
]

nb_execution_mode = "off"
nb_execution_raise_on_error = True
nb_merge_streams = True

copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable", None),
    "scipy": ("https://docs.scipy.org/doc/scipy", None),
}


# Keep this file discoverable to editors and make its path explicit for tools
# that inspect the documentation configuration.
DOCS_ROOT = Path(__file__).parent
