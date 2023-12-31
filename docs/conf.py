import os
import sys
from typing import List

sys.path.insert(0, os.path.abspath("."))

project = "lightning-experiments-logger"
copyright = "2023, Tobias Sebst"
author = "tobias.senst@googlemail.com"

autoapi_dirs = ["../experiments_addon"]

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.doctest",
    "sphinx.ext.intersphinx",
    "sphinx.ext.todo",
    "sphinx.ext.coverage",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosectionlabel",
]

templates_path = [""]

autoapi_generate_api_docs = True

exclude_patterns: List[str] = []

autoclass_content = "both"

autodoc_default_options = {
    "members": True,
    "member-order": "bysource",
    "special-members": "__init__",
}

html_theme = "sphinx_rtd_theme"

html_theme_options = {
    "collapse_navigation": False,
    "sticky_navigation": True,
    "navigation_depth": 6,
    "includehidden": True,
    "titles_only": False,
}

html_static_path = ["_static"]
