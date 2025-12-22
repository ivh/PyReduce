#
# Configuration file for the Sphinx documentation builder.
#

import os
import sys

path = os.path.join(__file__, "../..")
path = os.path.abspath(path)
sys.path.insert(0, path)

# -- Project information -----------------------------------------------------
from pyreduce import __version__

project = "PyReduce"
copyright = "2019, Ansgar Wehrhahn"
author = "Ansgar Wehrhahn"

version = __version__.split("+")[0]
release = __version__

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.viewcode",
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

master_doc = "index"
language = "en"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]
pygments_style = "sphinx"

# -- MyST configuration ------------------------------------------------------

myst_enable_extensions = [
    "colon_fence",
    "deflist",
]
myst_heading_anchors = 3

# -- Options for HTML output -------------------------------------------------

html_theme = "alabaster"
html_static_path = []

# -- Options for HTMLHelp output ---------------------------------------------

htmlhelp_basename = "PyReducedoc"

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {}
latex_documents = [
    (master_doc, "PyReduce.tex", "PyReduce Documentation", "Ansgar Wehrhahn", "manual")
]

# -- Options for manual page output ------------------------------------------

man_pages = [(master_doc, "pyreduce", "PyReduce Documentation", [author], 1)]

# -- Options for Texinfo output ----------------------------------------------

texinfo_documents = [
    (
        master_doc,
        "PyReduce",
        "PyReduce Documentation",
        author,
        "PyReduce",
        "Echelle spectrograph data reduction pipeline.",
        "Miscellaneous",
    )
]
