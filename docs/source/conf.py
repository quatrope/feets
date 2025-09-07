# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

import os
import sys
import pathlib

import feets

import m2r2

# this path is pointing to project/docs/source
CURRENT_PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))
FEETS_PATH = CURRENT_PATH.parent.parent

sys.path.insert(0, str(FEETS_PATH))


# -- Project information -----------------------------------------------------

project = "feets"
copyright = "2024, QuatroPe; Clariá, Felipe"
author = "QuatroPe; Clariá, Felipe"

# The full version, including alpha/beta/rc tags
release = feets.__version__
version = feets.__version__


# -- General configuration ------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    "sphinx.ext.intersphinx",
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "numpydoc",
    "nbsphinx",
]

nbsphinx_execute = "never"

numpydoc_show_class_members = False

autosummary_generate = True

# Add any paths that contain templates here, relative to this directory.
templates_path = ["_templates"]

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = ["_build", "source/.ipynb_checkpoints/*"]

# # The name of the Pygments (syntax highlighting) style to use.
pygments_style = "sphinx"

# If true, `todo` and `todoList` produce output, else they produce nothing.
todo_include_todos = False

# The suffix(es) of source filenames.
# You can specify multiple suffix as a list of string:
#
# source_suffix = ['.rst', '.md']
source_suffix = ".rst"

# The master toctree document.
master_doc = "index"


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
html_theme = "alabaster"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ["_static"]

html_favicon = "_static/favicon.ico"

html_comment = "Copyright (c) 2024, QuatroPe; Clariá, Felipe"

html_css_files = []

html_theme_options = dict(
    fixed_sidebar=True,
    description="feATURE eXTRACTOR FOR tIME sERIES.",
    extra_nav_links={"feets Source Code": "https://github.com/quatrope/feets"},
    github_repo="feets",
    github_user="quatrope",
    logo_name=True,
    logo="logo_small.png",
)

# Example configuration for intersphinx: refer to the Python standard library.
intersphinx_mapping = {"python": ("https://docs.python.org/3", None)}


# -- Options for HTMLHelp output ------------------------------------------

# Output file base name for HTML help builder.
htmlhelp_basename = "feetsdoc"


# -- Options for LaTeX output ---------------------------------------------

latex_elements = {
    # The paper size ('letterpaper' or 'a4paper').
    #
    # 'papersize': 'letterpaper',
    # The font size ('10pt', '11pt' or '12pt').
    #
    # 'pointsize': '10pt',
    # Additional stuff for the LaTeX preamble.
    #
    # 'preamble': '',
    # Latex figure (float) alignment
    #
    # 'figure_align': 'htbp',
}

# Grouping the document tree into LaTeX files. List of tuples
# (source start file, target name, title,
#  author, documentclass [howto, manual, or own class]).
latex_documents = [
    (master_doc, "feets.tex", "feets Documentation", author, "manual"),
]


# # -- Options for manual page output ---------------------------------------

# # One entry per manual page. List of tuples
# # (source start file, name, description, authors, manual section).
man_pages = [
    (master_doc, "feets", "feets Documentation", author.split("; "), 1)
]


# # -- Options for Texinfo output -------------------------------------------

# Grouping the document tree into Texinfo files. List of tuples
# (source start file, target name, title, author,
#  dir menu entry, description, category)
texinfo_documents = [
    (
        master_doc,
        "feets",
        "feets Documentation",
        author,
        "feets",
        "feATURE eXTRACTOR FOR tIME sERIES.",
        "Miscellaneous",
    ),
]


# =============================================================================
# INJECT README INTO THE RESTRUCTURED TEXT
# =============================================================================

DYNAMIC_RST = {
    # "README.md": "README.rst",
    "CHANGELOG.md": "CHANGELOG.rst",
}

for md_name, rst_name in DYNAMIC_RST.items():
    md_path = FEETS_PATH / md_name
    with open(md_path) as fp:
        readme_md = fp.read().split("<!-- BODY -->", 1)[-1]

    rst_path = CURRENT_PATH / "_dynamic" / rst_name

    with open(rst_path, "w") as fp:
        fp.write(".. FILE AUTO GENERATED !! \n")
        fp.write(m2r2.convert(readme_md))
        print(f"{md_path} -> {rst_path} regenerated!")


# =============================================================================
# MAKE FEATURES CONF
# =============================================================================

import jinja2  # noqa

FEATURES_LIST_TEMPLATE = jinja2.Template(
    r"""
{%for feature, data in features%}
- [{{feature}}](<{{data.path}}>)
{%-endfor%}
"""
)


def make_features_conf():
    """Generate the features.rst file.

    The features.rst file is a list of all the features available in feets,
    with links to their documentation.

    """
    feature_paths = {}
    for feature in feets.extractor_registry.registered_features:
        extractor = feets.extractor_registry.extractor_of(feature)

        title = f"{extractor.__module__}.{extractor.__qualname__}"

        path = f"/api/feets.extractors.html#{title}"

        feature_paths[feature] = {"path": path}

    markdown = FEATURES_LIST_TEMPLATE.render(
        {"features": sorted(feature_paths.items())}
    )

    rst_path = CURRENT_PATH / "_dynamic" / "features.rst"

    with open(rst_path, "w") as fp:
        fp.write(".. FILE AUTO GENERATED !! \n")
        fp.write(m2r2.convert(markdown))
        print(f"{rst_path} regenerated!")


make_features_conf()
