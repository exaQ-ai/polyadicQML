# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
sys.path.insert(0, os.path.abspath('..'))


# -- Project information -----------------------------------------------------

project = 'Polyadic QML'
copyright = '2020, William Cappelletti @ Entropica Labs'
author = 'William Cappelletti @ Entropica Labs'
version = '0.0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.coverage',
    'sphinx.ext.napoleon',
    'sphinx.ext.imgmath',
    'sphinx.ext.intersphinx',
    'recommonmark',
    "sphinx_rtd_theme",
]

imgmath_image_format = 'svg'
add_function_parentheses = False

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

# External sphinx doc referenced inside
intersphinx_mapping = {
    'numpy': ('https://numpy.org/doc/stable/', None),
    'cupy': ('https://docs-cupy.chainer.org/en/stable/', None),
    'qiskit': ('https://qiskit.org/documentation/', None),
}

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# -- Options for HTML output -------------------------------------------------

import sphinx_rtd_theme

html_theme = "sphinx_rtd_theme"

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

html_favicon = "favicon.ico"
