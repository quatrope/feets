#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute feets

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

from setuptools import setup, find_packages

os.environ["FEETS_IN_SETUP"] = "True"
import feets


# =============================================================================
# CONSTANTS
# =============================================================================

REQUIREMENTS = [
    "numpy",
    "scipy>=1,<2",
    "pytest",
    "statsmodels",
    "astropy>=6,<7",
    "pandas>=2,<3",
    "requests",
    "attrs",
    "joblib",
]


# =============================================================================
# FUNCTIONS
# =============================================================================


def do_setup():
    setup(
        name=feets.NAME,
        version=feets.VERSION,
        long_description=feets.DOC,
        description=feets.DOC.splitlines()[0],
        author=feets.AUTHORS,
        author_email=feets.EMAIL,
        url=feets.URL,
        license=feets.LICENSE,
        keywords=list(feets.KEYWORDS),
        include_package_data=True,
        classifiers=[
            "Development Status :: 4 - Beta",
            "Intended Audience :: Education",
            "Intended Audience :: Science/Research",
            "License :: OSI Approved :: MIT License",
            "Operating System :: OS Independent",
            "Programming Language :: Python",
            "Programming Language :: Python :: 2",
            "Programming Language :: Python :: 2.7",
            "Programming Language :: Python :: 3",
            "Programming Language :: Python :: 3.4",
            "Programming Language :: Python :: 3.5",
            "Programming Language :: Python :: Implementation :: CPython",
            "Topic :: Scientific/Engineering",
        ],
        packages=[pkg for pkg in find_packages() if pkg.startswith("feets")],
        install_requires=REQUIREMENTS,
    )


if __name__ == "__main__":
    do_setup()
