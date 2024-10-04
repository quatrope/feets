#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOCS
# =============================================================================

"""This file is for distribute feets

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os

os.environ["FEETS_IN_SETUP"] = "True"
import feets  # noqa

from setuptools import find_packages, setup  # noqa


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
    "custom_inherit",
    "dask",
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
