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

"""IO code for read some macho lightcurves

The files are gathered from the original FATS project tutorial:
https://github.com/isadoranun/tsfeat

"""


# =============================================================================
# IMPORTS
# =============================================================================

import os
import pathlib
import tarfile

import numpy as np

from ..libs import bunch


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = pathlib.Path(os.path.abspath(os.path.dirname(__file__)))

DATA_PATH = PATH / "data" / "macho"

DATASET_NAME = "MACHO"

DATASET_DESCRIPTION = (
    "The files are gathered from the original FATS project "
    "tutorial: https://github.com/isadoranun/tsfeat"
)

# =============================================================================
# FUNCTIONS
# =============================================================================


def available_MACHO_lc():
    """Retrieve a list with the available MACHO lightcurves"""
    return [fp.rsplit(".", 2)[0] for fp in os.listdir(DATA_PATH)]


def load_MACHO_example():
    """lightcurve of 2 bands (R, B) from the MACHO survey.
    The Id of the source is 1.3444.614

    Notes
    -----

    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    """
    return load_MACHO("lc_1.3444.614")


def load_MACHO(macho_id):
    """lightcurve of 2 bands (R, B) from the MACHO survey.

    Notes
    -----

    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    """
    # Read the data
    tarpath = DATA_PATH / f"{macho_id}.tar.bz2"
    rpath = "{}.R.mjd".format(macho_id)
    bpath = "{}.B.mjd".format(macho_id)
    with tarfile.open(tarpath, mode="r:bz2") as tf:
        rlc = np.loadtxt(tf.extractfile(rpath))
        blc = np.loadtxt(tf.extractfile(bpath))

    # split the R and B bands
    r_band = bunch.Bunch(
        "R",
        {"time": rlc[:, 0], "magnitude": rlc[:, 1], "error": rlc[:, 2]},
    )
    b_band = (
        bunch.Bunch(
            "B",
            {"time": blc[:, 0], "magnitude": blc[:, 1], "error": blc[:, 2]},
        ),
    )

    # create the bands
    data = bunch.Bunch("data", {"R": r_band, "B": b_band})

    # the bunch data
    lc = bunch.Bunch(
        "lc",
        {
            "id": macho_id,
            "ds_name": DATASET_NAME,
            "description": DATASET_DESCRIPTION,
            "data": data,
        },
    )

    return lc
