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

import numpy as np


# =============================================================================
# CONSTANTS
# =============================================================================

PATH = os.path.abspath(os.path.dirname(__file__))

DATA_PATH = os.path.join(PATH, "data", "macho")

AVAILABLE_IDS = [fp.rsplit(".", 2)[0] for fp in os.listdir(DATA_PATH)]


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_macho_example():
    """lightcurve of 2 bands (R, B) from the MACHO survey.
    The Id of the source is 1.3444.614

    Notes
    -----

    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    """
    return load_macho("lc_1.3444.614")


def load_macho(macho_id):
    """lightcurve of 2 bands (R, B) from the MACHO survey.

    Notes
    -----

    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    """
    if macho_id not in AVAILABLE_IDS:
        raise ValueError("macho_id must be one of: {}".format(AVAILABLE_IDS))
    tarfile = "{}.tar.bz2".format(macho_id)
    tarpath = os.path.join(DATA_PATH, tarfile)

    rpath = "{}.R.mjd".format(macho_id)
    bpath = "{}.B.mjd".format(macho_id)
    with tarfile.open(tarpath, mode="r:bz2") as tf:
        rlc = np.loadtxt(tf.extractfile(rpath))
        blc = np.loadtxt(tf.extractfile(bpath))
    rtime, rmag, rerror = rlc[:, 0], rlc[:, 1], rlc[:, 2]
    btime, bmag, berror = blc[:, 0], blc[:, 1], blc[:, 2]
    return rtime, btime, rmag, bmag, rerror, berror
