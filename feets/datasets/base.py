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

"""Base IO code for all datasets

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

DATA_PATH = os.path.join(PATH, "data")


# =============================================================================
# FUNCTIONS
# =============================================================================

def load_macho_example():
    """lightcurve of 2 bands from the MACHO survey.
    The Id of the source is 1.3444.614

    Notes
    -----

    The files are gathered from the original FATS project tutorial:
    https://github.com/isadoranun/tsfeat

    """
    path = os.path.join(DATA_PATH, "lc_1.3444.614.B_R.npz")
    with np.load(path) as npz:
        lc = (
            npz['mag'],
            npz['time'],
            npz['error'],
            npz['mag2'],
            npz['aligned_mag'],
            npz['aligned_mag2'],
            npz['aligned_time'],
            npz['aligned_error'],
            npz['aligned_error2'])
    return lc
