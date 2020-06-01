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
# DOC
# =============================================================================

"""feets.extractors.ext_dmdt Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from feets import extractors


# =============================================================================
# Test cases
# =============================================================================


def test_feets_dmdt():
    random = np.random.RandomState(42)
    ext = extractors.DeltamDeltat()
    params = ext.get_default_params()
    time = np.arange(0, 1000)

    values = np.empty(50)
    for idx in range(values.size):
        mags = random.normal(size=1000)
        feats = ext.fit(magnitude=mags, time=time, **params)
        values[idx] = np.sum(list(feats.values()))

    np.testing.assert_allclose(values.mean(), 424.56)


def test_dmdt_repr():
    ext = extractors.DeltamDeltat()
    assert (
        repr(ext)
        == "DeltamDeltat(dt_bins=<numpy.ndarray>, dm_bins=<numpy.ndarray>)"
    )
