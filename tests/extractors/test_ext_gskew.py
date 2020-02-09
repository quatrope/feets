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

"""feets.extractors.ext_signature Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import warnings

import numpy as np

from feets import extractors

from ..core import FeetsTestCase


# =============================================================================
# Test cases
# =============================================================================

class GSKewTest(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def test_gskew_linear_interpolation_problem(self):
        magnitude = [
            13.859, 13.854, 13.844, 13.881, 13.837, 13.885, 13.865, 13.9,
            13.819, 13.889, 13.89, 13.831, 13.869, 13.893, 13.825, 13.844,
            13.862, 13.853, 13.844, 13.85, 13.843, 13.839, 13.885, 13.859,
            13.865, 13.867, 13.874, 13.906, 13.819, 13.854, 13.891, 13.896,
            13.847, 13.862, 13.827, 13.849, 13.881, 13.871, 13.862, 13.846,
            13.865, 13.837, 13.819, 13.867, 13.833, 13.88, 13.868, 13.819,
            13.846, 13.842, 13.9, 13.88, 13.851, 13.885, 13.898, 13.824, 13.83,
            13.865, 13.823, 13.845, 13.874]

        lc = {
            "magnitude": np.array(magnitude),
            "time": np.arange(len(magnitude)),
            "error": self.random.rand(len(magnitude))}

        with warnings.catch_warnings():  # this launch mean of empty
            warnings.filterwarnings('ignore')

            ext = extractors.Gskew(interpolation="linear")
            result = ext.extract(features={}, **lc)
            assert np.isnan(result["Gskew"])

        ext = extractors.Gskew()  # by default interpolation is nearest
        result = ext.extract(features={}, **lc)
        assert not np.isnan(result["Gskew"])

        ext = extractors.Gskew(interpolation="nearest")
        result = ext.extract(features={}, **lc)

        assert not np.isnan(result["Gskew"])
