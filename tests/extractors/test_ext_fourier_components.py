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

"""feets.extractors.ext_fourier_components Tests"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from feets import extractors


from ..core import FeetsTestCase


# =============================================================================
# Test cases
# =============================================================================

class FourierTests(FeetsTestCase):

    def setUp(self):
        self.random = np.random.RandomState(42)

    def periodic_lc(self):
        N = 100
        mjd_periodic = np.arange(N)
        Period = 20
        cov = np.zeros([N, N])
        mean = np.zeros(N)
        for i in np.arange(N):
            for j in np.arange(N):
                cov[i, j] = np.exp(-(np.sin((np.pi / Period) * (i - j)) ** 2))
        data_periodic = self.random.multivariate_normal(mean, cov)
        error = self.random.normal(size=100, loc=0.001)
        lc = {
            "magnitude": data_periodic,
            "time": mjd_periodic,
            "error": error}
        return lc

    def test_fourier_optional_data(self):
        lc_error = self.periodic_lc()

        lc = lc_error.copy()
        lc["error"] = None

        ext = extractors.FourierComponents()

        self.assertNotEqual(
            ext.extract(features={}, **lc),
            ext.extract(features={}, **lc_error))
