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

import numpy as np

from feets import extractors

import matplotlib.pyplot as plt

from ..core import FeetsTestCase


# =============================================================================
# Test cases
# =============================================================================

class SignatureTests(FeetsTestCase):

    def test_plot_SignaturePhMag(self):
        ext = extractors.Signature()

        kwargs = ext.get_default_params()
        kwargs.update(
            feature="SignaturePhMag",
            value=[[1, 2, 3, 4]],
            ax=plt.gca(),
            plot_kws={},

            time=[1, 2, 3, 4],
            magnitude=[1, 2, 3, 4],
            error=[1, 2, 3, 4],

            features={"PeriodLS": 1, "Amplitude": 10})

        ext.plot(**kwargs)

    def test_multiple_peaks_period_ls(self):
        random = np.random.RandomState(54)

        lc = {
            "time": np.arange(100) + random.uniform(size=100),
            "magnitude": random.normal(size=100),
            "error": None}

        # one peak
        ls_ext_1 = extractors.LombScargle()
        ls_ext_2 = extractors.LombScargle(peaks=2)
        amp_ext = extractors.Amplitude()
        sig_ext = extractors.Signature()

        rs0 = ls_ext_1.extract(features={}, **lc)
        rs0.update(amp_ext.extract(features=rs0, **lc))
        rs0.update(sig_ext.extract(features=rs0, **lc))

        rs1 = ls_ext_2.extract(features={}, **lc)
        rs1.update(amp_ext.extract(features=rs1, **lc))
        rs1.update(sig_ext.extract(features=rs1, **lc))

        assert np.all(rs0["PeriodLS"][0] == rs1["PeriodLS"][0])
        assert np.all(rs0["Amplitude"] == rs1["Amplitude"])
        assert np.all(rs0["SignaturePhMag"] == rs1["SignaturePhMag"])
