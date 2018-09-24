#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2018 Bruno Sanchez

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
# FUTURE
# =============================================================================

from __future__ import unicode_literals


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class Signature(Extractor):

    data = ['magnitude', 'time']
    dependencies = ['PeriodLS', 'Amplitude']
    params = {"phase_bins": 18, "mag_bins": 12}

    features = []
    for i in range(params["phase_bins"]):
        for j in range(params["mag_bins"]):
            features.append("Signature_ph_{:02d}_mag_{:02d}".format(i, j))

    # this variable stores a sorted version of the features
    # because feets only stores a frozenset of the original features
    # for future validation.
    sorted_features = tuple(features)

    del i, j

    def fit(self, magnitude, time, PeriodLS, Amplitude, phase_bins, mag_bins):

        lc_yaxis = (magnitude - np.min(magnitude)) / np.float(Amplitude)

        # SHIFT TO BEGIN AT MINIMUM
        loc = np.argmin(lc_yaxis)
        lc_phase = np.remainder(time - time[loc], PeriodLS) / PeriodLS

        bins = (phase_bins, mag_bins)
        counts = np.histogram2d(lc_phase, lc_yaxis, bins=bins, normed=True)[0]

        result = zip(self.sorted_features,
                     counts.reshape(phase_bins * mag_bins))

        return dict(result)
