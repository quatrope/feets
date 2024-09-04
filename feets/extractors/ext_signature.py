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
    features = ["Signature"]

    def __init__(self, phase_bins=18, mag_bins=12):
        feature_attrs = []
        for i in range(mag_bins):
            for j in range(phase_bins):
                feature_attrs.append(f"ph_{j}_mag_{i}")

        self.phase_bins = phase_bins
        self.mag_bins = mag_bins
        self.feature_attrs = tuple(feature_attrs)

    def extract(self, magnitude, time, PeriodLS, Amplitude):
        phase_bins, mag_bins = self.phase_bins, self.mag_bins

        lc_yaxis = (magnitude - np.min(magnitude)) / np.float64(Amplitude)

        # SHIFT TO BEGIN AT MINIMUM
        loc = np.argmin(lc_yaxis)

        signatures = np.full(len(PeriodLS), None, dtype=object)
        for idx, period_ls in enumerate(PeriodLS):
            lc_phases = np.remainder(time - time[loc], period_ls) / period_ls

            bins = (phase_bins, mag_bins)

            count = np.histogram2d(
                lc_phases, lc_yaxis, bins=bins, density=True
            )[0]

            signature = zip(
                self.feature_attrs, count.reshape(phase_bins * mag_bins)
            )

            signatures[idx] = dict(signature)

        return {"Signature": signatures}
