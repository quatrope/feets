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

""""""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

import seaborn as sns

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class Signature(Extractor):

    data = ['magnitude', 'time']
    dependencies = ['PeriodLS', 'Amplitude']
    params = {"phase_bins": 18, "mag_bins": 12}

    features = ["SignaturePhMag"]

    def plot_feature(self, feature, value, ax, plot_kws,
                     phase_bins, mag_bins, **kwargs):

        ax.set_title(f"SignaturePhMag - {phase_bins}x{mag_bins}")
        ax.set_xlabel("Phase")
        ax.set_ylabel("Magnitude")
        sns.heatmap(value, ax=ax, **plot_kws)

    def fit(self, magnitude, time, PeriodLS, Amplitude, phase_bins, mag_bins):
        first_period = PeriodLS[0]
        lc_yaxis = (magnitude - np.min(magnitude)) / np.float(Amplitude)

        # SHIFT TO BEGIN AT MINIMUM
        loc = np.argmin(lc_yaxis)
        lc_phase = np.remainder(time - time[loc], first_period) / first_period

        bins = (phase_bins, mag_bins)
        signature = np.histogram2d(
            lc_phase, lc_yaxis, bins=bins, normed=True)[0]

        return {"SignaturePhMag": signature}
