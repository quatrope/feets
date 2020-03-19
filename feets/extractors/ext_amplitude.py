#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017-2020 Juan Cabral

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

import math

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class Amplitude(Extractor):
    """
    **Amplitude**

    The amplitude is defined as the half of the difference between the median
    of the maximum 5% and the median of the minimum 5% magnitudes. For a
    sequence of numbers from 0 to 1000 the amplitude should be equal to 475.5.

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    data = ['magnitude']
    features = ['Amplitude']

    def _median_min_max_5p(self, magnitude):
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        max5p = np.median(sorted_mag[-int(math.ceil(0.05 * N)):])
        min5p = np.median(sorted_mag[0:int(math.ceil(0.05 * N))])

        return min5p, max5p

    def plot_feature(self, feature, value, ax, plot_kws, magnitude, **kwargs):
        # Code the plot here

        # retrieve the amplitude limits
        min5p, max5p = self._median_min_max_5p(magnitude)

        # plot all the magnitudes
        plot_magnitude_kws = plot_kws.get("plot_magnitude_kws", {})
        plot_magnitude_kws.setdefault("marker", ".")
        plot_magnitude_kws.setdefault("ls", "")
        plot_magnitude_kws.setdefault("label", "Sample")

        ax.plot(magnitude, **plot_magnitude_kws)

        # plot the magnitude bar in the middle of the axis
        plot_ampb_kws = plot_kws.get("plot_amplitude_bar_kws", {})
        plot_ampb_kws.setdefault("marker", "o")
        plot_ampb_kws.setdefault("ls", "-")
        plot_ampb_kws.setdefault("label", "Amplitude")

        sample_idx = range(len(magnitude))
        msample = np.mean(sample_idx)

        amplitude_line = ax.plot(
            [msample, msample], [max5p, min5p], **plot_ampb_kws)[0]

        # fill between the amplitude limits
        plot_ampf_kws = plot_kws.get("plot_amplitude_fill_kws", {})
        plot_ampf_kws.setdefault("alpha", 0.15)
        plot_ampf_kws.setdefault("color", amplitude_line.get_color())
        plot_ampf_kws.setdefault("label", "_no_legend_")
        ax.fill_between(sample_idx, min5p, max5p, **plot_ampf_kws)

        ax.set_ylabel("Magnitude")
        ax.set_xlabel("Sample Index")

        ax.set_title(f"Amplitude={value:.4f}")
        ax.legend(loc="best")

        ax.invert_yaxis()

    def fit(self, magnitude):
        # retrieve the amplitude limits
        min5p, max5p = self._median_min_max_5p(magnitude)

        amplitude = (max5p - min5p) / 2.0
        return {"Amplitude": amplitude}
