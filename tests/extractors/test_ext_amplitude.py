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

"""feets.extractors.ext_amplitude Tests"""

# =============================================================================
# IMPORTS
# =============================================================================


from feets import extractors

from matplotlib.testing.decorators import check_figures_equal

import numpy as np


# =============================================================================
# Test cases
# =============================================================================


@check_figures_equal()
def test_plot_Amplitude(fig_test, fig_ref):

    magnitude = [1, 2, 3, 4]
    fvalue = 00.1

    # fig test
    test_ax = fig_test.subplots()
    ext = extractors.Amplitude()
    kwargs = ext.get_default_params()
    kwargs.update(
        feature="Amplitude",
        value=fvalue,
        ax=test_ax,
        plot_kws={},
        features={},
        magnitude=magnitude,
    )
    ext.plot(**kwargs)

    # expected
    min5p, max5p = ext._median_min_max_5p(magnitude)
    sample_idx = range(len(magnitude))
    msample = np.mean(sample_idx)

    exp_ax = fig_ref.subplots()
    exp_ax.plot(magnitude, marker=".", ls="", label="Sample")
    amplitude_line = exp_ax.plot(
        [msample, msample],
        [max5p, min5p],
        marker="o",
        ls="-",
        label="Amplitude",
    )[0]
    exp_ax.fill_between(
        sample_idx,
        min5p,
        max5p,
        alpha=0.15,
        color=amplitude_line.get_color(),
        label="_no_legend_",
    )

    exp_ax.set_ylabel("Magnitude")
    exp_ax.set_xlabel("Sample Index")
    exp_ax.set_title(f"Amplitude={fvalue:.4f}")
    exp_ax.legend(loc="best")
    exp_ax.invert_yaxis()
