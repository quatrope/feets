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

""""""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class EtaColor(Extractor):
    r"""

    **Eta_color** (:math:`\eta_{color}`)

    Variability index Eta_e (:math:`\eta^e`)
    calculated from the color light-curve.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Eta_color'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Eta_color': 1.991749074648397}

    References
    ----------

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    data = ['aligned_magnitude', 'aligned_time', 'aligned_magnitude2']
    features = ["Eta_color"]

    def fit(self, aligned_magnitude, aligned_time, aligned_magnitude2):
        N = len(aligned_magnitude)
        B_Rdata = aligned_magnitude - aligned_magnitude2

        w = 1.0 / np.power(aligned_time[1:] - aligned_time[:-1], 2)
        w_mean = np.mean(w)

        N = len(aligned_time)
        sigma2 = np.var(B_Rdata)

        S1 = sum(w * (B_Rdata[1:] - B_Rdata[:-1]) ** 2)
        S2 = sum(w)

        eta_B_R = (w_mean * np.power(aligned_time[N - 1] -
                   aligned_time[0], 2) * S1 / (sigma2 * S2 * N ** 2))

        return {"Eta_color": eta_B_R}
