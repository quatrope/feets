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

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Q31(Extractor):
    r"""
    **Q31** (:math:`Q_{3-1}`)

    :math:`Q_{3-1}` is the difference between the third quartile, :math:`Q_3`,
    and the first quartile, :math:`Q_1`, of a raw light curve.
    :math:`Q_1` is a split between the lowest 25% and the highest 75% of data.
    :math:`Q_3` is a split between the lowest 75% and the highest 25% of data.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Q31'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Q31': 1.3320376563134508}

    References
    ----------

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    features = ["Q31"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        q31 = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
        return {"Q31": q31}


class Q31Color(Extractor):
    r"""
    **Q31_color** (:math:`Q_{3-1|B-R}`)

    :math:`Q_{3-1}` applied to the difference between both bands of a light
    curve (B-R).

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Q31_color'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Q31_color': 1.8840489594535512}

    References
    ----------

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    features = ["Q31_color"]

    def __init__(self):
        pass

    def extract(self, aligned_magnitude, aligned_magnitude2):
        N = len(aligned_magnitude)
        b_r = aligned_magnitude[:N] - aligned_magnitude2[:N]
        q31_color = np.percentile(b_r, 75) - np.percentile(b_r, 25)
        return {"Q31_color": q31_color}
