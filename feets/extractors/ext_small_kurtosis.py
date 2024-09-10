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

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class SmallKurtosis(Extractor):
    r"""

    **SmallKurtosis**

    Small sample kurtosis of the magnitudes.

    .. math::

        SmallKurtosis = \frac{N (N+1)}{(N-1)(N-2)(N-3)}
            \sum_{i=1}^N (\frac{m_i-\hat{m}}{\sigma})^4 -
            \frac{3( N-1 )^2}{(N-2) (N-3)}

    For a normal distribution, the small kurtosis should be zero:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['SmallKurtosis'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'SmallKurtosis': 0.044451779515607193}

    See http://www.xycoon.com/peakedness_small_sample_test_1.htm

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.


    """

    features = ["SmallKurtosis"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        n = len(magnitude)
        mean = np.mean(magnitude)
        std = np.std(magnitude)

        S = sum(((magnitude - mean) / std) ** 4)

        c1 = float(n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3))
        c2 = float(3 * (n - 1) ** 2) / ((n - 2) * (n - 3))

        return {"SmallKurtosis": c1 * S - c2}
