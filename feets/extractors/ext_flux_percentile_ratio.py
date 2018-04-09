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

import math

import numpy as np

from .core import Extractor


# =============================================================================
# CONSTANTS
# =============================================================================

COMMON_DOC = r"""
In order to caracterize the sorted magnitudes distribution we use percentiles.
If :math:`F_{5, 95}` is the difference between 95% and 5% magnitude values,
we calculate the following:

- flux_percentile_ratio_mid20: ratio :math:`F_{40, 60}/F_{5, 95}`
- flux_percentile_ratio_mid35: ratio :math:`F_{32.5, 67.5}/F_{5, 95}`
- flux_percentile_ratio_mid50: ratio :math:`F_{25, 75}/F_{5, 95}`
- flux_percentile_ratio_mid65: ratio :math:`F_{17.5, 82.5}/F_{5, 95}`
- flux_percentile_ratio_mid80: ratio :math:`F_{10, 90}/F_{5, 95}`

For the first feature for example, in the case of a normal distribution, this
is equivalente to calculate:

.. math::

    \frac{erf^{-1}(2 \cdot 0.6-1)-erf^{-1}(2 \cdot 0.4-1)}
         {erf^{-1}(2 \cdot 0.95-1)-erf^{-1}(2 \cdot 0.05-1)}


So, the expected values for each of the flux percentile features are:

- flux_percentile_ratio_mid20 = 0.154
- flux_percentile_ratio_mid35 = 0.275
- flux_percentile_ratio_mid50 = 0.410
- flux_percentile_ratio_mid65 = 0.568
- flux_percentile_ratio_mid80 = 0.779

References
----------

.. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
   Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
   Rischard, M. (2011). On machine-learned classification of variable stars
   with sparse and noisy time-series data.
   The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.


"""


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class FluxPercentileRatioMid20(Extractor):
    __doc__ = COMMON_DOC

    data = ['magnitude']
    features = ["FluxPercentileRatioMid20"]

    def fit(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_60_index = int(math.ceil(0.60 * lc_length))
        F_40_index = int(math.ceil(0.40 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_40_60 = sorted_data[F_60_index] - sorted_data[F_40_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid20 = F_40_60 / F_5_95

        return {"FluxPercentileRatioMid20": F_mid20}


class FluxPercentileRatioMid35(Extractor):
    __doc__ = COMMON_DOC

    data = ['magnitude']
    features = ["FluxPercentileRatioMid35"]

    def fit(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_325_index = int(math.ceil(0.325 * lc_length))
        F_675_index = int(math.ceil(0.675 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_325_675 = sorted_data[F_675_index] - sorted_data[F_325_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid35 = F_325_675 / F_5_95

        return {"FluxPercentileRatioMid35": F_mid35}


class FluxPercentileRatioMid50(Extractor):
    __doc__ = COMMON_DOC

    data = ['magnitude']
    features = ["FluxPercentileRatioMid50"]

    def fit(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_25_index = int(math.ceil(0.25 * lc_length))
        F_75_index = int(math.ceil(0.75 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_25_75 = sorted_data[F_75_index] - sorted_data[F_25_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid50 = F_25_75 / F_5_95

        return {"FluxPercentileRatioMid50": F_mid50}


class FluxPercentileRatioMid65(Extractor):
    __doc__ = COMMON_DOC

    data = ['magnitude']
    features = ["FluxPercentileRatioMid65"]

    def fit(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_175_index = int(math.ceil(0.175 * lc_length))
        F_825_index = int(math.ceil(0.825 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_175_825 = sorted_data[F_825_index] - sorted_data[F_175_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid65 = F_175_825 / F_5_95

        return {"FluxPercentileRatioMid65": F_mid65}


class FluxPercentileRatioMid80(Extractor):
    __doc__ = COMMON_DOC

    data = ['magnitude']
    features = ["FluxPercentileRatioMid80"]

    def fit(self, magnitude):
        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)

        F_10_index = int(math.ceil(0.10 * lc_length))
        F_90_index = int(math.ceil(0.90 * lc_length))
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))

        F_10_90 = sorted_data[F_90_index] - sorted_data[F_10_index]
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]
        F_mid80 = F_10_90 / F_5_95

        return {"FluxPercentileRatioMid80": F_mid80}
