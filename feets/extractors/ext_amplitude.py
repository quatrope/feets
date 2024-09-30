#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

__doc__ = """"""


# =============================================================================
# IMPORTS
# =============================================================================

import math

import numpy as np

from .extractor import Extractor


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

    features = ["Amplitude"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        amplitude = (
            np.median(sorted_mag[-int(math.ceil(0.05 * N)) :])
            - np.median(sorted_mag[0 : int(math.ceil(0.05 * N))])
        ) / 2.0
        return {"Amplitude": amplitude}
