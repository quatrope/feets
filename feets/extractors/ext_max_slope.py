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

import numpy as np

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MaxSlope(Extractor):
    """
    **MaxSlope**

    Maximum absolute magnitude slope between two consecutive observations.

    Examining successive (time-sorted) magnitudes, the maximal first difference
    (value of delta magnitude over delta time)

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['MaxSlope'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'MaxSlope': 5.4943105823904741}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.


    """

    features = ["MaxSlope"]

    def __init__(self, timesort=True):
        self.timesort = timesort

    def extract(self, magnitude, time):
        if self.timesort:
            sort = np.argsort(time)
            time, magnitude = time[sort], magnitude[sort]

        slope = np.abs(magnitude[1:] - magnitude[:-1]) / (time[1:] - time[:-1])
        return {"MaxSlope": np.max(slope)}
