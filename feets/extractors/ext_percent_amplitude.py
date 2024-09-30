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


class PercentAmplitude(Extractor):
    r"""
    **PercentAmplitude**

    Largest percentage difference between either the max or min magnitude
    and the median.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['PercentAmplitude'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'PercentAmplitude': -168.991253993057}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["PercentAmplitude"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        median_data = np.median(magnitude)
        distance_median = np.abs(magnitude - median_data)
        max_distance = np.max(distance_median)

        percent_amplitude = max_distance / median_data

        return {"PercentAmplitude": percent_amplitude}
