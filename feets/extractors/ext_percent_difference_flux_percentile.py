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


class PercentDifferenceFluxPercentile(Extractor):
    r"""
    **PercentDifferenceFluxPercentile**

    Ratio of :math:`F_{5, 95}` over the median magnitude.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['PercentDifferenceFluxPercentile'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'PercentDifferenceFluxPercentile': -134.93590403825007}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["PercentDifferenceFluxPercentile"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        median_data = np.median(magnitude)

        sorted_data = np.sort(magnitude)
        lc_length = len(sorted_data)
        F_5_index = int(math.ceil(0.05 * lc_length))
        F_95_index = int(math.ceil(0.95 * lc_length))
        F_5_95 = sorted_data[F_95_index] - sorted_data[F_5_index]

        percent_difference = F_5_95 / median_data

        return {"PercentDifferenceFluxPercentile": percent_difference}
