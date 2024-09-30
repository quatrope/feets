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

from scipy import stats

from .extractor import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearTrend(Extractor):
    r"""
    **LinearTrend**

    Slope of a linear fit to the light-curve.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['LinearTrend'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'LinearTrend': -3.2084065290292509e-06}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["LinearTrend"]

    def __init__(self):
        pass

    def extract(self, magnitude, time):
        regression_slope = stats.linregress(time, magnitude)[0]
        return {"LinearTrend": regression_slope}
