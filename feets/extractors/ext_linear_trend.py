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

"""Linear trend extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

from scipy import stats

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearTrend(Extractor):
    r"""Linear trend extractor.

    **LinearTrend**

    Slope of a linear fit to the light-curve.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['LinearTrend'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'LinearTrend': np.float64(-0.008096282859385663)}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["LinearTrend"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, time):
        regression_slope = stats.linregress(time, magnitude)[0]
        return {"LinearTrend": regression_slope}
