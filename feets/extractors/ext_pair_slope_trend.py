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


class PairSlopeTrend(Extractor):
    r"""
    **PairSlopeTrend**

    Considering the last 30 (time-sorted) measurements of source magnitude,
    the fraction of increasing first differences minus the fraction of
    decreasing first differences.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['PairSlopeTrend'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'PairSlopeTrend': -0.16666666666666666}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["PairSlopeTrend"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        data_last = magnitude[-30:]

        pst = (
            float(
                len(np.where(np.diff(data_last) > 0)[0])
                - len(np.where(np.diff(data_last) <= 0)[0])
            )
            / 30
        )

        return {"PairSlopeTrend": pst}
