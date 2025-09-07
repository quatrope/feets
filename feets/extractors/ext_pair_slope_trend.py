#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

"""Pair slope trend extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PairSlopeTrend(Extractor):
    r"""Pair slope trend extractor.

    **PairSlopeTrend**

    Considering the last :math:`30` (time-sorted) measurements of source
    magnitude, the fraction of increasing first differences minus the fraction
    of decreasing first differences.

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['PairSlopeTrend'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'PairSlopeTrend': -0.0021333333333333343}
    """

    features = ["PairSlopeTrend"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array-like
        """
        data_last = magnitude[-30:]

        pst = (
            float(
                len(np.where(np.diff(data_last) > 0)[0])
                - len(np.where(np.diff(data_last) <= 0)[0])
            )
            / 30
        )

        return {"PairSlopeTrend": pst}
