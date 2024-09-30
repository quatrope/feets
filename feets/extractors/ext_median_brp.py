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


class MedianBRP(Extractor):
    r"""
    **MedianBRP** (Median buffer range percentage)

    Fraction (<= 1) of photometric points within amplitude/10
    of the median magnitude

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['MedianBRP'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'MedianBRP': 0.559}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["MedianBRP"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        median = np.median(magnitude)
        amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
        n = len(magnitude)

        count = np.sum(
            np.logical_and(
                magnitude < median + amplitude, magnitude > median - amplitude
            )
        )

        return {"MedianBRP": float(count) / n}
