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


class MedianAbsDev(Extractor):
    r"""

    **MedianAbsDev**

    The median absolute deviation is defined as the median discrepancy of the
    data from the median data:

    .. math::

        Median Absolute Deviation = median(|mag - median(mag)|)

    It should take a value close to 0.675 for a normal distribution:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['MedianAbsDev'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'MedianAbsDev': 0.66332131466690614}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["MedianAbsDev"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        median = np.median(magnitude)
        devs = abs(magnitude - median)
        return {"MedianAbsDev": np.median(devs)}
