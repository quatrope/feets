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

"""Median absolute deviation extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianAbsDev(Extractor):
    r"""Median absolute deviation extractor.

    **MedianAbsDev**

    The median absolute deviation is defined as the median discrepancy of the
    data from the median data:

    .. math::

        Median Absolute Deviation = median(|mag - median(mag)|)

    It should take a value close to :math:`0.675` for a normal distribution:

    Examples
    --------
    Median absolute deviation of a normal time series:

    >>> fs = feets.FeatureSpace(only=['MedianAbsDev'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'MedianAbsDev': np.float64(0.6660195149475938)}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["MedianAbsDev"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        median = np.median(magnitude)
        devs = abs(magnitude - median)
        mean_abs_dev = np.median(devs)
        return {"MedianAbsDev": mean_abs_dev}
