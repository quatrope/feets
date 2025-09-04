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

"""Median amplitude extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import math

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianAmplitude(Extractor):
    """Median amplitude extractor.

    **MedianAmplitude**

    This amplitude is defined as the half of the difference between the median
    of the maximum :math:`5%%` and the median of the minimum :math:`5%%`
    magnitudes. For a sequence of numbers from :math:`0` to :math:`1000` the
    amplitude should be equal to :math:`475.0`.

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    See Also
    --------
    feets.extractors.Amplitude

    Examples
    --------
    Median amplitude of increasing magnitudes from :math:`0` to :math:`1000`:

    >>> fs = feets.FeatureSpace(only=['MedianAmplitude'])
    >>> features = fs.extract(**lc_incremental)
    >>> features[0]
    {'MedianAmplitude': np.float64(475.0)}
    """

    features = ["MedianAmplitude"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array-like
        """
        N = len(magnitude)
        sorted_mag = np.sort(magnitude)

        amplitude = (
            np.median(sorted_mag[-int(math.ceil(0.05 * N)) :])
            - np.median(sorted_mag[0 : int(math.ceil(0.05 * N))])
        ) / 2.0
        return {"MedianAmplitude": amplitude}
