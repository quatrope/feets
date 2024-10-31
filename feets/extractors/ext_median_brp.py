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

"""Median buffer range percentage extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianBRP(Extractor):
    r"""Median buffer range percentage extractor.

    **MedianBRP** (Median buffer range percentage)

    Fraction (:math:`\leq 1`) of photometric points within
    :math:`\frac{amplitude}{10}` of the median magnitude

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["MedianBRP"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        median = np.median(magnitude)
        amplitude = (np.max(magnitude) - np.min(magnitude)) / 10
        n = len(magnitude)

        count = np.sum(
            np.logical_and(
                magnitude < median + amplitude, magnitude > median - amplitude
            )
        )

        median_brp = float(count) / n
        return {"MedianBRP": median_brp}
