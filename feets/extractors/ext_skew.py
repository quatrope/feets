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

"""Skewness extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

from scipy import stats

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Skew(Extractor):
    r"""Skewness extractor.

    **Skew**

    The skewness of a sample is defined as follow:

    .. math::

        Skewness = \frac{N}{(N-1)(N-2)}
            \sum_{i=1}^N (\frac{m_i-\hat{m}}{\sigma})^3

    For a normal distribution it should be equal to zero.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=["Skew"])
    >>> features = fs.extract(**lc)
    >>> features[0]
    {'Skew': np.float64(-0.047893540889984265)}

    References
    ----------
    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.
    """

    features = ["Skew"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        return {"Skew": stats.skew(magnitude)}
