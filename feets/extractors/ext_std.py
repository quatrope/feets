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


class Std(Extractor):
    r"""
    **Std** - Standard deviation of the magnitudes


    The standard deviation :math:`\sigma` of the sample is defined as:

    .. math::

        \sigma=\frac{1}{N-1}\sum_{i} (y_{i}-\hat{y})^2

    For example, a white noise time serie should have :math:`\sigma=1`

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Std'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Std': 0.99320419310116881}

    References
    ----------

    .. [richards2011machine] Richards, J. W., Starr, D. L., Butler, N. R.,
       Bloom, J. S., Brewer, J. M., Crellin-Quick, A., ... &
       Rischard, M. (2011). On machine-learned classification of variable stars
       with sparse and noisy time-series data.
       The Astrophysical Journal, 733(1), 10. Doi:10.1088/0004-637X/733/1/10.

    """

    features = ["Std"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        return {"Std": np.std(magnitude)}
