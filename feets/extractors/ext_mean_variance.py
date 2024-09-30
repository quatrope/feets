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


class MeanVariance(Extractor):
    r"""
    **Meanvariance** (:math:`\frac{\sigma}{\bar{m}}`)

    This is a simple variability index and is defined as the ratio of the
    standard deviation :math:`\sigma`, to the mean magnitude, :math:`\bar{m}`.
    If a light curve has strong  variability, :math:`\frac{\sigma}{\bar{m}}`
    of the light curve is generally  large.

    For a uniform distribution from 0 to 1, the mean is equal to 0.5 and the
    variance is equal to 1/12, thus the mean-variance should take a value
    close to 0.577:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Meanvariance'])
        >>> features, values = fs.extract(**lc_uniform)
        >>> dict(zip(features, values))
        {'Meanvariance': 0.5816791217381897}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    features = ["Meanvariance"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        return {"Meanvariance": np.std(magnitude) / np.mean(magnitude)}
