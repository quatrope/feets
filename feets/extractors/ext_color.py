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


class Color(Extractor):
    """
    **Color**

    The color is defined as the difference between the average magnitude of
    two different bands observations.

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Color'])
        >>> features, values = fs.extract(**lc)
        >>> dict(zip(features, values))
        {'Color': -0.33325502453332145}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    features = ["Color"]

    def __init__(self):
        pass

    def extract(self, magnitude, magnitude2):
        return {"Color": np.mean(magnitude) - np.mean(magnitude2)}
