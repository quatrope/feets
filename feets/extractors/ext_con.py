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


class Con(Extractor):
    r"""

    **Con**

    Index introduced for the selection of variable stars from the OGLE
    database (Wozniak 2000). To calculate Con, we count the number of three
    consecutive data points that are brighter or fainter than :math:`2\sigma`
    and normalize the number by :math:`N−2`.

    For a normal distribution and by considering just one star, Con should
    take values close to 0.045:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Con'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Con': 0.0476}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    features = ["Con"]

    def __init__(self, consecutiveStar=3):
        self.consecutiveStar = consecutiveStar

    def extract(self, magnitude):
        consecutiveStar = self.consecutiveStar

        N = len(magnitude)
        if N < consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - consecutiveStar + 1):
            flag = 0
            for j in range(consecutiveStar):
                if (
                    magnitude[i + j] > m + 2 * sigma
                    or magnitude[i + j] < m - 2 * sigma
                ):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1
        return {"Con": count * 1.0 / (N - consecutiveStar + 1)}
