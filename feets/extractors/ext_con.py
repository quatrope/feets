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

"""Con extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Con(Extractor):
    r"""Con extractor.

    **Con**

    Index introduced for the selection of variable stars from the OGLE
    database (Wozniak 2000). To calculate Con, we count the number of three
    consecutive data points that are brighter or fainter than :math:`2\sigma`
    and normalize the number by :math:`N-2`.

    For a normal distribution and by considering just one star, Con should
    take values close to :math:`0.045`.

    Parameters
    ----------
    consecutive_star : int, optional (default=3)
        Number of consecutive data points to consider.

    References
    ----------
    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    Examples
    --------
    Con of a normal time series:

    >>> fs = feets.FeatureSpace(only=["Con"], consecutive_star=1)
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Con': 0.042}
    """

    features = ["Con"]

    def __init__(self, consecutive_star=3):
        self.consecutive_star = consecutive_star

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array_like
        """
        consecutive_star = self.consecutive_star

        N = len(magnitude)
        if N < consecutive_star:
            return {"Con": 0}
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - consecutive_star + 1):
            flag = 0
            for j in range(consecutive_star):
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
        return {"Con": count * 1.0 / (N - consecutive_star + 1)}
