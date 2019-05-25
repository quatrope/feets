#!/usr/bin/env python
# -*- coding: utf-8 -*-

# The MIT License (MIT)

# Copyright (c) 2017 Juan Cabral

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# =============================================================================
# DOC
# =============================================================================

""""""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .core import Extractor


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
        >>> rs = fs.extract(**lc_normal)
        >>> rs.as_dict()
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
    data = ['magnitude']
    features = ["Con"]
    params = {"consecutiveStar": 3}

    def fit(self, magnitude, consecutiveStar):

        N = len(magnitude)
        if N < consecutiveStar:
            return 0
        sigma = np.std(magnitude)
        m = np.mean(magnitude)
        count = 0

        for i in range(N - consecutiveStar + 1):
            flag = 0
            for j in range(consecutiveStar):
                if(magnitude[i + j] > m + 2 * sigma or
                   magnitude[i + j] < m - 2 * sigma):
                    flag = 1
                else:
                    flag = 0
                    break
            if flag:
                count = count + 1

        con = count * 1.0 / (N - consecutiveStar + 1)
        return {"Con": con}
