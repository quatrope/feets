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

from statsmodels.tsa import stattools

from .core import Extractor


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================

class AutocorLength(Extractor):
    r"""
    **Autocor_length**

    The autocorrelation, also known as serial correlation, is the
    cross-correlation of a signal with itself. Informally, it is the similarity
    between observations as a function of the time lag between them. It is a
    mathematical tool for finding repeating patterns, such as the presence of
    a periodic signal obscured by noise, or identifying the missing fundamental
    frequency in a signal implied by its harmonic frequencies.

    For an observed series :math:`y_1, y_2,\dots,y_T`  with sample mean
    :math:`\bar{y}`, the sample lag :math:`-h` autocorrelation is given by:

    .. math::

       \rho_h = \frac{\sum_{t=h+1}^T (y_t - \bar{y})(y_{t-h}-\bar{y})}
                            {\sum_{t=1}^T (y_t - \bar{y})^2}

    Since the autocorrelation fuction of a light curve is given by a vector and
    we can only return one value as a feature, we define the length of the
    autocorrelation function where its value is smaller than  :math:`e^{-1}` .

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
    features = ['Autocor_length']
    params = {"nlags": 100}

    def fit(self, magnitude, nlags):

        AC = stattools.acf(magnitude, nlags=nlags, fft=False)
        k = next((index for index, value in
                 enumerate(AC) if value < np.exp(-1)), None)

        while k is None:
            nlags = nlags + 100
            AC = stattools.acf(magnitude, nlags=nlags, fft=False)
            k = next((index for index, value in
                      enumerate(AC) if value < np.exp(-1)), None)

        return {'Autocor_length': k}
