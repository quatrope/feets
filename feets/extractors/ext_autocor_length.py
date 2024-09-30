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

from statsmodels.tsa import stattools

from .extractor import Extractor


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

    features = ["Autocor_length"]

    def __init__(self, nlags=100):
        self.nlags = nlags

    def extract(self, magnitude):
        nlags = self.nlags

        AC = stattools.acf(magnitude, nlags=nlags)
        k = next(
            (index for index, value in enumerate(AC) if value < np.exp(-1)),
            None,
        )

        while k is None:
            nlags = nlags + 100
            AC = stattools.acf(magnitude, nlags=nlags)
            k = next(
                (
                    index
                    for index, value in enumerate(AC)
                    if value < np.exp(-1)
                ),
                None,
            )

        return {"Autocor_length": k}
