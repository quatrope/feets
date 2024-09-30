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


class Eta_e(Extractor):
    r"""

    **Eta_e** (:math:`\eta^e`)

    Variability index :math:`\eta` is the ratio of the mean of the square of
    successive differences to the variance of data points. The index was
    originally proposed to check whether the successive data points are
    independent or not. In other words, the index was developed to check if
    any trends exist in the data (von Neumann 1941). It is defined as:

    .. math::

        \eta = \frac{1}{(N-1)\sigma^2}
            \sum_{i=1}^{N-1} (m_{i+1}-m_i)^2


    The variability index should take a value close to 2 for a normal
    distribution.

    Although :math:`\eta` is a powerful index for quantifying variability
    characteristics of a time series, it does not take into account unequal
    sampling. Thus :math:`\eta^r` is defined as:

    .. math::

        \eta^e = \bar{w} \, (t_{N-1} - t_1)^2
                    \frac{\sum_{i=1}^{N-1} w_i (m_{i+1} - m_i)^2}
                        {\sigma^2 \sum_{i=1}^{N-1} w_i}

    Where:

    .. math::

        w_i = \frac{1}{(t_{i+1} - t_i)^2}


    Example:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Eta_e'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Eta_e': 2.0028592616231866}


    References
    ----------

    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    """

    features = ["Eta_e"]

    def __init__(self):
        pass

    def extract(self, magnitude, time):
        w = 1.0 / np.power(np.subtract(time[1:], time[:-1]), 2)
        w_mean = np.mean(w)

        N = len(time)
        sigma2 = np.var(magnitude)

        S1 = sum(w * (magnitude[1:] - magnitude[:-1]) ** 2)
        S2 = sum(w)

        eta_e = (
            w_mean
            * np.power(time[N - 1] - time[0], 2)
            * S1
            / (sigma2 * S2 * N**2)
        )

        return {"Eta_e": eta_e}
