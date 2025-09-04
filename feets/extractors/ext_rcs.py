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

"""Range of cumulative sum extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class RCS(Extractor):
    r"""Range of cumulative sum extractor.

    **Rcs** - Range of cumulative sum (:math:`R_{cs}`)

    :math:`R_{cs}` is the range of a cumulative sum (Ellaway 1978) of each
    light-curve and is defined as:

    .. math::

        R_{cs} = max(S) - min(S) \\
        S = \frac{1}{N \sigma} \sum_{i=1}^l (m_i - \bar{m})

    where :math:`max`(:math:`min`) is the maximum (minimum) value of :math:`S`
    and :math:`l=1,2, \dots, N`.

    :math:`R_{cs}` should take a value close to zero for any symmetric
    distribution.

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
    >>> fs = feets.FeatureSpace(only=['Rcs'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Rcs': np.float64(0.04951776697391974)}
    """

    features = ["Rcs"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array-like
        """
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return {"Rcs": R}
