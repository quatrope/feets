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


class RCS(Extractor):
    r"""
    **Rcs** - Range of cumulative sum (:math:`R_{cs}`)

    :math:`R_{cs}` is the range of a cumulative sum (Ellaway 1978) of each
    light-curve and is defined as:

    .. math::

        R_{cs} = max(S) - min(S) \\
        S = \frac{1}{N \sigma} \sum_{i=1}^l (m_i - \bar{m})

    where max(min) is the maximum (minimum) value of S and
    :math:`l=1,2, \dots, N`.

    :math:`R_{cs}` should take a value close to zero for any symmetric
    distribution:

    .. code-block:: pycon

        >>> fs = feets.FeatureSpace(only=['Rcs'])
        >>> features, values = fs.extract(**lc_normal)
        >>> dict(zip(features, values))
        {'Rcs': 0.0094459606901065168}

    References
    ----------

    .. [kim2011quasi] Kim, D. W., Protopapas, P., Byun, Y. I., Alcock, C.,
       Khardon, R., & Trichas, M. (2011). Quasi-stellar object selection
       algorithm using time variability and machine learning: Selection of
       1620 quasi-stellar object candidates from MACHO Large Magellanic Cloud
       database. The Astrophysical Journal, 735(2), 68.
       Doi:10.1088/0004-637X/735/2/68.

    """

    features = ["Rcs"]

    def __init__(self):
        pass

    def extract(self, magnitude):
        sigma = np.std(magnitude)
        N = len(magnitude)
        m = np.mean(magnitude)
        s = np.cumsum(magnitude - m) * 1.0 / (N * sigma)
        R = np.max(s) - np.min(s)
        return {"Rcs": R}
