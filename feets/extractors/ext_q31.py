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

"""Q31 related extractors."""


# =============================================================================
# IMPORTS
# =============================================================================

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Q31(Extractor):
    r"""Q31 extractor.

    **Q31** (:math:`Q_{3-1}`)

    :math:`Q_{3-1}` is the difference between the third quartile, :math:`Q_3`,
    and the first quartile, :math:`Q_1`, of a raw light curve.
    :math:`Q_1` is a split between the lowest 25% and the highest 75% of data.
    :math:`Q_3` is a split between the lowest 75% and the highest 25% of data.

    References
    ----------
    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['Q31'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Q31': np.float64(1.3329778116209337)}
    """

    features = ["Q31"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        """
        Parameters
        ----------
        magnitude : array-like
        """
        q31 = np.percentile(magnitude, 75) - np.percentile(magnitude, 25)
        return {"Q31": q31}


class Q31Color(Extractor):
    r"""Q31 color extractor.

    **Q31_color** (:math:`Q_{3-1|B-R}`)

    :math:`Q_{3-1}` applied to the difference between both bands of a light
    curve (B-R)

    References
    ----------
    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.

    Examples
    --------
    >>> fs = feets.FeatureSpace(only=['Q31_color'])
    >>> features = fs.extract(**lc_normal)
    >>> features[0]
    {'Q31_color': 1.9517477838539978}
    """

    features = ["Q31_color"]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, aligned_magnitude, aligned_magnitude2):
        """
        Parameters
        ----------
        aligned_magnitude : array-like
        aligned_magnitude2 : array-like
        """
        N = len(aligned_magnitude)
        b_r = aligned_magnitude[:N] - aligned_magnitude2[:N]
        q31_color = np.percentile(b_r, 75) - np.percentile(b_r, 25)
        return {"Q31_color": q31_color}
