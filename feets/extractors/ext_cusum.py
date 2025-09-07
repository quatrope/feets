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

"""Cumulative sum (CUSUM) extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Cusum as _Cusum

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Cusum(LightCurveExtractor):
    r"""Cusum — a range of cumulative sums.

    **Cusum** = :math:`\max(S) - \min(S)`

    where :math:`S_j = \frac1{N\sigma_m} \sum_{i=0}^j{\left(m\_i -
    \langle m \rangle\right)}`, :math:`N` is the number of observations,
    :math:`\langle m \rangle` is the mean magnitude and
    :math:`\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}`
    is the magnitude standard deviation.

    Parameters
    ----------
    transform : str or bool or None, optional
        Transformer to apply to the feature values. If str, must be one of:

        - 'default' - use default transformer for the feature, it same as
          giving True. The default for this feature is 'identity'
        - 'arcsinh' - Hyperbolic arcsine feature transformer
        - 'clipped_lg' - Decimal logarithm of a value clipped to a minimum
          value
        - 'identity' - Identity feature transformer
        - 'lg' - Decimal logarithm feature transformer
        - 'ln1p' - :math:`ln(1+x)` feature transformer
        - 'sqrt' - Square root feature transformer

        If bool, must be True to use default transformer or False to disable.
        If None, no transformation is applied.

    References
    ----------
    .. [kim2014epoch] Kim, D. W., Protopapas, P., Bailer-Jones, C. A.,
       Byun, Y. I., Chang, S. W., Marquette, J. B., & Shin, M. S. (2014).
       The EPOCH Project: I. Periodic Variable Stars in the EROS-2 LMC
       Database. arXiv preprint Doi:10.1051/0004-6361/201323252.
    """

    features = ["Cusum"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _Cusum(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [cusum] = self._extract(time, magnitude, error)
        return {"Cusum": cusum}
