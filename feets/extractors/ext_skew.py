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

"""Skew extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Skew as _Skew

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Skew(LightCurveExtractor):
    r"""Skewness of the magnitude distribution.

    .. math::

        G_1 = \frac{N}{(N - 1)(N - 2)}
        \frac{\sum_i(m_i - \langle m \rangle)^3}{\sigma_m^3}

    where :math:`N` is the number of observations, :math:`\langle m \rangle`
    is the mean magnitude,
    :math:`\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}` is the
    magnitude standard deviation.

    Parameters
    ----------
    transform : str or bool or None, optional
        Transformer to apply to the feature values. If str, must be one of:

        - 'default' - use default transformer for the feature, it same as
          giving True. The default for this feature is 'arcsinh'
        - 'arcsinh' - Hyperbolic arcsine feature transformer
        - 'clipped_lg' - Decimal logarithm of a value clipped to a minimum
          value
        - 'identity' - Identity feature transformer
        - 'lg' - Decimal logarithm feature transformer
        - 'ln1p' - :math:`ln(1+x)` feature transformer
        - 'sqrt' - Square root feature transformer

        If bool, must be True to use default transformer or False to disable.
        If None, no transformation is applied.
    """

    features = ["Skew"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _Skew(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [skew] = self._extract(time, magnitude, error)
        return {"Skew": skew}
