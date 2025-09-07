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

"""Reduced chi-squared extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import ReducedChi2 as _ReducedChi2

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class ReducedChi2(LightCurveExtractor):
    r"""Reduced chi-squared of magnitude measurements.

    .. math::

        \mathrm{reduced~}\chi^2
          = \frac1{N-1} \sum_i\left(\frac{m_i - \bar{m}}{\delta\_i}\right)^2

    where :math:`N` is the number of observations, and :math:`\bar{m}` is the
    weighted mean magnitude.

    This is a good measure of variability which takes into account observations
    uncertainties.

    Parameters
    ----------
    transform : str or bool or None, optional
        Transformer to apply to the feature values. If str, must be one of:

        - 'default' - use default transformer for the feature, it same as
          giving True. The default for this feature is 'ln1p'
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

    features = ["ReducedChi2"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _ReducedChi2(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [chi2] = self._extract(time, magnitude, error)
        return {"ReducedChi2": chi2}
