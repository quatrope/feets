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

"""Weighted mean extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import WeightedMean as _WeightedMean

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class WeightedMean(LightCurveExtractor):
    r"""Weighted mean magnitude.

    **WeightedMean** (:math:`\bar{m}`)

    .. math::

        \bar{m} = \frac{\sum_i m_i / \delta_i^2}{\sum_i 1 / \delta_i^2}.

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

    See Also
    --------
    feets.extractors.Mean
    """

    features = ["WeightedMean"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _WeightedMean(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [weighted_mean] = self._extract(time, magnitude, error)
        return {"WeightedMean": weighted_mean}
