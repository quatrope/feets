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

"""Median absolute deviation extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import MedianAbsoluteDeviation as _MedianAbsoluteDeviation

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianAbsDev(LightCurveExtractor):
    r"""Median absolute deviation.

    Median of the absolute value of the difference between magnitude and its
    median.

    .. math::

      \mathrm{Median}\left(|m_i - \mathrm{Median}(m)|\right)

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
    .. [disanto2016feature] D'Isanto, A., Cavuoti, S., Brescia, M., Donalek,
       C., Longo, G., Riccio, G., & Djorgovski, S. G. (2016).
       An analysis of feature relevance in the classification of astronomical
       transients with machine learning methods.
       Monthly Notices of the Royal Astronomical Society, 457(3), 3119-3132.
    """

    features = ["MedianAbsDev"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _MedianAbsoluteDeviation(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [median_absolute_deviation] = self._extract(time, magnitude, error)
        return {"MedianAbsDev": median_absolute_deviation}
