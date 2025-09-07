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

"""Median buffer range percentage extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import (
    MedianBufferRangePercentage as _MedianBufferRangePercentage,
)

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianBRP(LightCurveExtractor):
    r"""Median buffer range percentage.

    Fraction of observations inside the
    :math:`\mathrm{Median}(m) \pm q \times (\max(m) - \min(m)) / 2` interval.

    Parameters
    ----------
    quantile : positive float, default=0.10
        Relative range size, default is 0.10

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

    features = ["MedianBRP"]

    def __init__(self, quantile=0.10, transform=None):
        self.quantile = quantile
        self.transform = transform

        self._extract = _MedianBufferRangePercentage(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [median_brp] = self._extract(time, magnitude, error)
        return {"MedianBRP": median_brp}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "MedianBRP":
            [name] = self._extract.names
            percentile = name.split("_")[4]
            return {f"MedianBRP_{percentile}": value}

        return super().flatten_feature(feature, value)
