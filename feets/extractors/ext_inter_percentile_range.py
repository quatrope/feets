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

"""Inter-percentile range extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import InterPercentileRange as _InterPercentileRange

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class InterPercentileRange(LightCurveExtractor):
    """Inter-percentile range.

    .. math::

        Q(1 - p) - Q(p)

    where :math:`Q(p)` is the :math:`p`-th quantile of the magnitude
    distribution.

    Special cases are the interquartile range which is inter-percentile range
    for :math:`p = 0.25`, and the interdecile range, which is inter-percentile
    range for :math:`p = 0.1`.

    Parameters
    ----------
    quantile : positive float, default=0.25
        Range is
        :math:`(100%% * quantile, 100%% * (1 - quantile))`.
        Default quantile is 0.25
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
    """

    features = ["InterPercentileRange"]

    def __init__(self, quantile=0.25, transform=None):
        self.quantile = quantile
        self.transform = transform

        self._extract = _InterPercentileRange(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [inter_percentile_range] = self._extract(time, magnitude, error)
        return {"InterPercentileRange": inter_percentile_range}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "InterPercentileRange":
            [name] = self._extract.names
            percentile = name.split("_")[3]
            return {f"InterPercentileRange_{percentile}": value}

        return super().flatten_feature(feature, value)
