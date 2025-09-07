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

"""Percentage ratio extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import (
    MagnitudePercentageRatio as _MagnitudePercentageRatio,
)

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentageRatio(LightCurveExtractor):
    r"""Magnitude percentage ratio.

    .. math::

        \mathrm{magnitude~}q\mathrm{~to~}n\mathrm{~ratio}
        = \frac{Q(1-n) - Q(n)}{Q(1-d) - Q(d)}

    where :math:`n` and :math:`d` denotes user defined percentage, :math:`Q` is
    the quantile function of magnitude distribution.

    Parameters
    ----------
    quantile_numerator: positive float, default=0.40
        Numerator is inter-percentile range
        :math:`(100%% * q, 100%% (1 - q))`. Default value is 0.40
    quantile_denominator: positive float, default=0.05
        Denominator is inter-percentile range
        :math:`(100%% * q, 100%% (1 - q))`. Default value is 0.05
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

    features = ["PercentageRatio"]

    def __init__(
        self,
        quantile_numerator=0.40,
        quantile_denominator=0.05,
        transform=None,
    ):
        self.quantile_numerator = quantile_numerator
        self.quantile_denominator = quantile_denominator
        self.transform = transform

        self._extract = _MagnitudePercentageRatio(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [percentage_ratio] = self._extract(time, magnitude, error)
        return {"PercentageRatio": percentage_ratio}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "PercentageRatio":
            [name] = self._extract.names
            split_name = name.split("_")
            numerator, denominator = split_name[3], split_name[4]
            return {f"PercentageRatio_{numerator}_{denominator}": value}

        return super().flatten_feature(feature, value)
