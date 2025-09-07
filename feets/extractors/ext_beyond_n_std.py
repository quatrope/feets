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

"""Beyond-N-standard-deviation extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import BeyondNStd as _BeyondNStd

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class BeyondNStd(LightCurveExtractor):
    r"""Beyond-N-standard-deviation extractor.

    **BeyondNStd**

    Fraction of observations beyond :math:`N\,\sigma_m` from the mean
    magnitude :math:`\langle m \rangle`.

    .. math::

        \mathrm{beyond}~n\,\sigma_m = \frac{
            \sum_i I_{|m - \langle m \rangle| > n\,\sigma_m}(m_i)
        }{N}

    where :math:`I` is the indicator function, :math:`N` is the number of
    observations, :math:`\langle m \rangle` is the mean magnitude and
    :math:`\sigma_m = \sqrt{\sum_i (m_i - \langle m \rangle)^2 / (N-1)}`
    is the magnitude standard deviation.

    Parameters
    ----------
    nstd : positive float, default=1
        N, default is 1.0
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
    feets.extractors.WeightedBeyondNStd
    """

    features = ["BeyondNStd"]

    def __init__(self, nstd=1, transform=None):
        self.nstd = nstd
        self.transform = transform

        self._extract = _BeyondNStd(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [beyond_n_std] = self._extract(time, magnitude, error)
        return {"BeyondNStd": beyond_n_std}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "BeyondNStd":
            N = self.nstd
            return {f"Beyond{N}Std": value}

        return super().flatten_feature(feature, value)
