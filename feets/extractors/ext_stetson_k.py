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

"""Stetson K extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import StetsonK as _StetsonK

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StetsonK(LightCurveExtractor):
    r"""Stetson K coefficient described light curve shape.

    .. math::

        \mathrm{Stetson}~K =
            \frac{
                \sum_i\left|\frac{m_i - \bar{m}
            }{
                \delta_i}\right|}{\sqrt{N\,\chi^2}
            }

    where :math:`N` is the number of observations, :math:`\bar{m}` is the
    weighted mean magnitude, and

    .. math::

        \chi^2 =
            \sum_i\left(
                \frac{m_i - \langle m \rangle}{\delta_i}\right
            )^2

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
    .. [stetson1996automatic] Stetson, P. B. (1996). On the Automatic
       Determination of Light-Curve Parameters for Cepheid Variables.
       Publications of the Astronomical Society of the Pacific, 108(728),
       851-876. http://www.jstor.org/stable/40680814
    """

    features = ["StetsonK"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _StetsonK(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [stetson_k] = self._extract(time, magnitude, error)
        return {"StetsonK": stetson_k}
