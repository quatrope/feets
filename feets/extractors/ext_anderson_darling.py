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

"""Anderson-Darling extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import AndersonDarlingNormal as _AndersonDarlingNormal

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AndersonDarling(LightCurveExtractor):
    r"""Unbiased Anderson-Darling normality test statistic.

    **AndersonDarling** (:math:`A^2`)

    .. math::

        A^2 = & \left(1 + \frac{4}{N} - \frac{25}{N^2}\right) \cdot \\
              & \left(-N - \frac{1}{N} \sum_{i=0}^{N-1} (2i + 1)\ln\Phi_i +
              (2(N - i) - 1)\ln(1 - \Phi_i)\right)

    where :math:`\Phi_i = \Phi((m_i - \langle m \rangle) / \sigma_m)` is the
    standard cumulative distribution, :math:`N` is the number of
    observations, :math:`\langle m \rangle` is the mean magnitude and
    :math:`\sigma_m` is the magnitude standard deviation:

    .. math::

        \sigma_m = \sqrt{\frac{\sum_i (m_i - \langle m \rangle)^2}{N-1}}

    Parameters
    ----------
    transform : str or bool or None, optional
        Transformer to apply to the feature values. If str, must be one of:

        - 'default' - use default transformer for the feature, it same as
          giving True. The default for this feature is 'lg'
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

    features = ["AndersonDarling"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _AndersonDarlingNormal(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [anderson_darling_normal] = self._extract(time, magnitude, error)
        return {"AndersonDarling": anderson_darling_normal}
