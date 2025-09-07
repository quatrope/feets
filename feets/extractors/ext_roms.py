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

"""Roms extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Roms as _Roms

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Roms(LightCurveExtractor):
    r"""Robust median statistic.

    .. math::

        \text{Roms} =
          \frac{1}{N-1} \sum_{i=0}^{N-1}
          \frac{|m_i - \mathrm{median}(m_i)|}{\sigma_i}

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
    .. [enoch2003photometric] Enoch, M. L., Brown, M. E., & Burgasser, A. J.
       (2003). Photometric variability at the L/T dwarf boundary.
       The Astronomical Journal, 126(2), 1006.
    """

    features = ["Roms"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _Roms(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [roms] = self._extract(time, magnitude, error)
        return {"Roms": roms}
