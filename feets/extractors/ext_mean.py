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

"""Mean extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Mean as _Mean

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Mean(LightCurveExtractor):
    r"""Mean magnitude.

    .. math::

       \frac1{N} \sum_i m_i.

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
    feets.extractors.WeightedMean
    """

    features = ["Mean"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _Mean(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [mean] = self._extract(time, magnitude, error)
        return {"Mean": mean}
