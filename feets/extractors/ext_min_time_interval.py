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

"""Minimum time interval extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import MinimumTimeInterval as _MinimumTimeInterval

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MinTimeInterval(LightCurveExtractor):
    r"""Minimum time interval between consequent observations.

    .. math::

        \min{(t_{i+1} - t_i)}

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
    feets.extractors.MaxTimeInterval
    """

    features = ["MinTimeInterval"]

    def __init__(self, transform=None):
        self.transform = transform
        self._extract = _MinimumTimeInterval(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        """
        Parameters
        ----------
        time : array-like
        magnitude : array-like, optional
        error : array-like, optional
        """
        [minimum_time_interval] = self._extract(time, magnitude, error)
        return {"MinTimeInterval": minimum_time_interval}
