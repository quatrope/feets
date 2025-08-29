#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
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
from ...libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MinTimeInterval(LightCurveExtractor):
    features = ["MinTimeInterval"]

    def __init__(self, transform="identity"):
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
