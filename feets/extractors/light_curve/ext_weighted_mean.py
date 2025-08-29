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

"""Weighted mean extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import WeightedMean as _WeightedMean

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class WeightedMean(LightCurveExtractor):
    features = ["WeightedMean"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _WeightedMean(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [weighted_mean] = self._extract(time, magnitude, error)
        return {"WeightedMean": weighted_mean}
