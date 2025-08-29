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

"""Median absolute deviation extractor."""

# =============================================================================
# IMPORTS
# =============================================================================


from light_curve import MedianAbsoluteDeviation as _MedianAbsoluteDeviation

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianAbsDev(LightCurveExtractor):
    features = ["MedianAbsDev"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _MedianAbsoluteDeviation(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [median_absolute_deviation] = self._extract(time, magnitude, error)
        return {"MedianAbsDev": median_absolute_deviation}
