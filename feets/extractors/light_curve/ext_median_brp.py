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

"""Median buffer range percentage extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import (
    MedianBufferRangePercentage as _MedianBufferRangePercentage,
)

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianBRP(LightCurveExtractor):
    features = ["MedianBRP"]

    def __init__(self, quantile=0.10, transform="identity"):
        self.quantile = quantile
        self.transform = transform

        self._extract = _MedianBufferRangePercentage(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [median_brp] = self._extract(time, magnitude, error)
        return {"MedianBRP": median_brp}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "MedianBRP":
            [name] = self._extract.names
            percentile = name.split("_")[4]
            return {f"MedianBRP_{percentile}": value}

        return super().flatten_feature(feature, value)
