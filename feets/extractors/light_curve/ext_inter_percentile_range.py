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

"""Inter-percentile range extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import InterPercentileRange as _InterPercentileRange

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class InterPercentileRange(LightCurveExtractor):
    features = ["InterPercentileRange"]

    def __init__(self, quantile=0.25, transform="identity"):
        self.quantile = quantile
        self.transform = transform

        self._extract = _InterPercentileRange(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [inter_percentile_range] = self._extract(time, magnitude, error)
        return {"InterPercentileRange": inter_percentile_range}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "InterPercentileRange":
            [name] = self._extract.names
            percentile = name.split("_")[3]
            return {f"InterPercentileRange_{percentile}": value}

        return super().flatten_feature(feature, value)
