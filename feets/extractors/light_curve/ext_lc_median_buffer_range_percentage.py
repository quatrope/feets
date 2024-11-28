#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE

# =============================================================================
# IMPORTS
# =============================================================================

import copy

from light_curve import MedianBufferRangePercentage

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.10, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMedianBufferRangePercentage(LightCurveExtractor):
    lc_feature_ext = MedianBufferRangePercentage
    features = ["lc_median_buffer_range_percentage"]

    def __init__(self, median_buffer_range_percentage_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if median_buffer_range_percentage_kwds is None
            else median_buffer_range_percentage_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [median_buffer_range_percentage] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {
            "lc_median_buffer_range_percentage": median_buffer_range_percentage
        }
