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

from .utils import preprocess_data
from ..extractor import Extractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.10, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMedianBufferRangePercentage(Extractor):
    lc_feature_ext = MedianBufferRangePercentage
    features = ["lc_median_buffer_range_percentage"]

    def __init__(self, median_buffer_range_percentage_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if median_buffer_range_percentage_kwds is None
            else median_buffer_range_percentage_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time, magnitude, sigma = preprocess_data(magnitude=magnitude)

        [median_buffer_range_percentage] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, sigma)

        return {
            "lc_median_buffer_range_percentage": median_buffer_range_percentage
        }
