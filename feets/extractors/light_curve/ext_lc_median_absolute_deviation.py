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

from light_curve import MedianAbsoluteDeviation

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMedianAbsoluteDeviation(LightCurveExtractor):
    lc_feature_ext = MedianAbsoluteDeviation
    features = ["lc_median_absolute_deviation"]

    def __init__(self, median_absolute_deviation_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if median_absolute_deviation_kwds is None
            else median_absolute_deviation_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [median_absolute_deviation] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {"lc_median_absolute_deviation": median_absolute_deviation}
