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

from light_curve import PercentDifferenceMagnitudePercentile

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.05, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurvePercentDifferenceMagnitudePercentile(LightCurveExtractor):
    lc_feature_ext = PercentDifferenceMagnitudePercentile
    features = ["lc_percent_difference_magnitude_percentile"]

    def __init__(self, percent_difference_magnitude_percentile_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if percent_difference_magnitude_percentile_kwds is None
            else percent_difference_magnitude_percentile_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [percent_difference_magnitude_percentile] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {
            "lc_percent_difference_magnitude_percentile": percent_difference_magnitude_percentile
        }
