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

from light_curve import LinearTrend

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveLinearTrend(LightCurveExtractor):
    lc_feature_ext = LinearTrend
    features = [
        "lc_linear_trend",
        "lc_linear_trend_sigma",
        "lc_linear_trend_reduced_chi2",
    ]

    def __init__(self, linear_trend_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if linear_trend_kwds is None
            else linear_trend_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        [linear_trend, linear_trend_sigma, reduced_chi2] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {
            "lc_linear_trend": linear_trend,
            "lc_linear_trend_sigma": linear_trend_sigma,
            "lc_linear_trend_reduced_chi2": reduced_chi2,
        }
