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

from light_curve import LinearTrend as _LinearTrend

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearTrend(LightCurveExtractor):
    features = [
        "LinearTrend",
        "LinearTrend_Sigma",
        "LinearTrend_ReducedChi2",
    ]

    def __init__(self, linear_trend_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if linear_trend_kwds is None
            else linear_trend_kwds
        )
        self.lightcurve_ext = _LinearTrend(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        [linear_trend, linear_trend_sigma, reduced_chi2] = self.lightcurve_ext(
            time, magnitude, error
        )
        return {
            "LinearTrend": linear_trend,
            "LinearTrend_Sigma": linear_trend_sigma,
            "LinearTrend_ReducedChi2": reduced_chi2,
        }
