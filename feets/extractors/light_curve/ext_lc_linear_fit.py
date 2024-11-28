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

from light_curve import LinearFit

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveLinearFit(LightCurveExtractor):
    lc_feature_ext = LinearFit
    features = [
        "lc_linear_fit_slope",
        "lc_linear_fit_slope_sigma",
        "lc_linear_fit_reduced_chi2",
    ]

    def __init__(self, linear_fit_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if linear_fit_kwds is None
            else linear_fit_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error):
        [slope, slope_sigma, reduced_chi2] = LinearFit(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {
            "lc_linear_fit_slope": slope,
            "lc_linear_fit_slope_sigma": slope_sigma,
            "lc_linear_fit_reduced_chi2": reduced_chi2,
        }
