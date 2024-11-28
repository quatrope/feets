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

from light_curve import LinearFit as _LinearFit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearFit(LightCurveExtractor):
    features = [
        "LinearFitSlope",
        "LinearFitSigma",
        "LinearFitChi2",
    ]

    def __init__(self, linear_fit_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if linear_fit_kwds is None
            else linear_fit_kwds
        )
        self.lightcurve_ext = _LinearFit(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error):
        [slope, slope_sigma, reduced_chi2] = self.lightcurve_ext(
            time, magnitude, error
        )
        return {
            "LinearFitSlope": slope,
            "LinearFitSigma": slope_sigma,
            "LinearFitChi2": reduced_chi2,
        }
