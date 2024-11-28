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

from light_curve import MaximumSlope

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMaximumSlope(LightCurveExtractor):
    lc_feature_ext = MaximumSlope
    features = ["lc_maximum_slope"]

    def __init__(self, maximum_slope_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if maximum_slope_kwds is None
            else maximum_slope_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        [maximum_slope] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_maximum_slope": maximum_slope}
