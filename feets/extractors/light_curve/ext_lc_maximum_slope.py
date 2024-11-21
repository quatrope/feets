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

from .utils import preprocess_data
from ..extractor import Extractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMaximumSlope(Extractor):
    lc_feature_ext = MaximumSlope
    features = ["lc_maximum_slope"]

    def __init__(self, maximum_slope_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if maximum_slope_kwds is None
            else maximum_slope_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time, magnitude, sigma = preprocess_data(
            time=time, magnitude=magnitude
        )

        [maximum_slope] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_maximum_slope": maximum_slope}
