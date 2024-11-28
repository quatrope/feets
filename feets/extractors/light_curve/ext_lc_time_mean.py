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

from light_curve import TimeMean

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveTimeMean(LightCurveExtractor):
    lc_feature_ext = TimeMean
    features = ["lc_time_mean"]

    def __init__(self, time_mean_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if time_mean_kwds is None
            else time_mean_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        [time_mean] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_time_mean": time_mean}
