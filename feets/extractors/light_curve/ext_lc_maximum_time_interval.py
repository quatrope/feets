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

from light_curve import MaximumTimeInterval

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveMaximumTimeInterval(LightCurveExtractor):
    lc_feature_ext = MaximumTimeInterval
    features = ["lc_maximum_time_interval"]

    def __init__(self, maximum_time_interval_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if maximum_time_interval_kwds is None
            else maximum_time_interval_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        [maximum_time_interval] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_maximum_time_interval": maximum_time_interval}
