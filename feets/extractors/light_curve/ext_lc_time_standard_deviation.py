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

from light_curve import TimeStandardDeviation

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


class LightCurveTimeStandardDeviation(Extractor):
    lc_feature_ext = TimeStandardDeviation
    features = ["lc_time_standard_deviation"]

    def __init__(self, time_standard_deviation_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if time_standard_deviation_kwds is None
            else time_standard_deviation_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time):
        time, magnitude, sigma = preprocess_data(time=time)

        [time_standard_deviation] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, sigma)

        return {"lc_time_standard_deviation": time_standard_deviation}
