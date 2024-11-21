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


class LightCurveTimeMean(Extractor):
    lc_feature_ext = TimeMean
    features = ["lc_time_mean"]

    def __init__(self, time_mean_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if time_mean_kwds is None
            else time_mean_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time):
        time, magnitude, sigma = preprocess_data(time=time)

        [time_mean] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_time_mean": time_mean}
