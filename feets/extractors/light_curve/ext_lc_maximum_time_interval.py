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


class LightCurveMaximumTimeInterval(Extractor):
    lc_feature_ext = MaximumTimeInterval
    features = ["lc_maximum_time_interval"]

    def __init__(self, maximum_time_interval_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if maximum_time_interval_kwds is None
            else maximum_time_interval_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time):
        time, magnitude, sigma = preprocess_data(time=time)

        [maximum_time_interval] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_maximum_time_interval": maximum_time_interval}
