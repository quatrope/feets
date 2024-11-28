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

from light_curve import MaximumTimeInterval as _MaximumTimeInterval

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MaxTimeInterval(LightCurveExtractor):
    features = ["MaxTimeInterval"]

    def __init__(self, max_time_interval_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if max_time_interval_kwds is None
            else max_time_interval_kwds
        )
        self.lightcurve_ext = _MaximumTimeInterval(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        [maximum_time_interval] = self.lightcurve_ext(time, magnitude, error)
        return {"MaxTimeInterval": maximum_time_interval}
