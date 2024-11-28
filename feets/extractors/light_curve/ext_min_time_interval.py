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

from light_curve import MinimumTimeInterval as _MinimumTimeInterval

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MinTimeInterval(LightCurveExtractor):
    features = ["MinTimeInterval"]

    def __init__(self, minimum_time_interval_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if minimum_time_interval_kwds is None
            else minimum_time_interval_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        [minimum_time_interval] = _MinimumTimeInterval(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"MinTimeInterval": minimum_time_interval}
