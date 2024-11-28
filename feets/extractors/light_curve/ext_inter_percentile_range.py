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

from light_curve import InterPercentileRange as _InterPercentileRange

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.25, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class InterPercentileRange(LightCurveExtractor):
    features = ["InterPercentileRange"]

    def __init__(self, inter_percentile_range_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if inter_percentile_range_kwds is None
            else inter_percentile_range_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [inter_percentile_range] = _InterPercentileRange(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {"InterPercentileRange": inter_percentile_range}
