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

from light_curve import InterPercentileRange

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.25, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveInterPercentileRange(LightCurveExtractor):
    lc_feature_ext = InterPercentileRange
    features = ["lc_inter_percentile_range"]

    def __init__(self, inter_percentile_range_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if inter_percentile_range_kwds is None
            else inter_percentile_range_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [inter_percentile_range] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_inter_percentile_range": inter_percentile_range}
