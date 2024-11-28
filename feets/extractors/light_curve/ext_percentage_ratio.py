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

from light_curve import MagnitudePercentageRatio as _MagnitudePercentageRatio

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {
    "quantile_numerator": 0.40,
    "quantile_denominator": 0.05,
    "transform": "default",
}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentageRatio(LightCurveExtractor):
    features = ["PercentageRatio"]

    def __init__(self, magnitude_percentage_ratio_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if magnitude_percentage_ratio_kwds is None
            else magnitude_percentage_ratio_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [magnitude_percentage_ratio] = _MagnitudePercentageRatio(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {"PercentageRatio": magnitude_percentage_ratio}
