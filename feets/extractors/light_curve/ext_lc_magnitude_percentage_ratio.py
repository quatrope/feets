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

from light_curve import MagnitudePercentageRatio

from .utils import preprocess_data
from ..extractor import Extractor
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


class LightCurveMagnitudePercentageRatio(Extractor):
    lc_feature_ext = MagnitudePercentageRatio
    features = ["lc_magnitude_percentage_ratio"]

    def __init__(self, magnitude_percentage_ratio_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if magnitude_percentage_ratio_kwds is None
            else magnitude_percentage_ratio_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time, magnitude, sigma = preprocess_data(magnitude=magnitude)

        [magnitude_percentage_ratio] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, sigma)

        return {"lc_magnitude_percentage_ratio": magnitude_percentage_ratio}
