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

from light_curve import WeightedMean

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


class LightCurveWeightedMean(Extractor):
    lc_feature_ext = WeightedMean
    features = ["lc_weighted_mean"]

    def __init__(self, weighted_mean_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if weighted_mean_kwds is None
            else weighted_mean_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, error):
        time, magnitude, sigma = preprocess_data(
            magnitude=magnitude, error=error
        )

        [weighted_mean] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_weighted_mean": weighted_mean}
