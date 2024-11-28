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

from light_curve import StetsonK

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveStetsonK(LightCurveExtractor):
    lc_feature_ext = StetsonK
    features = ["lc_stetson_k"]

    def __init__(self, stetson_k_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if stetson_k_kwds is None
            else stetson_k_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [stetson_k] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_stetson_k": stetson_k}
