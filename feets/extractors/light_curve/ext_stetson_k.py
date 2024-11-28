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

from light_curve import StetsonK as _StetsonK

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StetsonK(LightCurveExtractor):
    features = ["StetsonK"]

    def __init__(self, stetson_k_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if stetson_k_kwds is None
            else stetson_k_kwds
        )
        self.lightcurve_ext = _StetsonK(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [stetson_k] = self.lightcurve_ext(time, magnitude, error)
        return {"StetsonK": stetson_k}
