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

from light_curve import BeyondNStd

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"nstd": 1, "transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveBeyondNStd(LightCurveExtractor):
    lc_feature_ext = BeyondNStd
    features = ["lc_beyond_n_std"]

    def __init__(self, beyond_n_std_wkds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if beyond_n_std_wkds is None
            else beyond_n_std_wkds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [beyond_n_std] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_beyond_n_std": beyond_n_std}
