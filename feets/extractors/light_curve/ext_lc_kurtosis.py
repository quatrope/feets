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

from light_curve import Kurtosis

from .lc_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveKurtosis(LightCurveExtractor):
    lc_feature_ext = Kurtosis
    features = ["lc_kurtosis"]

    def __init__(self, kurtosis_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if kurtosis_kwds is None
            else kurtosis_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [lc_kurtosis] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, error
        )

        return {"lc_kurtosis": lc_kurtosis}
