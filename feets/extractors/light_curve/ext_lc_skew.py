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

from light_curve import Skew

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


class LightCurveSkew(Extractor):
    lc_feature_ext = Skew
    features = ["lc_skew"]

    def __init__(self, skew_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS) if skew_kwds is None else skew_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time, magnitude, sigma = preprocess_data(magnitude=magnitude)

        [skew] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_skew": skew}
