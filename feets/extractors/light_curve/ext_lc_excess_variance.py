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

from light_curve import ExcessVariance

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


class LightCurveEta(Extractor):
    lc_feature_ext = ExcessVariance
    features = ["lc_excess_variance"]

    def __init__(self, excess_variance_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if excess_variance_kwds is None
            else excess_variance_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, error):
        time, magnitude, sigma = preprocess_data(
            magnitude=magnitude, error=error
        )

        [excess_variance] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_excess_variance": excess_variance}
