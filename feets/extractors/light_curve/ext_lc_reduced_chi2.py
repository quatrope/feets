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

from light_curve import ReducedChi2

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


class LightCurveReducedChi2(Extractor):
    lc_feature_ext = ReducedChi2
    features = ["lc_chi2"]

    def __init__(self, reduced_chi2_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if reduced_chi2_kwds is None
            else reduced_chi2_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, error):
        time, magnitude, sigma = preprocess_data(
            magnitude=magnitude, error=error
        )

        [chi2] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_chi2": chi2}
