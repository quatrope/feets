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

from light_curve import AndersonDarlingNormal

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


class LightCurveAndersonDarlingNormal(Extractor):
    lc_feature_ext = AndersonDarlingNormal
    features = {"lc_anderson_darling_normal"}

    def __init__(self, anderson_darling_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if anderson_darling_kwds is None
            else anderson_darling_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time, magnitude, sigma = preprocess_data(magnitude=magnitude)

        [anderson_darling_normal] = self.lc_feature_ext(
            **self.lightcurve_kwds
        )(time, magnitude, sigma)

        return {"lc_anderson_darling_normal": anderson_darling_normal}
