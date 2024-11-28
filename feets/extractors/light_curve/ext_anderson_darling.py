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

from light_curve import AndersonDarlingNormal as _AndersonDarlingNormal

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AndersonDarling(LightCurveExtractor):
    features = ["AndersonDarling"]

    def __init__(self, anderson_darling_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if anderson_darling_kwds is None
            else anderson_darling_kwds
        )
        self.lightcurve_ext = _AndersonDarlingNormal(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [anderson_darling_normal] = self.lightcurve_ext(time, magnitude, error)
        return {"AndersonDarling": anderson_darling_normal}
