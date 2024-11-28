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

from light_curve import Eta as _Eta

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Eta(LightCurveExtractor):
    features = ["Eta"]

    def __init__(self, eta_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS) if eta_kwds is None else eta_kwds
        )
        self.lightcurve_ext = _Eta(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [eta] = self.lightcurve_ext(time, magnitude, error)
        return {"Eta": eta}
