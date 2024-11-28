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

from light_curve import Roms as _Roms

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Roms(LightCurveExtractor):
    features = ["Roms"]

    def __init__(self, roms_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS) if roms_kwds is None else roms_kwds
        )
        self.lightcurve_ext = _Roms(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [roms] = self.lightcurve_ext(time, magnitude, error)
        return {"Roms": roms}
