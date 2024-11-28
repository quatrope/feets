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

from light_curve import PercentAmplitude as _PercentAmplitude

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentAmplitude(LightCurveExtractor):
    features = ["PercentAmplitude"]

    def __init__(self, percent_amplitude_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if percent_amplitude_kwds is None
            else percent_amplitude_kwds
        )
        self.lightcurve_ext = _PercentAmplitude(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [percent_amplitude] = self.lightcurve_ext(time, magnitude, error)
        return {"PercentAmplitude": percent_amplitude}
