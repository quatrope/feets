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

from light_curve import Periodogram as _Periodogram

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {
    "peaks": 3,
    "resolution": 10,
    "max_freq_factor": 1,
    "nyquist": "average",
    "fast": True,
}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveLombScargle(LightCurveExtractor):
    features = ["LightCurve_PeriodLS", "Period_s_to_n"]

    def __init__(self, light_curve_lomb_scargle_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if light_curve_lomb_scargle_kwds is None
            else light_curve_lomb_scargle_kwds
        )
        self.lightcurve_ext = _Periodogram(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        periodogram = self.lightcurve_ext(time, magnitude, error)
        transpose = np.reshape(periodogram, (-1, 2))
        [period, period_s_to_n] = np.transpose(transpose)

        return {
            "LightCurve_PeriodLS": period,
            "Period_s_to_n": period_s_to_n,
        }
