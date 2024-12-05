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


from light_curve import Periodogram as _Periodogram

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveLombScargle(LightCurveExtractor):
    features = ["LightCurve_PeriodLS", "Period_s_to_n"]

    def __init__(
        self,
        peaks=3,
        resolution=10,
        max_freq_factor=1,
        nyquist="average",
        fast=True,
    ):
        self.peaks = peaks
        self.resolution = resolution
        self.max_freq_factor = max_freq_factor
        self.nyquist = nyquist
        self.fast = fast

        self._extract = _Periodogram(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        periodogram = self._extract(time, magnitude, error)
        transpose = np.reshape(periodogram, (-1, 2))
        [period, period_s_to_n] = np.transpose(transpose)

        return {
            "LightCurve_PeriodLS": period,
            "Period_s_to_n": period_s_to_n,
        }
