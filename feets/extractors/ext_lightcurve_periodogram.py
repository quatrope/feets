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

from light_curve import Periodogram

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_Periodogram(Extractor):
    features = [
        "lightcurve_Periodogram_period",
        "lightcurve_Periodogram_period_s_to_n",
    ]

    def __init__(
        self,
        peaks=1,
        resolution=10,
        max_freq_factor=1,
        nyquist="average",
        fast=True,
        lightcurve_kwds=None,
    ):
        self.peaks = peaks
        self.resolution = resolution
        self.max_freq_factor = max_freq_factor
        self.nyquist = nyquist
        self.fast = fast
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time = np.array(time, dtype=np.float64)
        magnitude = np.array(magnitude, dtype=np.float64)

        feature = Periodogram(
            peaks=self.peaks,
            resolution=self.resolution,
            max_freq_factor=self.max_freq_factor,
            nyquist=self.nyquist,
            fast=self.fast,
            **self.lightcurve_kwds,
        )

        results = np.array(feature(time, magnitude))
        transpose = np.reshape(results, (-1, 2))
        [period, period_s_to_n] = np.transpose(transpose)

        return {
            "lightcurve_Periodogram_period": period,
            "lightcurve_Periodogram_period_s_to_n": period_s_to_n,
        }
