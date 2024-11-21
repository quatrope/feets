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

from light_curve import Periodogram

import numpy as np

from .utils import preprocess_data
from ..extractor import Extractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {
    "peaks": 1,
    "resolution": 10,
    "max_freq_factor": 1,
    "nyquist": "average",
    "fast": True,
}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurvePeriodogram(Extractor):
    lc_feature_ext = Periodogram
    features = ["lc_period", "lc_period_s_to_n"]

    def __init__(self, periodogram_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if periodogram_kwds is None
            else periodogram_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time, magnitude, sigma = preprocess_data(
            time=time, magnitude=magnitude
        )

        periodogram = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )
        transpose = np.reshape(periodogram, (-1, 2))
        [period, period_s_to_n] = np.transpose(transpose)

        return {"lc_period": period, "lc_period_s_to_n": period_s_to_n}
