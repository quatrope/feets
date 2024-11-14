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

from light_curve import LinearTrend

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_LinearTrend(Extractor):
    features = [
        "lightcurve_LinearTrend",
        "lightcurve_LinearTrend_sigma",
        "lightcurve_LinearTrend_noise",
    ]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time = np.array(time, dtype=np.float64)
        magnitude = np.array(magnitude, dtype=np.float64)

        feature = LinearTrend(**self.lightcurve_kwds)
        [slope, error, noise_level] = feature(time, magnitude)

        return {
            "lightcurve_LinearTrend": slope,
            "lightcurve_LinearTrend_sigma": error,
            "lightcurve_LinearTrend_noise": noise_level,
        }
