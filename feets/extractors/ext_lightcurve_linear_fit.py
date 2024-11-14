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

from light_curve import LinearFit

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_LinearFit(Extractor):
    features = [
        "lightcurve_LinearFit_slope",
        "lightcurve_LinearFit_slope_sigma",
        "lightcurve_LinearFit_reduced_chi2",
    ]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude, error):
        time = np.array(time, dtype=np.float64)
        magnitude = np.array(magnitude, dtype=np.float64)
        error = np.array(error, dtype=np.float64)

        feature = LinearFit(**self.lightcurve_kwds)
        [slope, error, reduced_chi2] = feature(time, magnitude, error)

        return {
            "lightcurve_LinearFit_slope": slope,
            "lightcurve_LinearFit_slope_sigma": error,
            "lightcurve_LinearFit_reduced_chi2": reduced_chi2,
        }
