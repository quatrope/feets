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

from light_curve import ReducedChi2

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_ReducedChi2(Extractor):
    features = [
        "lightcurve_ReducedChi2",
    ]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude, error):
        magnitude = np.array(magnitude, dtype=np.float64)
        error = np.array(error, dtype=np.float64)
        time = np.zeros_like(magnitude)

        feature = ReducedChi2(**self.lightcurve_kwds)

        [result] = feature(time, magnitude, error)

        return {
            "lightcurve_ReducedChi2": result,
        }
