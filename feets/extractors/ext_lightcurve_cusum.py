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

from light_curve import Cusum

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_Cusum(Extractor):
    features = ["lightcurve_Cusum"]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        magnitude = np.array(magnitude, dtype=np.float64)
        time = np.arange(len(magnitude), dtype=np.float64)

        feature = Cusum(**self.lightcurve_kwds)
        [result] = feature(time, magnitude)

        return {"lightcurve_Cusum": result}
