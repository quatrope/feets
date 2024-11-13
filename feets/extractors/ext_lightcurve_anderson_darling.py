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

from light_curve import AndersonDarlingNormal

import numpy as np

from .extractor import Extractor
from ..libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_AndersonDarlingNormal(Extractor):
    features = {"lightcurve_AndersonDarlingNormal"}

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        magnitude = np.array(magnitude, dtype=np.float64)
        time = np.zeros_like(magnitude)

        feature = AndersonDarlingNormal(**self.lightcurve_kwds)
        [result] = feature(time, magnitude)

        # TODO: Check if this is the correct formula
        # result = 1 / (1.0 + np.exp(-10 * (ander - 0.3)))

        return {"lightcurve_AndersonDarlingNormal": result}
