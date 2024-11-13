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

from light_curve import Duration

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_Duration(Extractor):
    features = ["lightcurve_Duration"]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time):
        time = np.array(time, dtype=np.float64)
        magnitude = np.zeros_like(time)

        feature = Duration(**self.lightcurve_kwds)
        [result] = feature(time, magnitude)

        return {"lightcurve_Duration": result}
