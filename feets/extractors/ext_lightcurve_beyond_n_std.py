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

from light_curve import BeyondNStd

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_BeyondNStd(Extractor):
    features = ["lightcurve_BeyondNStd"]

    def __init__(self, nstd=1, lightcurve_kwds=None):
        self.nstd = nstd
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        magnitude = np.array(magnitude, dtype=np.float64)
        time = np.zeros_like(magnitude)

        feature = BeyondNStd(nstd=self.nstd, **self.lightcurve_kwds)
        [result] = feature(time, magnitude)

        return {"lightcurve_BeyondNStd": result}
