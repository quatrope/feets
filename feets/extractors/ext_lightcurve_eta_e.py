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

from light_curve import EtaE

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_EtaE(Extractor):
    features = ["lightcurve_EtaE"]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time = np.array(time, dtype=np.float64)
        magnitude = np.array(magnitude, dtype=np.float64)

        feature = EtaE(**self.lightcurve_kwds)
        [result] = feature(time, magnitude)

        return {"lightcurve_EtaE": result}
