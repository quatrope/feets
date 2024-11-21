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

from light_curve import EtaE

from .utils import preprocess_data
from ..extractor import Extractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveEtaE(Extractor):
    lc_feature_ext = EtaE
    features = ["lc_eta_e"]

    def __init__(self, eta_e_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if eta_e_kwds is None
            else eta_e_kwds
        )

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, time, magnitude):
        time, magnitude, sigma = preprocess_data(
            time=time, magnitude=magnitude
        )

        [eta_e] = self.lc_feature_ext(**self.lightcurve_kwds)(
            time, magnitude, sigma
        )

        return {"lc_eta_e": eta_e}
