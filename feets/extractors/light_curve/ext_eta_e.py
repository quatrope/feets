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

from light_curve import EtaE as _EtaE

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class EtaE(LightCurveExtractor):
    features = ["EtaE"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _EtaE(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        [eta_e] = self._extract(time, magnitude, error)
        return {"EtaE": eta_e}
