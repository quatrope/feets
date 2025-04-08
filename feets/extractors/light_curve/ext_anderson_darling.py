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

from light_curve import AndersonDarlingNormal as _AndersonDarlingNormal

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class AndersonDarling(LightCurveExtractor):
    features = ["AndersonDarling"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _AndersonDarlingNormal(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [anderson_darling_normal] = self._extract(time, magnitude, error)
        return {"AndersonDarling": anderson_darling_normal}
