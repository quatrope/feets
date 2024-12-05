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

from light_curve import ReducedChi2 as _ReducedChi2

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class ReducedChi2(LightCurveExtractor):
    features = ["ReducedChi2"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _ReducedChi2(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [chi2] = self._extract(time, magnitude, error)
        return {"ReducedChi2": chi2}
