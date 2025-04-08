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


from light_curve import PercentAmplitude as _PercentAmplitude

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentAmplitude(LightCurveExtractor):
    features = ["PercentAmplitude"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _PercentAmplitude(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [percent_amplitude] = self._extract(time, magnitude, error)
        return {"PercentAmplitude": percent_amplitude}
