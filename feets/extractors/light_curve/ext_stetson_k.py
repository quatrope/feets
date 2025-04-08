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

from light_curve import StetsonK as _StetsonK

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class StetsonK(LightCurveExtractor):
    features = ["StetsonK"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _StetsonK(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [stetson_k] = self._extract(time, magnitude, error)
        return {"StetsonK": stetson_k}
