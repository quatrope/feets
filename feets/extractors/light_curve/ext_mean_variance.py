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

from light_curve import MeanVariance as _MeanVariance

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MeanVariance(LightCurveExtractor):
    features = ["MeanVariance"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _MeanVariance(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [mean_variance] = self._extract(time, magnitude, error)
        return {"MeanVariance": mean_variance}
