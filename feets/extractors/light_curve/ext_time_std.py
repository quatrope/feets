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

from light_curve import TimeStandardDeviation as _TimeStandardDeviation

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class TimeStd(LightCurveExtractor):
    features = ["TimeStd"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _TimeStandardDeviation(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude=None, error=None):
        [time_std] = self._extract(time, magnitude, error)
        return {"TimeStd": time_std}
