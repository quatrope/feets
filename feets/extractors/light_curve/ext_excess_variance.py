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

from light_curve import ExcessVariance as _ExcessVariance

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class ExcessVariance(LightCurveExtractor):
    features = ["ExcessVariance"]

    def __init__(self, excess_variance_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if excess_variance_kwds is None
            else excess_variance_kwds
        )
        self.lightcurve_ext = _ExcessVariance(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        [excess_variance] = self.lightcurve_ext(time, magnitude, error)
        return {"ExcessVariance": excess_variance}
