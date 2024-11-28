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

from light_curve import MedianAbsoluteDeviation as _MedianAbsoluteDeviation

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"transform": "default"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianAbsDev(LightCurveExtractor):
    features = ["MedianAbsDev"]

    def __init__(self, median_absolute_deviation_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if median_absolute_deviation_kwds is None
            else median_absolute_deviation_kwds
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [median_absolute_deviation] = _MedianAbsoluteDeviation(
            **self.lightcurve_kwds
        )(time, magnitude, error)

        return {"MedianAbsDev": median_absolute_deviation}
