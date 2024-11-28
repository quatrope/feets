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

from light_curve import (
    MedianBufferRangePercentage as _MedianBufferRangePercentage,
    Extractor as _Extractor,
)

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.10, "transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MedianBRP(LightCurveExtractor):
    features = ["MedianBRP"]

    def __init__(self, median_brp_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if median_brp_kwds is None
            else median_brp_kwds
        )
        self.lightcurve_kwds["quantile"] = np.atleast_1d(
            self.lightcurve_kwds["quantile"]
        )

        exts = []
        for quantile in self.lightcurve_kwds["quantile"]:
            kwds = copy.deepcopy(self.lightcurve_kwds)
            kwds["quantile"] = quantile
            exts.append(_MedianBufferRangePercentage(**kwds))

        self.lightcurve_ext = _Extractor(*exts)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        median_buffer_range_percentage = self.lightcurve_ext(
            time, magnitude, error
        )
        return {"MedianBRP": median_buffer_range_percentage}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "MedianBRP":
            names = self.lightcurve_ext.names
            percentiles = [name.split("_")[4] for name in names]
            return {
                f"MedianBRP_{p}": val for p, val in zip(percentiles, value)
            }

        return super().flatten_feature(feature, value)
