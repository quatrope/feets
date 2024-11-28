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
    PercentDifferenceMagnitudePercentile as _PercentDifferenceMagnitudePercentile,
    Extractor as _Extractor,
)

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"quantile": 0.05, "transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentDiffPercentile(LightCurveExtractor):
    features = ["PercentDiffPercentile"]

    def __init__(self, percent_diff_percentile_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if percent_diff_percentile_kwds is None
            else percent_diff_percentile_kwds
        )
        self.lightcurve_kwds["quantile"] = np.atleast_1d(
            self.lightcurve_kwds["quantile"]
        )

        exts = []
        for quantile in self.lightcurve_kwds["quantile"]:
            kwds = copy.deepcopy(self.lightcurve_kwds)
            kwds["quantile"] = quantile
            exts.append(_PercentDifferenceMagnitudePercentile(**kwds))

        self.lightcurve_ext = _Extractor(*exts)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        percent_difference_magnitude_percentile = self.lightcurve_ext(
            time, magnitude, error
        )
        return {
            "PercentDiffPercentile": percent_difference_magnitude_percentile
        }

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "PercentDiffPercentile":
            names = self.lightcurve_ext.names
            percentiles = [name.split("_")[4] for name in names]
            return {
                f"PercentDiffPercentile_{p}": val
                for p, val in zip(percentiles, value)
            }

        return super().flatten_feature(feature, value)
