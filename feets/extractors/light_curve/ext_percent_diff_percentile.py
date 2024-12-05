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


from light_curve import (
    PercentDifferenceMagnitudePercentile as _PercentDifferenceMagnitudePercentile,
)

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentDiffPercentile(LightCurveExtractor):
    features = ["PercentDiffPercentile"]

    def __init__(self, quantile=0.05, transform="identity"):
        self.quantile = quantile
        self.transform = transform

        self.lightcurve_ext = _PercentDifferenceMagnitudePercentile(
            **self.params
        )

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [percent_diff_percentile] = self.lightcurve_ext(time, magnitude, error)
        return {"PercentDiffPercentile": percent_diff_percentile}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "PercentDiffPercentile":
            [name] = self.lightcurve_ext.names
            percentile = name.split("_")[4]
            return {f"PercentDiffPercentile_{percentile}": value}

        return super().flatten_feature(feature, value)
