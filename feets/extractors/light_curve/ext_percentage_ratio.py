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
    MagnitudePercentageRatio as _MagnitudePercentageRatio,
    Extractor as _Extractor,
)

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {
    "quantile_numerator": [0.40, 0.325, 0.25, 0.175, 0.10],
    "quantile_denominator": [0.05, 0.05, 0.05, 0.05, 0.05],
    "transform": "identity",
}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentageRatio(LightCurveExtractor):
    features = ["PercentageRatio"]

    def __init__(self, percentage_ratio_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if percentage_ratio_kwds is None
            else percentage_ratio_kwds
        )
        self.lightcurve_kwds["quantile_numerator"] = np.atleast_1d(
            self.lightcurve_kwds["quantile_numerator"]
        )
        self.lightcurve_kwds["quantile_denominator"] = np.atleast_1d(
            self.lightcurve_kwds["quantile_denominator"]
        )

        if len(self.lightcurve_kwds["quantile_numerator"]) != len(
            self.lightcurve_kwds["quantile_denominator"]
        ):
            raise ValueError(
                "quantile_numerator and quantile_denominator should have the same length"
            )

        exts = []
        for numerator, denominator in zip(
            self.lightcurve_kwds["quantile_numerator"],
            self.lightcurve_kwds["quantile_denominator"],
        ):
            kwds = copy.deepcopy(self.lightcurve_kwds)
            kwds["quantile_numerator"] = numerator
            kwds["quantile_denominator"] = denominator
            exts.append(_MagnitudePercentageRatio(**kwds))

        self.lightcurve_ext = _Extractor(*exts)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        magnitude_percentage_ratio = self.lightcurve_ext(
            time, magnitude, error
        )
        return {"PercentageRatio": magnitude_percentage_ratio}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "PercentageRatio":
            names = self.lightcurve_ext.names
            numerators = [name.split("_")[3] for name in names]
            denominators = [name.split("_")[4] for name in names]
            return {
                f"PercentageRatio_{num}_{den}": val
                for num, den, val in zip(numerators, denominators, value)
            }

        return super().flatten_feature(feature, value)
