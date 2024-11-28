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

from light_curve import OtsuSplit as _OtsuSplit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class OtsuSplit(LightCurveExtractor):
    features = [
        "OtsuMeanDiff",
        "OtsuStdLower",
        "OtsuStdUpper",
        "OtsuLowerToAllRatio",
    ]

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        [
            otsu_mean_diff,
            otsu_std_lower,
            otsu_std_upper,
            otsu_lower_to_all_ratio,
        ] = _OtsuSplit()(time, magnitude, error)

        return {
            "OtsuMeanDiff": otsu_mean_diff,
            "OtsuStdLower": otsu_std_lower,
            "OtsuStdUpper": otsu_std_upper,
            "OtsuLowerToAllRatio": otsu_lower_to_all_ratio,
        }
