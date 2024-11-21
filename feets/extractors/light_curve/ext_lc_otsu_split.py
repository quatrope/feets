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

from light_curve import OtsuSplit

from .utils import preprocess_data
from ..extractor import Extractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LightCurveOtsuSplit(Extractor):
    lc_feature_ext = OtsuSplit
    features = [
        "lc_otsu_mean_diff",
        "lc_otsu_std_lower",
        "lc_otsu_std_upper",
        "lc_otsu_lower_to_all_ratio",
    ]

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        time, magnitude, sigma = preprocess_data(magnitude=magnitude)

        [
            otsu_mean_diff,
            otsu_std_lower,
            otsu_std_upper,
            otsu_lower_to_all_ratio,
        ] = self.lc_feature_ext()(time, magnitude, sigma)

        return {
            "lc_otsu_mean_diff": otsu_mean_diff,
            "lc_otsu_std_lower": otsu_std_lower,
            "lc_otsu_std_upper": otsu_std_upper,
            "lc_otsu_lower_to_all_ratio": otsu_lower_to_all_ratio,
        }
