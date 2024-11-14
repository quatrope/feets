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

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_OtsuSplit(Extractor):
    features = [
        "lightcurve_Otsu_mean_diff",
        "lightcurve_Otsu_std_lower",
        "lightcurve_Otsu_std_upper",
        "lightcurve_Otsu_lower_to_all_ratio",
    ]

    def __init__(self, lightcurve_kwds=None):
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        magnitude = np.array(magnitude, dtype=np.float64)
        time = np.zeros_like(magnitude)

        feature = OtsuSplit(**self.lightcurve_kwds)
        [mean_diff, std_lower, std_upper, lower_to_all_ratio] = feature(
            time, magnitude
        )

        return {
            "lightcurve_Otsu_mean_diff": mean_diff,
            "lightcurve_Otsu_std_lower": std_lower,
            "lightcurve_Otsu_std_upper": std_upper,
            "lightcurve_Otsu_lower_to_all_ratio": lower_to_all_ratio,
        }
