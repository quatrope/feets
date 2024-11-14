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

from light_curve import MagnitudePercentageRatio

import numpy as np

from .extractor import Extractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Lightcurve_MagnitudePercentageRatio(Extractor):
    features = [
        "lightcurve_MagnitudePercentageRatio",
    ]

    def __init__(
        self,
        quantile_numerator=0.40,
        quantile_denominator=0.05,
        lightcurve_kwds=None,
    ):
        self.quantile_numerator = quantile_numerator
        self.quantile_denominator = quantile_denominator
        self.lightcurve_kwds = lightcurve_kwds or {}

    @doctools.doc_inherit(Extractor.extract)
    def extract(self, magnitude):
        magnitude = np.array(magnitude, dtype=np.float64)
        time = np.zeros_like(magnitude)

        feature = MagnitudePercentageRatio(
            quantile_numerator=self.quantile_numerator,
            quantile_denominator=self.quantile_denominator,
            **self.lightcurve_kwds,
        )

        [result] = feature(time, magnitude)

        return {
            "lightcurve_MagnitudePercentageRatio": result,
        }
