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

from light_curve import LinearFit as _LinearFit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearFit(LightCurveExtractor):
    features = [
        "LinearFit_Slope",
        "LinearFit_Sigma",
        "LinearFit_ReducedChi2",
    ]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _LinearFit(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error):
        [slope, slope_sigma, reduced_chi2] = self._extract(
            time, magnitude, error
        )
        return {
            "LinearFit_Slope": slope,
            "LinearFit_Sigma": slope_sigma,
            "LinearFit_ReducedChi2": reduced_chi2,
        }
