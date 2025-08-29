#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2024, QuatroPe, Felipe Clariá
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

"""Linear trend extractor."""


# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import LinearTrend as _LinearTrend

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinearTrend(LightCurveExtractor):
    features = [
        "LinearTrend",
        "LinearTrend_Sigma",
        "LinearTrend_ReducedChi2",
    ]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _LinearTrend(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        """
        Parameters
        ----------
        time : array-like
        magnitude : array-like
        error : array-like, optional
        """
        [linear_trend, linear_trend_sigma, reduced_chi2] = self._extract(
            time, magnitude, error
        )
        return {
            "LinearTrend": linear_trend,
            "LinearTrend_Sigma": linear_trend_sigma,
            "LinearTrend_ReducedChi2": reduced_chi2,
        }
