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

"""Percentage ratio extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import (
    MagnitudePercentageRatio as _MagnitudePercentageRatio,
)

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools

# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class PercentageRatio(LightCurveExtractor):
    features = ["PercentageRatio"]

    def __init__(
        self,
        quantile_numerator=0.40,
        quantile_denominator=0.05,
        transform="identity",
    ):
        self.quantile_numerator = quantile_numerator
        self.quantile_denominator = quantile_denominator
        self.transform = transform

        self._extract = _MagnitudePercentageRatio(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [percentage_ratio] = self._extract(time, magnitude, error)
        return {"PercentageRatio": percentage_ratio}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "PercentageRatio":
            [name] = self._extract.names
            split_name = name.split("_")
            numerator, denominator = split_name[3], split_name[3]
            return {f"PercentageRatio_{numerator}_{denominator}": value}

        return super().flatten_feature(feature, value)
