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

"""Beyond-N-standard-deviation extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import BeyondNStd as _BeyondNStd

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class BeyondNStd(LightCurveExtractor):
    features = ["BeyondNStd"]

    def __init__(self, nstd=1, transform="identity"):
        self.nstd = nstd
        self.transform = transform

        self._extract = _BeyondNStd(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [beyond_n_std] = self._extract(time, magnitude, error)
        return {"BeyondNStd": beyond_n_std}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "BeyondNStd":
            N = self.nstd
            return {f"Beyond{N}Std": value}

        return super().flatten_feature(feature, value)
