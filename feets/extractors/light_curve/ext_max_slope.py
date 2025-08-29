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

"""Maximum slope extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import MaximumSlope as _MaximumSlope

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class MaxSlope(LightCurveExtractor):
    features = ["MaxSlope"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _MaximumSlope(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, magnitude, error=None):
        """
        Parameters
        ----------
        time : array-like
        magnitude : array-like
        error : array-like, optional
        """
        [maximum_slope] = self._extract(time, magnitude, error)
        return {"MaxSlope": maximum_slope}
