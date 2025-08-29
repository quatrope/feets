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

"""Skew extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import Skew as _Skew

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Skew(LightCurveExtractor):
    features = ["Skew"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _Skew(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        """
        Parameters
        ----------
        magnitude : array-like
        time : array-like, optional
        error : array-like, optional
        """
        [skew] = self._extract(time, magnitude, error)
        return {"Skew": skew}
