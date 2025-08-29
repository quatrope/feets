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

"""Roms extractor."""

# =============================================================================
# IMPORTS
# =============================================================================


from light_curve import Roms as _Roms

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class Roms(LightCurveExtractor):
    features = ["Roms"]

    def __init__(self, transform="identity"):
        self.transform = transform
        self._extract = _Roms(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, error, time=None):
        """
        Parameters
        ----------
        magnitude : array-like
        error : array-like
        time : array-like, optional
        """
        [roms] = self._extract(time, magnitude, error)
        return {"Roms": roms}
