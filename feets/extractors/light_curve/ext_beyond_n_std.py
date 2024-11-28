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

import copy

from light_curve import BeyondNStd as _BeyondNStd, Extractor as _Extractor

import numpy as np

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {"nstd": 1, "transform": "identity"}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class BeyondNStd(LightCurveExtractor):
    features = ["BeyondNStd"]

    def __init__(self, beyond_n_std_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if beyond_n_std_kwds is None
            else beyond_n_std_kwds
        )
        self.lightcurve_kwds["nstd"] = np.atleast_1d(
            self.lightcurve_kwds["nstd"]
        )

        exts = []
        for nstd in self.lightcurve_kwds["nstd"]:
            kwds = copy.deepcopy(self.lightcurve_kwds)
            kwds["nstd"] = nstd
            exts.append(_BeyondNStd(**kwds))

        self.lightcurve_ext = _Extractor(*exts)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, magnitude, time=None, error=None):
        beyond_n_std = self.lightcurve_ext(time, magnitude, error)
        return {"BeyondNStd": beyond_n_std}

    @doctools.doc_inherit(LightCurveExtractor.flatten_feature)
    def flatten_feature(self, feature, value):
        if feature == "BeyondNStd":
            names = self.lightcurve_ext.names
            Ns = [name.split("_")[1] for name in names]
            return {f"Beyond{N}Std": val for N, val in zip(Ns, value)}

        return super().flatten_feature(feature, value)
