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

from light_curve import VillarFit as _VillarFit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# CONSTANTS
# =============================================================================

LIGHTCURVE_KWDS = {
    "algorithm": "mcmc",
    "mcmc_niter": 128,
    "lmsder_niter": 10,
    "ceres_niter": 10,
    "ceres_loss_reg": None,
    "init": None,
    "bounds": None,
    "ln_prior": None,
    "transform": None,
}


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class VillarFit(LightCurveExtractor):
    features = [
        "VillarFit_Amplitude",
        "VillarFit_Baseline",
        "VillarFit_ReferenceTime",
        "VillarFit_RiseTime",
        "VillarFit_FallTime",
        "VillarFit_PlateauRelAmplitude",
        "VillarFit_PlateauDuration",
        "VillarFit_ReducedChi2",
    ]

    def __init__(self, linear_fit_kwds=None):
        self.lightcurve_kwds = (
            copy.deepcopy(LIGHTCURVE_KWDS)
            if linear_fit_kwds is None
            else linear_fit_kwds
        )
        self.lightcurve_ext = _VillarFit(**self.lightcurve_kwds)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, flux, flux_error):
        [
            amplitude,
            baseline,
            reference_time,
            rise_time,
            fall_time,
            plateau_rel_amplitude,
            plateau_duration,
            reduced_chi2,
        ] = self.lightcurve_ext(time, flux, flux_error)
        return {
            "VillarFit_Amplitude": amplitude,
            "VillarFit_Baseline": baseline,
            "VillarFit_ReferenceTime": reference_time,
            "VillarFit_RiseTime": rise_time,
            "VillarFit_FallTime": fall_time,
            "VillarFit_PlateauRelAmplitude": plateau_rel_amplitude,
            "VillarFit_PlateauDuration": plateau_duration,
            "VillarFit_ReducedChi2": reduced_chi2,
        }
