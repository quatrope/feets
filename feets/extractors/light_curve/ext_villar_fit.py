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

from light_curve import VillarFit as _VillarFit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


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

    def __init__(
        self,
        algorithm="mcmc",
        mcmc_niter=128,
        lmsder_niter=10,
        ceres_niter=10,
        ceres_loss_reg=None,
        init=None,
        bounds=None,
        ln_prior=None,
        transform=None,
    ):
        self.algorithm = algorithm
        self.mcmc_niter = mcmc_niter
        self.lmsder_niter = lmsder_niter
        self.ceres_niter = ceres_niter
        self.ceres_loss_reg = ceres_loss_reg
        self.init = init
        self.bounds = bounds
        self.ln_prior = ln_prior
        self.transform = transform

        self._extract = _VillarFit(**self.params)

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
        ] = self._extract(time, flux, flux_error)

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
