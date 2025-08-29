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

"""Bazin fit extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import BazinFit as _BazinFit

from .light_curve_extractor import LightCurveExtractor
from ...libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class BazinFit(LightCurveExtractor):
    features = [
        "BazinFit_Amplitude",
        "BazinFit_Baseline",
        "BazinFit_ReferenceTime",
        "BazinFit_RiseTime",
        "BazinFit_FallTime",
        "BazinFit_ReducedChi2",
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

        self._extract = _BazinFit(**self.params)

    @doctools.doc_inherit(LightCurveExtractor.extract)
    def extract(self, time, flux, flux_error):
        """
        Parameters
        ----------
        time : array-like
        flux : array-like
        flux_error : array-like
        """
        [
            amplitude,
            baseline,
            reference_time,
            rise_time,
            fall_time,
            reduced_chi2,
        ] = self._extract(time, flux, flux_error)

        return {
            "BazinFit_Amplitude": amplitude,
            "BazinFit_Baseline": baseline,
            "BazinFit_ReferenceTime": reference_time,
            "BazinFit_RiseTime": rise_time,
            "BazinFit_FallTime": fall_time,
            "BazinFit_ReducedChi2": reduced_chi2,
        }
