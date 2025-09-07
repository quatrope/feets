#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright (c) 2017-2024, Cabral, Juan
# Copyright (c) 2025, QuatroPe; Clariá, Felipe
# License: MIT
# Full Text:
#     https://github.com/quatrope/feets/blob/master/LICENSE


# =============================================================================
# DOC
# =============================================================================

"""Linear exponential fit extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import LinexpFit as _LinexpFit

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class LinexpFit(LightCurveExtractor):
    r"""Linexp function fit.

    Four fit parameters and goodness of fit (reduced :math:`\chi^2`) of the
    Linexp function developed for core-collapsed supernovae:

    .. math::

        f(t) = A \frac{(t-t_0)}{\tau} \times
        \exp{\left(\frac{(t-t_0)}{\tau}\right)} + B.

    Note, that the Linexp function is developed to be used with fluxes, not
    magnitudes.

    **LinexpFit_Amplitude** (:math:`A`)

    Amplitude of the Linexp function

    **LinexpFit_ReferenceTime** (:math:`t_0`)

    Reference time of the Linexp fit

    **LinexpFit_FallTime** (:math:`\tau`)

    Fall time of the Linexp function

    **LinexpFit_Baseline** (:math:`B`)

    Baseline of the Linexp function

    **LinexpFit_ReducedChi2** (reduced :math:`\chi^2`)

    Linexp fit quality

    Parameters
    ----------
    algorithm : str
        Non-linear least-square algorithm, supported values are:
        'mcmc', 'ceres', 'mcmc-ceres', 'lmsder', 'mcmc-lmsder'.
    mcmc_niter : int, optional
        Number of MCMC iterations, default is 128.
    ceres_niter : int, optional
        Number of Ceres iterations, default is 10.
    ceres_loss_reg : float, optional
        Ceres loss regularization, default is to use square norm as is, if set
        to a number, the loss function is regularized to discriminate outlier
        residuals larger than this value. Default is None which means no
        regularization.
    lmsder_niter : int, optional
        Number of LMSDER iterations, default is 10.
    init : list or None, optional
        Initial conditions, must be `None` or a `list` of `float`s or `None`s.
        The length of the list must be 4, `None` values will be replaced
        with some default values. It is supported by MCMC only.
    bounds : list of tuples or None, optional
        Boundary conditions, must be `None` or a `list` of `tuple`s of `float`s
        or `None`s. The length of the list must be 4, boundary conditions must
        include initial conditions, `None` values will be replaced with some
        broad defaults. It is supported by MCMC only.
    ln_prior : str or list of ln_prior.LnPrior1D or None, optional
        Prior for MCMC, None means no prior. It is specified by a string
        literal or a list of 5 `light_curve.ln_prior.LnPrior1D` objects, see
        `light_curve.ln_prior` submodule for corresponding functions. Available
        string literals are:

        - 'no': no prior

    transform : bool or None, optional
        If `False` or `None` (default) output is not transformed. If `True` output
        is transformed as following:

        - Half-amplitude A is transformed as :math:`zp - 2.5 lg(2*A)`,
          :math:`zp = 8.9`, so that the amplitude is assumed to be the
          object peak flux in Jy.
        - baseline flux is normalised by :math:`A: baseline -> baseline / A`
        - reference time is removed
        - goodness of fit is transformed as :math:`ln(reduced chi^2 + 1)` to
          reduce its spread
        - other parameters are not transformed

    """

    features = [
        "LinexpFit_Amplitude",
        "LinexpFit_Baseline",
        "LinexpFit_ReferenceTime",
        "LinexpFit_FallTime",
        "LinexpFit_ReducedChi2",
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

        self._extract = _LinexpFit(**self.params)

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
            reference_time,
            fall_time,
            baseline,
            reduced_chi2,
        ] = self._extract(time, flux, flux_error)

        return {
            "LinexpFit_Amplitude": amplitude,
            "LinexpFit_Baseline": baseline,
            "LinexpFit_ReferenceTime": reference_time,
            "LinexpFit_FallTime": fall_time,
            "LinexpFit_ReducedChi2": reduced_chi2,
        }
