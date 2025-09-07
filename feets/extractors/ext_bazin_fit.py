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

"""Bazin fit extractor."""

# =============================================================================
# IMPORTS
# =============================================================================

from light_curve import BazinFit as _BazinFit

from .light_curve_extractor import LightCurveExtractor
from ..libs import doctools


# =============================================================================
# EXTRACTOR CLASS
# =============================================================================


class BazinFit(LightCurveExtractor):
    r"""Bazin function fit.

    Five fit parameters and goodness of fit (reduced :math:`\chi^2`) of
    the Bazin function developed for core-collapsed supernovae:

    .. math::

        f(t) = A \frac{
                \mathrm{e}^{ -(t-t_0)/\tau_\mathrm{fall} }
            }{
                1 + \mathrm{e}^{ -(t - t_0)/\tau_\mathrm{rise} }
            } + B.

    Note, that the Bazin function is developed to be used with fluxes,
    not magnitudes. Also note a typo in the Eq. (1) of the original
    paper ([bazin2009supernova]_), the minus sign is missed in the "rise"
    exponent.

    **BazinFit_Amplitude** (:math:`A`)

    Half amplitude of the Bazin function.

    **BazinFit_Baseline** (:math:`B`)

    Baseline of the Bazin function.

    **BazinFit_ReferenceTime** (:math:`t_0`)

    Reference time of the Bazin fit.

    **BazinFit_RiseTime** (:math:`\tau_\mathrm{rise}`)

    Rise time of the Bazin function.

    **BazinFit_FallTime** (:math:`\tau_\mathrm{fall}`)

    Fall time of the Bazin function.

    **BazinFit_ReducedChi2** (reduced :math:`\chi^2`)

    Bazin fit quality.

    Parameters
    ----------
    algorithm : str
        Non-linear least-square algorithm, supported values are:
        'mcmc', 'ceres', 'mcmc-ceres', 'lmsder', 'mcmc-lmsder'.
    mcmc_niter : int, optional
        Number of MCMC iterations, default is 128
    ceres_niter : int, optional
        Number of Ceres iterations, default is 10
    ceres_loss_reg : float, optional
        Ceres loss regularization, default is to use square norm as is, if set
        to a number, the loss function is regularized to discriminate outlier
        residuals larger than this value.
        Default is None which means no regularization.
    lmsder_niter : int, optional
        Number of LMSDER iterations, default is 10
    init : list or None, optional
        Initial conditions, must be `None` or a `list` of `float`s or `None`s.
        The length of the list must be 5, `None` values will be replaced
        with some default values. It is supported by MCMC only
    bounds : list of tuples or None, optional
        Boundary conditions, must be `None` or a `list` of `tuple`s of `float`s
        or `None`s. The length of the list must be 5, boundary conditions must
        include initial conditions, `None` values will be replaced with some
        broad defaults. It is supported by MCMC only
    ln_prior : str or list of ln_prior.LnPrior1D or None, optional
        Prior for MCMC, None means no prior. It is specified by a string
        literal or a list of 5 `light_curve.ln_prior.LnPrior1D` objects, see
        `light_curve.ln_prior` submodule for corresponding functions. Available
        string literals are:

        - 'no': no prior

    transform : str or bool or None, optional
        If `False` or `None` (default) output is not transformed. If `True`
        output is transformed as following:

        - Half-amplitude A is transformed as :math:`zp - 2.5 lg(2*A)`,
          :math:`zp = 8.9`, so that the amplitude is assumed to be the object
          peak flux in Jy.
        - baseline flux is normalised by :math:`A: baseline -> baseline / A`
        - reference time is removed
        - goodness of fit is transformed as :math:`ln(reduced chi^2 + 1)` to
          reduce its spread
        - other parameters are not transformed

    References
    ----------
    .. [bazin2009supernova] Bazin, G., Palanque-Delabrouille, N., Rich, J.,
       Ruhlmann-Kleider, V., Aubourg, E., Le Guillou, L., ... & Walker, E. S.
       (2009). The core-collapse rate from the Supernova Legacy Survey.
       Astronomy & Astrophysics, 499(3), 653-660.
    """

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
